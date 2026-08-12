//! Training ops, and the label sweep that feeds them.
//!
//! `train_drafter` and `train_lora` are the daemon's only micro-step-preemptible
//! handlers: each request advances one resident session by one quantum, keyed by
//! `run_id`, and returns a terminal `train_done` or a non-terminal
//! `train_progress`. The yield itself is driven by the caller re-enqueueing, not
//! by anything here. Note the quantum unit differs — steps for LoRA, EPOCHS for
//! the drafter — which is why the drafter's worst-case block is much larger.
//!
//! Sessions are in-memory only, so neither survives a daemon restart.

// Handler bodies were lifted verbatim out of `main()`, so they depend on the same
// root-level imports and arch aliases (`qwen35`, `deepseek4`, `minimax`, `lfm2moe`,
// `qwen2`, `prompt_frame`) that the crate root sets up. Glob-importing the root
// keeps that dependency in one place instead of re-deriving it per module.
use crate::*;

pub(crate) fn pflash_labels(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    let Some(corpus) = msg.get("corpus").and_then(|v| v.as_str()).map(String::from) else {
        daemon_state
            .out
            .error("pflash_labels: missing 'corpus'".to_string());
        return;
    };
    let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from) else {
        daemon_state
            .out
            .error("pflash_labels: missing 'output'".to_string());
        return;
    };
    let seq = msg
        .get("seq")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(512);
    let block = msg
        .get("block")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(64);
    let n_chunks = msg
        .get("n_chunks")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(40);
    let Some(m) = daemon_state.model.as_ref() else {
        daemon_state
            .out
            .error("pflash_labels: no model loaded".to_string());
        return;
    };
    let (Some(weights), Some(config), Some(tokenizer)) = (
        m.q35_weights.as_ref(),
        m.q35_config.as_ref(),
        m.tokenizer.as_ref(),
    ) else {
        daemon_state
            .out
            .error("pflash_labels: resident model is not a qwen3.5-family model".to_string());
        return;
    };
    let fa = qwen35::full_attention_layers(config);
    if fa.is_empty() {
        daemon_state
            .out
            .error("pflash_labels: no FullAttention layers".to_string());
        return;
    }
    let shallow = fa[0];
    let mid = fa[fa.len() / 2];
    let text = match std::fs::read_to_string(&corpus) {
        Ok(t) => t,
        Err(e) => {
            daemon_state
                .out
                .error(format!("pflash_labels: read {corpus}: {e}"));
            return;
        }
    };
    let all = tokenizer.encode(&text);
    if all.len() < n_chunks * seq {
        daemon_state.out.error(format!(
            "pflash_labels: corpus too small: {} toks < {}",
            all.len(),
            n_chunks * seq
        ));
        return;
    }
    let mut out_file = match std::fs::File::create(&output) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(e) => {
            daemon_state
                .out
                .error(format!("pflash_labels: create {output}: {e}"));
            return;
        }
    };
    let mut failed = false;
    for ci in 0..n_chunks {
        let toks = all[ci * seq..(ci + 1) * seq].to_vec();
        match qwen35::capture_pflash_block_scores(
            &mut daemon_state.gpu,
            weights,
            config,
            &toks,
            block,
            &[shallow, mid],
        ) {
            Ok(scores) => {
                let line = serde_json::json!({
                    "chunk": ci,
                    "tokens": toks,
                    "shallow_scores": scores[0],
                    "mid_scores": scores[1],
                });
                if writeln!(out_file, "{line}").is_err() {
                    failed = true;
                    break;
                }
            }
            Err(e) => {
                daemon_state
                    .out
                    .error(format!("pflash_labels: chunk {ci}: {e}"));
                failed = true;
                break;
            }
        }
    }
    use std::io::Write as _;
    let _ = out_file.flush();
    if failed {
        return;
    }
    // Dump the shared fp32 embedding once (the drafter shares it RO).
    let embed_path = format!("{output}.embed.bin");
    let embed_dims = match qwen35::dump_embed_fp32(
        &mut daemon_state.gpu,
        weights,
        config,
        std::path::Path::new(&embed_path),
    ) {
        Ok(d) => Some(d),
        Err(e) => {
            daemon_state.out.error(format!("pflash_labels: embed: {e}"));
            None
        }
    };
    let resp = serde_json::json!({
        "type": "pflash_labels",
        "output": output,
        "embed": embed_dims.map(|_| embed_path.clone()),
        "embed_vocab": embed_dims.map(|(v, _)| v),
        "embed_dim": embed_dims.map(|(_, d)| d),
        "n_chunks": n_chunks,
        "seq": seq,
        "block": block,
        "shallow_layer": shallow,
        "mid_layer": mid,
    });
    daemon_state.out.emit(resp);
}

pub(crate) fn train_drafter(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    // Micro-step-PREEMPTIBLE SSM-drafter training as a daemon op. Runs
    // up to `quantum` EPOCHS per request and keeps a resident
    // DrafterTrainSession alive between requests (keyed by `run_id`);
    // the runner re-enqueues the low-priority training lease each
    // quantum so it time-slices with interactive serving. Numerics are
    // verbatim from the whole-run loop (drafter_loop_init/run_epochs/
    // finish reproduce train_ssm_drafter_loop). Per-eval-epoch stream
    // uses type `train_epoch` (not `train_progress`) so the runner's
    // adapter only sees ONE quantum-boundary `train_progress`/`train_done`.
    let run_id = msg
        .get("run_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let t = msg
        .get("train")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let quantum = msg
        .get("quantum")
        .or_else(|| t.get("quantum"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(25)
        .max(1);

    // CONTINUE the resident session iff its run_id matches; else START
    // fresh (loading labels + building drafter/optimizer once).
    let continue_run = !run_id.is_empty()
        && daemon_state
            .drafter_train_session
            .as_ref()
            .map(|s| s.run_id == run_id)
            .unwrap_or(false);
    if !continue_run {
        daemon_state.drafter_train_session = None; // drop any stale session, free VRAM
        let arch = msg
            .get("arch")
            .and_then(|v| v.as_str())
            .unwrap_or("ssm")
            .to_string();
        if arch != "ssm" {
            daemon_state.out.error(format!(
                "train_drafter: arch '{arch}' not implemented (only ssm; step 3)"
            ));
            return;
        }
        // Parse the train/labels blocks into the SHARED TrainCfg.
        let labels = msg
            .get("labels")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let getu = |o: &serde_json::Value, k: &str, d: usize| -> usize {
            o.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let getf = |o: &serde_json::Value, k: &str, d: f32| -> f32 {
            o.get(k)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(d)
        };
        let cfg = hipfire_train::train_loop::TrainCfg {
            seq: getu(&labels, "seq", 512),
            block: getu(&labels, "block", 64),
            n_eval: getu(&labels, "n_eval", 20),
            epochs: getu(&t, "epochs", 300),
            lr: getf(&t, "lr", 1e-3),
            wd: getf(&t, "wd", 0.0),
            tau: getf(&t, "tau", 0.1),
            eval_every: getu(&t, "eval_every", 15),
            report_train: t
                .get("report_train")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        };
        let source = labels
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("file");
        if source != "file" {
            daemon_state.out.error(format!("train_drafter: label source '{source}' not implemented (only file; capture is step 4)"),
            );
            return;
        }
        let Some(path) = labels
            .get("path")
            .and_then(|v| v.as_str())
            .map(String::from)
        else {
            daemon_state
                .out
                .error("train_drafter: labels.path required for source=file".to_string());
            return;
        };
        let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from) else {
            daemon_state
                .out
                .error("train_drafter: 'output' (checkpoint path) required".to_string());
            return;
        };

        // ── load cached labels + frozen target embedding (file source) ──
        let mut ls = match hipfire_train::labels::load_daemon_labels(
            &mut daemon_state.gpu,
            &path,
            cfg.seq,
        ) {
            Ok(ls) => ls,
            Err(e) => {
                daemon_state
                    .out
                    .error(format!("train_drafter: load labels {path}: {e}"));
                return;
            }
        };
        let shuffle_seed = getu(&labels, "shuffle_seed", 0x5EED) as u64;
        hipfire_train::labels::shuffle_in_place(
            &mut ls.chunks,
            &mut ls.label_mid,
            &mut ls.base_shallow,
            shuffle_seed,
        );

        // ── build the SSM drafter from the request config ──
        let dc = msg
            .get("config")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let mut dcfg = hipfire_train::ssm_drafter::SsmDrafterConfig::tiny(10000.0, 1e-5);
        dcfg.h_draft = getu(&dc, "h_draft", 512);
        dcfg.n_layers = getu(&dc, "n_layers", 3);
        dcfg.inter = getu(&dc, "inter", 1024);
        dcfg.n_kv = getu(&dc, "n_kv", 4);
        dcfg.head_dim = getu(&dc, "head_dim", 64);
        let (h_t, vocab) = (ls.h_t, ls.vocab);
        let drafter = match hipfire_train::ssm_drafter::SsmDrafter::new(
            &mut daemon_state.gpu,
            ls.embed,
            h_t,
            vocab,
            dcfg,
            cfg.seq,
        ) {
            Ok(d) => d,
            Err(e) => {
                daemon_state
                    .out
                    .error(format!("train_drafter: build drafter: {e}"));
                return;
            }
        };
        // Set up the resumable loop state (bar, optimizer, scores scratch).
        let st = match hipfire_train::train_loop::drafter_loop_init(
            &mut daemon_state.gpu,
            &drafter,
            &ls.chunks,
            &ls.label_mid,
            &ls.base_shallow,
            &cfg,
        ) {
            Ok(s) => s,
            Err(e) => {
                daemon_state
                    .out
                    .error(format!("train_drafter: loop init: {e}"));
                return;
            }
        };
        let nparams: usize = drafter.param_sizes().iter().sum();
        let _ = writeln!(
            daemon_state.out.sink,
            "{}",
            serde_json::json!({
                "type": "train_start", "arch": arch, "params": nparams,
                "chunks": ls.chunks.len(), "n_train": ls.chunks.len().saturating_sub(cfg.n_eval),
                "n_eval": cfg.n_eval, "epochs": cfg.epochs,
                "run_id": run_id, "quantum": quantum,
            })
        );
        let _ = daemon_state.out.sink.flush();
        daemon_state.drafter_train_session = Some(DrafterTrainSession {
            run_id: run_id.clone(),
            drafter,
            chunks: ls.chunks,
            label_mid: ls.label_mid,
            cfg,
            st,
            output,
            quantum,
        });
    }

    // ── run ONE quantum of epochs, streaming per-epoch `train_epoch` ──
    let quantum_result: Result<(), String> = {
        let sess = daemon_state
            .drafter_train_session
            .as_mut()
            .expect("session present after start/return");
        let ep_end = (sess.st.ep + sess.quantum).min(sess.cfg.epochs);
        hipfire_train::train_loop::drafter_loop_run_epochs(
            &mut daemon_state.gpu,
            &sess.drafter,
            sess.chunks.as_slice(),
            sess.label_mid.as_slice(),
            &sess.cfg,
            &mut sess.st,
            ep_end,
            |ep, train_loss, corr, best, best_ep, train_corr| {
                let mut ev = serde_json::json!({
                    "type": "train_epoch", "epoch": ep, "train_loss": train_loss,
                    "eval": corr, "best": best, "best_epoch": best_ep,
                });
                if let Some(tc) = train_corr {
                    ev["train_rho"] = serde_json::json!(tc);
                }
                daemon_state.out.emit(ev);
            },
        )
        .map_err(|e| e.to_string())
    };
    if let Err(e) = quantum_result {
        daemon_state.drafter_train_session = None;
        daemon_state
            .out
            .error(format!("train_drafter: train loop: {e}"));
        return;
    }

    let done = daemon_state
        .drafter_train_session
        .as_ref()
        .map(|s| s.st.ep >= s.cfg.epochs)
        .unwrap_or(false);
    if done {
        // Final quantum: finish (free scratch) → checkpoint best-eval
        // weights → terminal event. `take()` drops the resident session.
        let sess = daemon_state
            .drafter_train_session
            .take()
            .expect("done implies present");
        let output = sess.output.clone();
        let run_id = sess.run_id.clone();
        let report = hipfire_train::train_loop::drafter_loop_finish(&mut daemon_state.gpu, sess.st);
        let saved = hipfire_train::labels::save_ssm_drafter_weights(
            &output,
            &report.best_weights,
            report.best_epoch as u32,
        );
        let _ = writeln!(
            daemon_state.out.sink,
            "{}",
            serde_json::json!({
                "type": "train_done",
                "best_eval": report.best_eval, "best_epoch": report.best_epoch,
                "bar": report.bar, "final_eval": report.final_eval,
                "beat_bar": report.best_eval > report.bar,
                "checkpoint": if saved.is_ok() { Some(output.clone()) } else { None },
                "checkpoint_error": saved.err().map(|e| e.to_string()),
                "run_id": run_id,
            })
        );
        let _ = daemon_state.out.sink.flush();
    } else {
        // Quantum done but run unfinished: report progress and keep the
        // session resident. The runner re-enqueues; training yields to
        // any pending interactive request before the next quantum.
        let sess = daemon_state
            .drafter_train_session
            .as_ref()
            .expect("unfinished implies present");
        let _ = writeln!(
            daemon_state.out.sink,
            "{}",
            serde_json::json!({
                "type": "train_progress", "run_id": sess.run_id,
                "epoch": sess.st.ep, "total": sess.cfg.epochs,
                "eval": sess.st.final_eval, "best": sess.st.best_eval,
                "done": false,
            })
        );
        let _ = daemon_state.out.sink.flush();
    }
}

pub(crate) fn train_lora(daemon_state: &mut DaemonState, msg: &serde_json::Value) {
    // Micro-step-PREEMPTIBLE LoRA-on-frozen training as a daemon op.
    // Runs up to `quantum` steps per request and keeps a resident
    // LoraTrainSession alive between requests (keyed by `run_id`); the
    // runner re-enqueues the low-priority training lease each quantum so
    // it time-slices with interactive serving. Compute is verbatim from
    // the validated whole-run loop (hipfire_train, overfit_supra50m.rs):
    // forward → loss → backward-THROUGH-ADAPTERS → AdamW, then a final
    // HFLORA01 adapter dump. NOTE: trains hipfire-train's own un-fused
    // LlamaModel, NOT the served qwen35 adapters (a follow-on).
    // `data=overfit` is a deterministic synthetic batch.
    const IGNORE: i32 = -100;
    let run_id = msg
        .get("run_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let train = msg
        .get("train")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let quantum = msg
        .get("quantum")
        .or_else(|| train.get("quantum"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(25)
        .max(1);

    // CONTINUE the resident session iff its run_id matches; else START
    // fresh (loading the model + building the batch/optimizer once).
    let continue_run = !run_id.is_empty()
        && daemon_state
            .lora_train_session
            .as_ref()
            .map(|s| s.run_id == run_id)
            .unwrap_or(false);
    if !continue_run {
        daemon_state.lora_train_session = None; // drop any stale session, free VRAM
        let Some(output) = msg.get("output").and_then(|v| v.as_str()).map(String::from) else {
            daemon_state
                .out
                .error("train_lora: 'output' (adapter checkpoint path) required".to_string());
            return;
        };
        let Some(base_dir) = msg
            .get("model")
            .or_else(|| msg.get("base"))
            .and_then(|v| v.as_str())
            .map(String::from)
        else {
            daemon_state
                .out
                .error("train_lora: 'model' (fp32 base model dir) required".to_string());
            return;
        };
        let getu = |k: &str, d: usize| -> usize {
            train
                .get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let getf = |k: &str, d: f32| -> f32 {
            train
                .get(k)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(d)
        };
        let steps = getu("steps", 200);
        let rank = getu("rank", 16);
        let seq = getu("seq", 8);
        let n_seqs = getu("n_seqs", 3);
        let alpha = getf("alpha", 32.0);
        let lr = getf("lr", 5e-3);
        let data_mode = msg
            .get("data")
            .and_then(|v| v.as_str())
            .unwrap_or("overfit");
        if data_mode != "overfit" {
            daemon_state.out.error(format!("train_lora: data source '{data_mode}' not implemented (only 'overfit' synthetic batch is wired; real-corpus loading is a follow-on)"),
            );
            return;
        }
        let _ = writeln!(
            daemon_state.out.sink,
            "{}",
            serde_json::json!({
                "type": "train_start", "op": "train_lora", "base": base_dir,
                "steps": steps, "rank": rank, "alpha": alpha, "lr": lr,
                "run_id": run_id, "quantum": quantum,
            })
        );
        let _ = daemon_state.out.sink.flush();
        let built: Result<LoraTrainSession, String> = (|| {
            let dir = std::path::Path::new(&base_dir);
            if !dir.exists() {
                return Err(format!("base model dir not found: {base_dir}"));
            }
            let (cfg, weights) = hipfire_train::loader::load_llama_fp32(&mut daemon_state.gpu, dir)
                .map_err(|e| e.to_string())?;
            let vocab = cfg.vocab_size;
            let model = hipfire_train::model::LlamaModel::from_f32_weights(
                &mut daemon_state.gpu,
                &cfg,
                weights,
                seq,
                rank,
                alpha,
            )
            .map_err(|e| e.to_string())?;
            let pos: Vec<f32> = (0..seq).map(|t| t as f32).collect();
            let batch: Vec<(Vec<u32>, Vec<f32>)> = (0..n_seqs)
                .map(|s| {
                    let toks: Vec<u32> = (0..seq)
                        .map(|t| (((t + 1) * 2654435761 + s * 40503) % vocab) as u32)
                        .collect();
                    let mut tgts: Vec<f32> = (0..seq).map(|t| toks[(t + 1) % seq] as f32).collect();
                    tgts[seq - 1] = IGNORE as f32;
                    (toks, tgts)
                })
                .collect();
            let target_tokens = (n_seqs * (seq - 1)).max(1) as f32;
            let sizes = model.lora_param_sizes();
            let opt = hipfire_train::optim::AdamW::new(
                &mut daemon_state.gpu,
                &sizes,
                lr,
                0.9,
                0.999,
                1e-8,
                0.0,
            )
            .map_err(|e| e.to_string())?;
            // total = steps + 1: the final pass is eval-only (no update),
            // matching the validated whole-run `for step in 0..=steps`.
            Ok(LoraTrainSession {
                run_id: run_id.clone(),
                model,
                opt,
                batch,
                pos,
                target_tokens,
                step: 0,
                total: steps + 1,
                initial_ce: 0.0,
                last_ce: 0.0,
                output,
                vocab,
            })
        })();
        match built {
            Ok(sess) => daemon_state.lora_train_session = Some(sess),
            Err(e) => {
                daemon_state.out.error(format!("train_lora: {e}"));
                return;
            }
        }
    }

    // Run ONE quantum of steps on the resident session. Destructure the
    // &mut session into disjoint field bindings so the per-step
    // forward/backward (reads `model`) and `opt.step` (mut `opt`) don't
    // trip the borrow checker through a single `sess`.
    let quantum_result: Result<(), String> = {
        let sess = daemon_state
            .lora_train_session
            .as_mut()
            .expect("session present after start/return");
        let LoraTrainSession {
            model,
            opt,
            batch,
            pos,
            target_tokens,
            step,
            total,
            initial_ce,
            last_ce,
            ..
        } = sess;
        (|| {
            let end = (*step + quantum).min(*total);
            while *step < end {
                let s = *step;
                let mut total_loss = 0.0f32;
                for (toks, tgts) in batch.iter() {
                    let acts = hipfire_train::model::model_forward(
                        &mut daemon_state.gpu,
                        &*model,
                        toks,
                        pos.as_slice(),
                    )
                    .map_err(|e| e.to_string())?;
                    let (loss, grads) = hipfire_train::model::model_loss_backward(
                        &mut daemon_state.gpu,
                        &*model,
                        &acts,
                        tgts,
                        IGNORE,
                    )
                    .map_err(|e| e.to_string())?;
                    total_loss += loss;
                    // Last pass (step == total-1) is eval-only.
                    if s < *total - 1 {
                        let params = model.lora_params();
                        let gflat = hipfire_train::model::flatten_lora_grads(&grads);
                        opt.step(&mut daemon_state.gpu, &params, &gflat)
                            .map_err(|e| e.to_string())?;
                    }
                    // Free per-step activations + grads. model_forward /
                    // model_loss_backward allocate fresh GPU scratch each
                    // step and neither frees it; without this the resident
                    // session leaks VRAM across steps → OOM after a few
                    // hundred steps (the overfit example only "works"
                    // because it runs alone on a big-VRAM box).
                    hipfire_train::model::free_model_acts(&mut daemon_state.gpu, acts)
                        .map_err(|e| e.to_string())?;
                    for g in grads {
                        daemon_state
                            .gpu
                            .free_tensor(g.daq)
                            .map_err(|e| e.to_string())?;
                        daemon_state
                            .gpu
                            .free_tensor(g.dbq)
                            .map_err(|e| e.to_string())?;
                        daemon_state
                            .gpu
                            .free_tensor(g.dav)
                            .map_err(|e| e.to_string())?;
                        daemon_state
                            .gpu
                            .free_tensor(g.dbv)
                            .map_err(|e| e.to_string())?;
                        daemon_state
                            .gpu
                            .free_tensor(g.dnorm1)
                            .map_err(|e| e.to_string())?;
                        daemon_state
                            .gpu
                            .free_tensor(g.dnorm2)
                            .map_err(|e| e.to_string())?;
                    }
                }
                *last_ce = total_loss / *target_tokens;
                if s == 0 {
                    *initial_ce = *last_ce;
                }
                *step += 1;
            }
            Ok(())
        })()
    };
    if let Err(e) = quantum_result {
        daemon_state.lora_train_session = None;
        daemon_state.out.error(format!("train_lora: {e}"));
        return;
    }

    let done = daemon_state
        .lora_train_session
        .as_ref()
        .map(|s| s.step >= s.total)
        .unwrap_or(false);
    if done {
        // Final quantum: dump the adapter and finish. `take()` drops the
        // resident session (frees VRAM) before we emit the terminal event.
        let sess = daemon_state
            .lora_train_session
            .take()
            .expect("done implies present");
        // Persist the trained adapter: layer-major [aq,bq,av,bv] f32
        // tensors. Minimal container (magic + count + per-tensor
        // shape/data) — a serving-loadable format is a follow-on.
        let dump: Result<usize, String> = (|| {
            let params = sess.model.lora_params();
            let mut buf: Vec<u8> = Vec::new();
            buf.extend_from_slice(b"HFLORA01");
            buf.extend_from_slice(&(params.len() as u32).to_le_bytes());
            for t in &params {
                let data = daemon_state
                    .gpu
                    .download_f32(t)
                    .map_err(|e| e.to_string())?;
                buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
                for &d in &t.shape {
                    buf.extend_from_slice(&(d as u32).to_le_bytes());
                }
                buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
                for &f in &data {
                    buf.extend_from_slice(&f.to_le_bytes());
                }
            }
            std::fs::write(&sess.output, &buf)
                .map_err(|e| format!("write adapter {}: {e}", sess.output))?;
            Ok(params.len())
        })();
        match dump {
            Ok(n_trainable) => {
                let _ = writeln!(
                    daemon_state.out.sink,
                    "{}",
                    serde_json::json!({
                        "type": "train_done", "op": "train_lora",
                        "initial_per_tok_ce": sess.initial_ce,
                        "final_per_tok_ce": sess.last_ce,
                        "steps": sess.total - 1, "trainable_tensors": n_trainable,
                        "baseline_ce_ln_vocab": (sess.vocab as f32).ln(),
                        "output": sess.output, "run_id": sess.run_id,
                        "note": "trained hipfire-train LlamaModel LoRA (overfit synthetic batch); served-qwen35 adapters + real-corpus loading are follow-ons",
                    })
                );
                let _ = daemon_state.out.sink.flush();
            }
            Err(e) => daemon_state.out.error(format!("train_lora: {e}")),
        }
    } else {
        // Quantum done but run unfinished: report progress and keep the
        // session resident. The runner re-enqueues; training yields to
        // any pending interactive request before the next quantum.
        let sess = daemon_state
            .lora_train_session
            .as_ref()
            .expect("unfinished implies present");
        let _ = writeln!(
            daemon_state.out.sink,
            "{}",
            serde_json::json!({
                "type": "train_progress", "run_id": sess.run_id,
                "step": sess.step, "total": sess.total,
                "per_tok_ce": sess.last_ce, "done": false,
            })
        );
        let _ = daemon_state.out.sink.flush();
    }
}
