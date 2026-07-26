// SPDX-License-Identifier: Apache-2.0
//! Shared drafter training loop — ListNet top-1 ranking loss + AdamW, against
//! pre-captured (or in-process) PFlash block-importance labels.
//!
//! ONE loop, called by both the standalone `ssm_drafter_train` example and the
//! daemon `train_drafter` op (see docs/plans/2026-06-19-train-as-daemon-op.md).
//! The caller owns label sourcing / shuffling / IO; this owns the epoch loop,
//! the loss, the eval metric, and best-eval tracking. Progress is surfaced via
//! the `on_epoch` callback (the example prints; the daemon emits JSONL).

use crate::ops::pflash_score::pflash_score_forward;
use crate::optim::AdamW;
use crate::ssm_drafter::{
    free_ssm_drafter_acts, free_ssm_drafter_grads, ssm_drafter_backward, ssm_drafter_forward_train,
    SsmDrafter,
};
use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

/// Training hyperparameters. The eval split is the LAST `n_eval` chunks (the
/// caller is expected to have shuffled, so that tail is a random hold-out).
#[derive(Clone, Copy)]
pub struct TrainCfg {
    pub seq: usize,
    pub block: usize,
    pub n_eval: usize,
    pub epochs: usize,
    pub lr: f32,
    pub wd: f32,
    pub tau: f32,
    pub eval_every: usize,
    /// Also measure mean Spearman over the TRAIN split each eval epoch (oracle
    /// diagnostic: bounds representational capacity / cosine-K-head fit). Costs an
    /// extra forward over every train chunk — only enable for probes.
    pub report_train: bool,
}

/// Outcome of a training run (best-eval checkpoint is the model that generalizes).
pub struct DrafterTrainReport {
    pub best_eval: f32,
    pub best_epoch: usize,
    pub bar: f32,
    pub final_eval: f32,
    /// Host snapshot of the drafter params (in `params()` order) at the best-eval
    /// epoch — the generalizing model to checkpoint. Empty if no eval ran.
    pub best_weights: Vec<Vec<f32>>,
}

// ── ranking math (shared with any drafter-vs-target evaluation) ──

pub fn rank(a: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..a.len()).collect();
    idx.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap());
    let mut r = vec![0.0f32; a.len()];
    for (pos, &i) in idx.iter().enumerate() {
        r[i] = pos as f32;
    }
    r
}

pub fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().sum::<f32>() as f64 / n,
        b.iter().sum::<f32>() as f64 / n,
    );
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
        c += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 {
        0.0
    } else {
        (c / (va.sqrt() * vb.sqrt())) as f32
    }
}

/// Spearman rank correlation.
pub fn spearman(a: &[f32], b: &[f32]) -> f32 {
    pearson(&rank(a), &rank(b))
}

/// Temperature softmax (ListNet top-1 target distribution).
pub fn softmax_t(x: &[f32], tau: f32) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| ((v - m) / tau).exp()).collect();
    let z: f32 = e.iter().sum();
    e.into_iter().map(|v| v / z).collect()
}

/// Mean Spearman(drafter, target-mid) over chunk indices `[start, end)`.
pub fn eval_ssm_drafter_range(
    gpu: &mut Gpu,
    drafter: &SsmDrafter,
    chunks: &[Vec<u32>],
    label_mid: &[Vec<f32>],
    start: usize,
    end: usize,
    cfg: &TrainCfg,
) -> f32 {
    let nb = cfg.seq / cfg.block;
    let last = cfg.seq - 1;
    let kvd = drafter.cfg.kv_dim();
    let pos: Vec<f32> = (0..cfg.seq).map(|t| t as f32).collect();
    let sc = gpu.zeros(&[nb], DType::F32).unwrap();
    let mut s = 0.0;
    for i in start..end {
        let a = ssm_drafter_forward_train(gpu, drafter, &chunks[i], &pos).unwrap();
        pflash_score_forward(gpu, &a.score_k, &sc, cfg.seq, kvd, cfg.block, nb, last).unwrap();
        let pred = gpu.download_f32(&sc).unwrap();
        s += spearman(&pred, &label_mid[i]);
        free_ssm_drafter_acts(gpu, a).unwrap();
    }
    let _ = gpu.free_tensor(sc);
    s / (end - start).max(1) as f32
}

/// Mean Spearman(drafter, target-mid) over the eval split (last `n_eval` chunks).
pub fn eval_ssm_drafter(
    gpu: &mut Gpu,
    drafter: &SsmDrafter,
    chunks: &[Vec<u32>],
    label_mid: &[Vec<f32>],
    cfg: &TrainCfg,
) -> f32 {
    let n_chunks = chunks.len();
    eval_ssm_drafter_range(
        gpu,
        drafter,
        chunks,
        label_mid,
        n_chunks - cfg.n_eval,
        n_chunks,
        cfg,
    )
}

/// Cross-epoch state for a resumable drafter training run. Held resident by the
/// daemon between micro-step quanta so `train_drafter` time-slices with
/// interactive serving. `drafter_loop_init` → N× `drafter_loop_run_epochs` →
/// `drafter_loop_finish` reproduces `train_ssm_drafter_loop` byte-for-byte.
pub struct DrafterLoopState {
    pub opt: AdamW,
    pub scores_dev: GpuTensor,
    pub best_eval: f32,
    pub best_epoch: usize,
    pub final_eval: f32,
    pub best_weights: Vec<Vec<f32>>,
    pub bar: f32,
    /// Next epoch to run (0..cfg.epochs).
    pub ep: usize,
    // Derived constants (computed once in init, constant for the run).
    n_train: usize,
    nb: usize,
    last: usize,
    kvd: usize,
    pos: Vec<f32>,
}

/// Set up the loop: compute `bar`, build the optimizer + scores scratch, and
/// init best-eval tracking. Verbatim from the pre-`for ep` head of the whole-run
/// loop; `st.ep` starts at 0.
pub fn drafter_loop_init(
    gpu: &mut Gpu,
    drafter: &SsmDrafter,
    chunks: &[Vec<u32>],
    label_mid: &[Vec<f32>],
    base_shallow: &[Vec<f32>],
    cfg: &TrainCfg,
) -> HipResult<DrafterLoopState> {
    let nb = cfg.seq / cfg.block;
    let last = cfg.seq - 1;
    let n_chunks = chunks.len();
    let n_train = n_chunks - cfg.n_eval;
    let kvd = drafter.cfg.kv_dim();
    let pos: Vec<f32> = (0..cfg.seq).map(|t| t as f32).collect();

    let bar: f32 = (n_train..n_chunks)
        .map(|i| spearman(&base_shallow[i], &label_mid[i]))
        .sum::<f32>()
        / cfg.n_eval as f32;

    let sizes = drafter.param_sizes();
    let opt = AdamW::new(gpu, &sizes, cfg.lr, 0.9, 0.999, 1e-8, cfg.wd)?;
    let scores_dev = gpu.zeros(&[nb], DType::F32)?;

    Ok(DrafterLoopState {
        opt,
        scores_dev,
        best_eval: f32::NEG_INFINITY,
        best_epoch: 0,
        final_eval: 0.0,
        best_weights: Vec::new(),
        bar,
        ep: 0,
        n_train,
        nb,
        last,
        kvd,
        pos,
    })
}

/// Run epochs `st.ep .. ep_end` (advancing `st.ep`). The per-epoch body is
/// verbatim from the whole-run loop. `on_epoch(epoch, train_loss, eval, best,
/// best_epoch, train_corr)` fires on eval epochs.
pub fn drafter_loop_run_epochs(
    gpu: &mut Gpu,
    drafter: &SsmDrafter,
    chunks: &[Vec<u32>],
    label_mid: &[Vec<f32>],
    cfg: &TrainCfg,
    st: &mut DrafterLoopState,
    ep_end: usize,
    mut on_epoch: impl FnMut(usize, f32, f32, f32, usize, Option<f32>),
) -> HipResult<()> {
    while st.ep < ep_end {
        let ep = st.ep;
        let mut ep_loss = 0.0f32;
        for i in 0..st.n_train {
            let acts = ssm_drafter_forward_train(gpu, drafter, &chunks[i], &st.pos)?;
            pflash_score_forward(
                gpu,
                &acts.score_k,
                &st.scores_dev,
                cfg.seq,
                st.kvd,
                cfg.block,
                st.nb,
                st.last,
            )?;
            let pred = gpu.download_f32(&st.scores_dev)?;
            // ListNet top-1: L = -Σ p_label log p_pred ; dL/dpred = (p_pred - p_label)/τ
            let pl = softmax_t(&label_mid[i], cfg.tau);
            let pp = softmax_t(&pred, cfg.tau);
            let mut ds = vec![0.0f32; st.nb];
            let mut l = 0.0f32;
            for b in 0..st.nb {
                l -= pl[b] * pp[b].max(1e-12).ln();
                ds[b] = (pp[b] - pl[b]) / cfg.tau;
            }
            ep_loss += l;
            let dscores = gpu.upload_f32(&ds, &[st.nb])?;
            let grads =
                ssm_drafter_backward(gpu, drafter, &acts, &dscores, cfg.block, st.nb, st.last)?;
            st.opt.step(gpu, &drafter.params(), &grads.flat())?;
            free_ssm_drafter_acts(gpu, acts)?;
            free_ssm_drafter_grads(gpu, grads)?;
            gpu.free_tensor(dscores)?;
        }
        // Live heartbeat on stderr (unbuffered) — a per-epoch signal between the
        // (every-`eval_every`) eval prints, so long runs are never silent/blind.
        {
            use std::io::Write;
            eprint!(
                "\r  [train] ep {:>4}/{}  loss {:.4}   ",
                ep + 1,
                cfg.epochs,
                ep_loss / st.n_train as f32
            );
            let _ = std::io::stderr().flush();
        }
        if ep % cfg.eval_every == 0 || ep == cfg.epochs - 1 {
            let corr = eval_ssm_drafter(gpu, drafter, chunks, label_mid, cfg);
            st.final_eval = corr;
            if corr > st.best_eval {
                st.best_eval = corr;
                st.best_epoch = ep;
                // Snapshot the generalizing weights (host) for checkpointing.
                st.best_weights = drafter
                    .params()
                    .iter()
                    .map(|t| gpu.download_f32(t))
                    .collect::<HipResult<_>>()?;
            }
            let train_corr = if cfg.report_train {
                Some(eval_ssm_drafter_range(
                    gpu, drafter, chunks, label_mid, 0, st.n_train, cfg,
                ))
            } else {
                None
            };
            on_epoch(
                ep,
                ep_loss / st.n_train as f32,
                corr,
                st.best_eval,
                st.best_epoch,
                train_corr,
            );
        }
        st.ep += 1;
    }
    Ok(())
}

/// Free the scores scratch and produce the report from the accumulated state.
pub fn drafter_loop_finish(gpu: &mut Gpu, st: DrafterLoopState) -> DrafterTrainReport {
    let _ = gpu.free_tensor(st.scores_dev);
    DrafterTrainReport {
        best_eval: st.best_eval,
        best_epoch: st.best_epoch,
        bar: st.bar,
        final_eval: st.final_eval,
        best_weights: st.best_weights,
    }
}

/// Train an SSM drafter to reproduce the target's mid-layer block ranking.
/// `on_epoch(epoch, train_loss, eval, best, best_epoch)` fires on eval epochs.
/// Returns the best-eval/bar/final report; the drafter's weights are left at
/// their FINAL (not necessarily best) state — caller checkpoints on best if it
/// wants the generalizing model.
///
/// Thin wrapper over the resumable `drafter_loop_{init,run_epochs,finish}` form:
/// init → run all epochs in one shot → finish.
pub fn train_ssm_drafter_loop(
    gpu: &mut Gpu,
    drafter: &SsmDrafter,
    chunks: &[Vec<u32>],
    label_mid: &[Vec<f32>],
    base_shallow: &[Vec<f32>],
    cfg: &TrainCfg,
    on_epoch: impl FnMut(usize, f32, f32, f32, usize, Option<f32>),
) -> HipResult<DrafterTrainReport> {
    let mut st = drafter_loop_init(gpu, drafter, chunks, label_mid, base_shallow, cfg)?;
    drafter_loop_run_epochs(
        gpu, drafter, chunks, label_mid, cfg, &mut st, cfg.epochs, on_epoch,
    )?;
    Ok(drafter_loop_finish(gpu, st))
}
