mod commands;
mod model;

use clap::{Parser, Subcommand};
use hipfire_config::load_config_bundle;
use std::ffi::OsString;

#[derive(Debug, Parser)]
#[command(
    name = "hipfire",
    version = hipfire_build_info::VERSION,
    about = "hipfire LLM inference CLI",
    long_about = "hipfire runs the local operator TUI, background inference server, model inventory, eval, benchmark, diagnostics, and artifact tools.\n\nRunning `hipfire` with no command opens the operator TUI; press `?` there for the in-app key reference.",
    after_help = "Examples:\n  hipfire                         Open the operator TUI (`?` for keys)\n  hipfire help                    Show the command summary\n  hipfire start                   Start the background server\n  hipfire status --json           Machine-readable server status\n  hipfire list                    Local models, sizes, and capabilities\n  hipfire download Qwen/Qwen3.5-9B   Fetch a model into the local store\n  hipfire induct Qwen/Qwen3.5-9B     Fetch, calibrate and quantize in one go\n  hipfire bench --model Qwen3.5-30B-A3B\n\n`--json` is available on start, stop, restart, status, list, and inspect.\nUse `hipfire <command> help` or `hipfire <command> --help` for detailed command help."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Start the background hipfire server
    Start(commands::daemon::StartArgs),
    /// Stop the background hipfire server
    Stop(commands::daemon::StopArgs),
    /// Restart the background hipfire server
    Restart(commands::daemon::RestartArgs),
    /// Show background server status
    Status(commands::daemon::StatusArgs),
    /// Start the hipfire HTTP server (OpenAI-compatible)
    #[command(hide = true)]
    Serve(commands::serve::ServeArgs),
    /// Run the inference daemon in the foreground (JSON-lines over stdin/stdout)
    Daemon(commands::daemon::DaemonRunArgs),
    /// Load a model and generate a response (one-shot)
    #[command(hide = true)]
    Chat(commands::chat::ChatArgs),
    /// Download a model repository (`org/name`) into the local store
    Download(commands::download::DownloadArgs),
    /// Bring an external model into a named `.hfq` — calibrate, quantize, fold
    /// sidecars. Accepts a HuggingFace `org/name` or a local safetensors dir.
    Induct(commands::induct::InductArgs),
    /// Import an external checkpoint (GGUF, safetensors) into a `.hfq`
    Import(commands::interop::ImportArgs),
    /// Export a `.hfq` back to an external format
    Export(commands::interop::ExportArgs),
    /// Pack a HuggingFace directory into a `.hfa` archive, or restore/verify one
    ///
    /// NOT `optimize`: that rewrites a `.hfq` into an arch-optimal weight
    /// layout. This is the lossless container round-trip.
    Repack(commands::interop::RepackArgs),
    /// List locally available models
    #[command(alias = "models")]
    List(commands::list::ListArgs),
    /// Detail the contents of a .hfq artefact (arch, shape, quant histogram, tensors)
    ///
    /// Diffusion containers are detected automatically and additionally report
    /// their pipeline summary (class, components, weight format, runtime
    /// support) — what `hipfire diffusion inspect` used to print separately.
    Inspect(commands::inspect::InspectArgs),
    /// Quantize a model artefact
    Quantize(commands::quantize::QuantizeArgs),
    /// Convert model artefacts (drafters, MTP heads)
    Convert {
        #[command(subcommand)]
        command: commands::quantize::ConvertCommand,
    },
    /// Run the quant admission/model evaluation harness
    Eval(commands::forward::EvalArgs),
    /// Live terminal monitor for GPU, memory, and daemon state
    Monitor,
    /// Kernel Atlas: inspect, count, and render Atlas rows
    Atlas(commands::tools::PassthroughArgs),
    /// Steering-vector harness
    Steer(commands::tools::PassthroughArgs),
    /// Harmful-neuron probe
    HneuronsProbe(commands::tools::PassthroughArgs),
    /// Inspect a .hfq artefact (verify, list, extract, meta-get/set, rearch)
    Hfq(commands::tools::PassthroughArgs),
    /// Quick daemon benchmark: load time, TTFT, pp512 prefill t/s, tg128 decode t/s
    Bench(commands::bench::BenchArgs),
    /// Diagnose the local Hipfire install, runtime, daemon, and monitoring prerequisites
    Doctor(commands::doctor::DoctorArgs),
    /// List the environment variables hipfire reads, with descriptions
    #[command(hide = true)]
    Env(commands::env::EnvArgs),
    /// Measure host, GPU-copy, and model storage bandwidth
    HostProfile(commands::forward::HostProfileArgs),
    /// Collect Tier-1 calibration artifacts (Hessian/imatrix/router-histogram) in one model load
    CollectArtifacts(commands::forward::CollectArtifactsArgs),
    /// Reshuffle a canonical .hfq into an arch-optimal layout (<model>.<arch>.hfq)
    ///
    /// The `repack` alias is gone: `repack` is a DIFFERENT operation — the
    /// HF-dir <-> `.hfa` archive round-trip — and one name for two things is
    /// worse than a longer name for one.
    Optimize(commands::forward::OptimizeArgs),
    /// Compose/decompose .hfq packaging: bundle a base + role/feature sidecars
    /// into one container, or split a bundle back into its component files.
    Model(commands::model::ModelArgs),
    /// GPU resource lock for multi-agent coordination (acquire/release/status)
    #[command(hide = true)]
    #[command(alias = "gpu-lock")]
    Lock(commands::lock::LockArgs),
    /// Run observational coherence detectors over a captured token stream
    ///
    /// Reads the demo/daemon stdout on stdin (the `DFlash tokens: [..]` /
    /// `AR tokens: [..]` line) and emits a JSON verdict with `ok` / `soft_warn`.
    /// Front-end for the `hipfire-detect` DetectorBank; replaces the Path-A
    /// token-attractor Python heredocs in the coherence gates.
    #[command(hide = true)]
    Detect(commands::detect::DetectArgs),
    /// Import, generate, and quantize diffusion models stored as .hfq artifacts
    ///
    /// Inspection is not here: `hipfire inspect <artefact>` autodetects a
    /// diffusion container and prints its pipeline summary.
    ///
    /// Runtime note: runnable `.hfq` diffusion artifacts still perform CLIP
    /// tokenization as host-side setup. `txt2img`, `img2img`, and `smoke` can
    /// opt into `--rocm-device-id` to route currently GPU-backed generation
    /// boundaries through ROCm.
    /// Diffusers cache discovery also lists transformer-denoiser pipelines such
    /// as Flux, Krea, Qwen Image, and Qwen Image Edit so clients can see
    /// convertible models, but native serving still requires a runnable `.hfq`
    /// artifact and a matching diffusion runtime.
    ///
    /// `hipfire serve --model <diffusion.hfq>` pre-warms the resolved diffusion
    /// pipeline cache directly instead of routing the artifact through the chat
    /// daemon loader. The server exposes the same hybrid path through the Stable
    /// Diffusion API extension fields `rocm_device_id` or
    /// `hipfire_rocm_device_id` on `/sdapi/v1/txt2img` and `/sdapi/v1/img2img`
    /// requests, through the same keys in `override_settings`, or through the
    /// persisted `/sdapi/v1/options` value `hipfire_rocm_device_id`. Persisted
    /// `/sdapi/v1/options` values for `send_images`, `save_images`,
    /// `outdir_samples`, `outdir_txt2img_samples`, `outdir_img2img_samples`,
    /// `outdir_grids`, `outdir_txt2img_grids`, and `outdir_img2img_grids`
    /// act as generation defaults unless the request or `override_settings`
    /// supplies a more specific value.
    ///
    /// `/sdapi/v1/progress` tracks active SDAPI sampling steps and updates
    /// `current_image` with live PNG previews decoded from intermediate
    /// latents, then leaves the final generated PNG there after a successful
    /// HFQ diffusion request completes; WebUI's `skip_current_image=true`
    /// progress query suppresses only that response's preview payload. The
    /// `/sdapi/v1/skip` endpoint records WebUI-compatible skip state without
    /// interrupting the whole request; `/sdapi/v1/interrupt` is the cancellation
    /// path.
    /// `/sdapi/v1/memory` returns WebUI-shaped host RAM stats and marks CUDA
    /// memory stats unavailable because Hipfire uses HIP/ROCm. WebUI's
    /// create/train embedding and hypernetwork endpoints are registered for
    /// client compatibility and return an `info` response explaining that
    /// native training is not implemented by the SDAPI layer.
    /// WebUI's optional server command endpoints (`server-kill`,
    /// `server-restart`, and `server-stop`) are registered as disabled
    /// compatibility no-ops so SDAPI clients do not see 404s, but external
    /// clients cannot stop or restart the Hipfire process through them.
    ///
    /// SDAPI sampler fields follow WebUI's split controls: full scheduler names
    /// such as `DDIM`, `DPM++ 2M`, and `DPM++ 3M` are accepted directly, while schedule
    /// modifiers such as `Automatic` and `Karras` combine with `sampler_name`
    /// or `sampler_index` (for example `Euler` + `Karras` becomes
    /// `Euler Karras`).
    ///
    /// SDAPI img2img and inpaint support WebUI resize modes 0 (stretch), 1
    /// (crop and resize), 2 (resize and fill), and 3 (latent upscale). Modes
    /// 0-2 resize init and mask images before VAE encoding; mode 3 keeps the
    /// init image at its source dimensions, VAE-encodes it, then resizes the
    /// latent tensor to the requested output shape;
    /// `/sdapi/v1/latent-upscale-modes` advertises Hipfire's nearest-neighbor
    /// latent resize aliases. `seed_resize_from_w` and `seed_resize_from_h`
    /// generate the initial noise at the requested source dimensions and resize
    /// it to the target latent shape before sampling. Hipfire also accepts
    /// common WebUI generation fields such as `styles`, `restore_faces`,
    /// `tiling`, `eta`, `s_churn`,
    /// `s_tmin`, `s_tmax`, `s_noise`, `override_settings_restore_afterwards`,
    /// `disable_extra_networks`, and `comments`; fields that do not affect the
    /// native runtime are returned in response `parameters` and listed in
    /// `info.ignored_fields` when active.
    /// `do_not_save_samples` suppresses disk writes even when `save_images` is
    /// true. `return_grid` appends a generated batch grid to the response image
    /// list for multi-image outputs, and `do_not_save_grid` suppresses grid disk
    /// writes independently of sample writes. Masked img2img also honors
    /// WebUI's `inpainting_mask_invert`, `mask_blur`,
    /// `mask_blur_x`, `mask_blur_y`, `mask_round`, and `inpainting_fill`
    /// options; default fill (0) is applied in image space before VAE encode,
    /// original (1) leaves init pixels unchanged, and latent noise (2) /
    /// latent nothing (3) additionally alter masked latents. WebUI's
    /// `inpaint_full_res` and
    /// `inpaint_full_res_padding` crop masked regions for processing and
    /// composite the generated crop back onto the init image. SDAPI requests
    /// can also import common WebUI `infotext` fields when those fields are not
    /// explicitly set in JSON. Non-empty `script_name` and `script_args`
    /// payloads are rejected because Hipfire exposes no SDAPI selectable
    /// scripts. `alwayson_scripts` accepts empty or disabled default extension
    /// payloads, but active script payloads are rejected. Txt2img high-res
    /// generation is implemented as a batched first-pass txt2img
    /// generation followed by a second-pass img2img generation at the high-res
    /// target dimensions. SDAPI high-res requests accept `enable_hr`,
    /// `firstphase_width`,
    /// `firstphase_height`, `hr_scale`, `hr_upscaler`, `hr_resize_x`,
    /// `hr_resize_y`, `hr_second_pass_steps`, `hr_checkpoint_name`,
    /// `hr_prompt`, `hr_negative_prompt`, `hr_sampler_name`, and
    /// `hr_scheduler`; `hr_checkpoint_name` may point to another resolvable
    /// diffusion HFQ artifact for the second pass.
    ///
    /// The runtime accepts Q4F16_G64, f16, bf16, f32, Q8F16, Q4_K,
    /// HFQ4G128, HFQ4G256, and HFQ6G256 tensor payloads. Other packed payloads
    /// require a matching diffusion dequantizer/runtime implementation.
    Diffusion(commands::diffusion::DiffusionArgs),
    /// Query the running hipfire admin API for scripts and agents
    #[command(alias = "op")]
    Admin(commands::admin::AdminArgs),
    /// Regenerate the committed CLI docs (docs/cli.md + man pages) from this
    /// clap definition. Hidden: a maintenance command, not part of the
    /// user-facing surface; run via `cargo run -p hipfire-cli -- gen-docs`.
    #[command(hide = true)]
    GenDocs(commands::gen_docs::GenDocsArgs),
    /// Render the shared config schema. Hidden: maintenance command for docs
    /// and operator UI schema artifacts.
    #[command(hide = true)]
    GenConfigSchema(commands::gen_config_schema::GenConfigSchemaArgs),
    /// Regenerate the env-var docs (docs/env-vars.md) by scanning the source
    /// tree. Hidden: a maintenance command; run via
    /// `cargo run -p hipfire-cli -- gen-env-docs`.
    #[command(hide = true)]
    GenEnvDocs(commands::gen_env_docs::GenEnvDocsArgs),
    /// Regenerate the model-support matrix artifacts (the generated tables in
    /// crates/hipfire-model + the chart in MODEL-SUPPORT.md) from
    /// docs/model-support.toml. Hidden: a maintenance command; run via
    /// `cargo run -p hipfire-cli -- gen-model-support`.
    #[command(hide = true)]
    GenModelSupport(commands::gen_model_support::GenModelSupportArgs),
}

fn main() -> anyhow::Result<()> {
    hipfire_runtime::logging::init_stderr_logging("cli", "hipfire=info,hipfire_server=info,warn");

    let cli = Cli::parse_from(normalize_help_args());
    let loaded_config = load_config_bundle();
    let config = loaded_config.config.clone();

    // `main` is deliberately NOT `#[tokio::main]`. That would start runtime
    // worker threads before dispatch, for every command — including the ones
    // that now live in this binary and own the process while they run:
    //
    //   * `daemon` blocks forever on a `hipfire_rdna::Gpu`, which is !Send;
    //   * `quantize` calls `env::set_var` to bridge `--beam` into the QTIP
    //     path, which is a data race the moment another thread exists, and
    //     was only ever safe because it used to be its own single-threaded
    //     process;
    //   * `eval` (step 4) builds its OWN multi-thread runtime and calls
    //     `block_on`, which panics if a runtime is already running the caller.
    //
    // So the runtime is built here, lazily, and only by the arms that await
    // something. Everything else runs on a bare main thread exactly as it did
    // when it was a separate executable.
    let rt = || -> anyhow::Result<tokio::runtime::Runtime> {
        Ok(tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?)
    };

    match cli.command {
        // Bare `hipfire` as a systemd service means serve, not the TUI.
        None if hipfire_runtime::logging::stderr_is_journal() => {
            rt()?.block_on(commands::serve::run(Default::default(), loaded_config))
        }
        None => hipfire_tui::run(),
        Some(Command::Start(args)) => rt()?.block_on(commands::daemon::start(args, loaded_config)),
        Some(Command::Stop(args)) => rt()?.block_on(commands::daemon::stop(args, loaded_config)),
        Some(Command::Restart(args)) => {
            rt()?.block_on(commands::daemon::restart(args, loaded_config))
        }
        Some(Command::Status(args)) => {
            rt()?.block_on(commands::daemon::status(args, loaded_config))
        }
        Some(Command::Serve(args)) => rt()?.block_on(commands::serve::run(args, loaded_config)),
        Some(Command::Daemon(args)) => commands::daemon::run_worker(args),
        Some(Command::Chat(args)) => rt()?.block_on(commands::chat::run(args, loaded_config)),
        Some(Command::Download(args)) => commands::download::run_download(args),
        Some(Command::Induct(args)) => commands::induct::run_induct(args, loaded_config),
        Some(Command::Import(args)) => commands::interop::run_import(args),
        Some(Command::Export(args)) => commands::interop::run_export(args),
        Some(Command::Repack(args)) => commands::interop::run_repack(args),
        Some(Command::List(args)) => commands::list::run(args, loaded_config),
        Some(Command::Inspect(args)) => commands::inspect::run(args, loaded_config),
        Some(Command::Quantize(args)) => commands::quantize::run(args),
        Some(Command::Convert { command }) => commands::quantize::run_convert(command),
        Some(Command::Eval(args)) => commands::forward::run_eval(args, loaded_config),
        Some(Command::Monitor) => commands::tools::run_monitor(),
        Some(Command::Atlas(args)) => commands::tools::run_atlas(args),
        Some(Command::Steer(args)) => commands::tools::run_steer(args),
        Some(Command::HneuronsProbe(args)) => commands::tools::run_hneurons_probe(args),
        Some(Command::Hfq(args)) => commands::tools::run_hfq(args),
        Some(Command::Bench(args)) => rt()?.block_on(commands::bench::run(args, loaded_config)),
        Some(Command::Doctor(args)) => rt()?.block_on(commands::doctor::run(args, loaded_config)),
        Some(Command::Env(args)) => commands::env::run(args),
        Some(Command::HostProfile(args)) => {
            commands::forward::run_host_profile(args, loaded_config)
        }
        Some(Command::CollectArtifacts(args)) => {
            commands::forward::run_collect_artifacts(args, loaded_config)
        }
        Some(Command::Optimize(args)) => commands::forward::run_optimize(args, loaded_config),
        Some(Command::Model(args)) => commands::model::run(args, loaded_config),
        Some(Command::Lock(args)) => commands::lock::run(args),
        Some(Command::Detect(args)) => commands::detect::run(args),
        Some(Command::Diffusion(args)) => commands::diffusion::run(args, loaded_config),
        Some(Command::Admin(args)) => rt()?.block_on(commands::admin::run(args, config)),
        Some(Command::GenDocs(args)) => commands::gen_docs::run(args),
        Some(Command::GenConfigSchema(args)) => commands::gen_config_schema::run(args),
        Some(Command::GenEnvDocs(args)) => commands::gen_env_docs::run(args),
        Some(Command::GenModelSupport(args)) => commands::gen_model_support::run(args),
    }
}

fn normalize_help_args() -> Vec<OsString> {
    let mut args = std::env::args_os().collect::<Vec<_>>();
    if args.len() >= 2 && args[1] == "help" {
        args.remove(1);
        args.push("--help".into());
    } else if args.len() >= 3 && args.last().is_some_and(|arg| arg == "help") {
        args.pop();
        args.push("--help".into());
    }
    args
}
