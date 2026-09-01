# Command-Line Help for `hipfire`

This document contains the help content for the `hipfire` command-line program.

**Command Overview:**

* [`hipfire`↴](#hipfire)
* [`hipfire start`↴](#hipfire-start)
* [`hipfire stop`↴](#hipfire-stop)
* [`hipfire restart`↴](#hipfire-restart)
* [`hipfire status`↴](#hipfire-status)
* [`hipfire daemon`↴](#hipfire-daemon)
* [`hipfire download`↴](#hipfire-download)
* [`hipfire induct`↴](#hipfire-induct)
* [`hipfire import`↴](#hipfire-import)
* [`hipfire import gguf`↴](#hipfire-import-gguf)
* [`hipfire import safetensors`↴](#hipfire-import-safetensors)
* [`hipfire export`↴](#hipfire-export)
* [`hipfire export safetensors`↴](#hipfire-export-safetensors)
* [`hipfire repack`↴](#hipfire-repack)
* [`hipfire lora`↴](#hipfire-lora)
* [`hipfire lora export`↴](#hipfire-lora-export)
* [`hipfire lora merge`↴](#hipfire-lora-merge)
* [`hipfire lora convert`↴](#hipfire-lora-convert)
* [`hipfire artifact`↴](#hipfire-artifact)
* [`hipfire artifact audit-calibration`↴](#hipfire-artifact-audit-calibration)
* [`hipfire artifact compare-calibration`↴](#hipfire-artifact-compare-calibration)
* [`hipfire artifact compare-calibration-stability`↴](#hipfire-artifact-compare-calibration-stability)
* [`hipfire artifact compare-residuals`↴](#hipfire-artifact-compare-residuals)
* [`hipfire artifact moe-router-profile`↴](#hipfire-artifact-moe-router-profile)
* [`hipfire calibrate`↴](#hipfire-calibrate)
* [`hipfire two-pass`↴](#hipfire-two-pass)
* [`hipfire npu`↴](#hipfire-npu)
* [`hipfire npu pair-hfp`↴](#hipfire-npu-pair-hfp)
* [`hipfire jobs`↴](#hipfire-jobs)
* [`hipfire jobs list`↴](#hipfire-jobs-list)
* [`hipfire jobs status`↴](#hipfire-jobs-status)
* [`hipfire jobs watch`↴](#hipfire-jobs-watch)
* [`hipfire jobs cancel`↴](#hipfire-jobs-cancel)
* [`hipfire list`↴](#hipfire-list)
* [`hipfire inspect`↴](#hipfire-inspect)
* [`hipfire quantize`↴](#hipfire-quantize)
* [`hipfire convert`↴](#hipfire-convert)
* [`hipfire convert dflash`↴](#hipfire-convert-dflash)
* [`hipfire convert dspark`↴](#hipfire-convert-dspark)
* [`hipfire convert draft-mq4`↴](#hipfire-convert-draft-mq4)
* [`hipfire convert mtp-extract`↴](#hipfire-convert-mtp-extract)
* [`hipfire convert mtp-merge`↴](#hipfire-convert-mtp-merge)
* [`hipfire eval`↴](#hipfire-eval)
* [`hipfire monitor`↴](#hipfire-monitor)
* [`hipfire atlas`↴](#hipfire-atlas)
* [`hipfire steer`↴](#hipfire-steer)
* [`hipfire hneurons-probe`↴](#hipfire-hneurons-probe)
* [`hipfire hfq`↴](#hipfire-hfq)
* [`hipfire bench`↴](#hipfire-bench)
* [`hipfire doctor`↴](#hipfire-doctor)
* [`hipfire host-profile`↴](#hipfire-host-profile)
* [`hipfire collect-artifacts`↴](#hipfire-collect-artifacts)
* [`hipfire optimize`↴](#hipfire-optimize)
* [`hipfire model`↴](#hipfire-model)
* [`hipfire model compose`↴](#hipfire-model-compose)
* [`hipfire model decompose`↴](#hipfire-model-decompose)
* [`hipfire model induct`↴](#hipfire-model-induct)
* [`hipfire model inspect`↴](#hipfire-model-inspect)
* [`hipfire diffusion`↴](#hipfire-diffusion)
* [`hipfire diffusion import`↴](#hipfire-diffusion-import)
* [`hipfire diffusion preflight`↴](#hipfire-diffusion-preflight)
* [`hipfire diffusion txt2img`↴](#hipfire-diffusion-txt2img)
* [`hipfire diffusion img2img`↴](#hipfire-diffusion-img2img)
* [`hipfire diffusion smoke`↴](#hipfire-diffusion-smoke)
* [`hipfire diffusion quantize`↴](#hipfire-diffusion-quantize)
* [`hipfire diffusion calibrate`↴](#hipfire-diffusion-calibrate)
* [`hipfire diffusion quant-diff`↴](#hipfire-diffusion-quant-diff)
* [`hipfire diffusion calib-eval`↴](#hipfire-diffusion-calib-eval)
* [`hipfire admin`↴](#hipfire-admin)
* [`hipfire admin status`↴](#hipfire-admin-status)
* [`hipfire admin chat`↴](#hipfire-admin-chat)
* [`hipfire admin health`↴](#hipfire-admin-health)
* [`hipfire admin models`↴](#hipfire-admin-models)
* [`hipfire admin config`↴](#hipfire-admin-config)
* [`hipfire admin training`↴](#hipfire-admin-training)
* [`hipfire admin diagnostics`↴](#hipfire-admin-diagnostics)
* [`hipfire admin logs`↴](#hipfire-admin-logs)
* [`hipfire admin get`↴](#hipfire-admin-get)
* [`hipfire admin set-password`↴](#hipfire-admin-set-password)

## `hipfire`

hipfire runs the local operator TUI, background inference server, model inventory, eval, benchmark, diagnostics, and artifact tools.

Running `hipfire` with no command opens the operator TUI; press `?` there for the in-app key reference.

**Usage:** `hipfire [COMMAND]`

Examples:
  hipfire                         Open the operator TUI (`?` for keys)
  hipfire help                    Show the command summary
  hipfire start                   Start the background server
  hipfire status --json           Machine-readable server status
  hipfire list                    Local models, sizes, and capabilities
  hipfire download Qwen/Qwen3.5-9B   Fetch a model into the local store
  hipfire induct Qwen/Qwen3.5-9B     Fetch, calibrate and quantize in one go
  hipfire bench --model Qwen3.5-30B-A3B

`--json` is available on start, stop, restart, status, list, and inspect.
Use `hipfire <command> help` or `hipfire <command> --help` for detailed command help.

###### **Subcommands:**

* `start` — Start the background hipfire server
* `stop` — Stop the background hipfire server
* `restart` — Restart the background hipfire server
* `status` — Show background server status
* `daemon` — Run the inference daemon in the foreground (JSON-lines over stdin/stdout)
* `download` — Download a model repository (`org/name`) into the local store
* `induct` — Bring an external model into a named `.hfq` — calibrate, quantize, fold sidecars. Accepts a HuggingFace `org/name` or a local safetensors dir
* `import` — Import an external checkpoint (GGUF, safetensors) into a `.hfq`
* `export` — Export a `.hfq` back to an external format
* `repack` — Pack a HuggingFace directory into a `.hfa` archive, or restore/verify one
* `lora` — Derive, merge or convert a steering adapter
* `artifact` — Audit and compare calibration / residual artifacts
* `calibrate` — Capture activation statistics into a `.calib.hfq`
* `two-pass` — Calibrate then quantize in one run
* `npu` — NPU artifact tooling (linux only)
* `jobs` — Submit, watch and cancel background jobs (downloads, training)
* `list` — List locally available models
* `inspect` — Detail the contents of a .hfq artefact (arch, shape, quant histogram, tensors)
* `quantize` — Quantize a model artefact
* `convert` — Convert model artefacts (drafters, MTP heads)
* `eval` — Run the quant admission/model evaluation harness
* `monitor` — Live terminal monitor for GPU, memory, and daemon state
* `atlas` — Kernel Atlas: inspect, count, and render Atlas rows
* `steer` — Steering-vector harness
* `hneurons-probe` — Harmful-neuron probe
* `hfq` — Inspect a .hfq artefact (verify, list, extract, meta-get/set, rearch)
* `bench` — Quick daemon benchmark: load time, TTFT, pp512 prefill t/s, tg128 decode t/s
* `doctor` — Diagnose the local Hipfire install, runtime, daemon, and monitoring prerequisites
* `host-profile` — Measure host, GPU-copy, and model storage bandwidth
* `collect-artifacts` — Collect Tier-1 calibration artifacts (Hessian/imatrix/router-histogram) in one model load
* `optimize` — Reshuffle a canonical .hfq into an arch-optimal layout (<model>.<arch>.hfq)
* `model` — Compose/decompose .hfq packaging: bundle a base + role/feature sidecars into one container, or split a bundle back into its component files
* `diffusion` — Import, generate, and quantize diffusion models stored as .hfq artifacts
* `admin` — Query the running hipfire admin API for scripts and agents



## `hipfire start`

Start the background hipfire server

**Usage:** `hipfire start [OPTIONS]`

Examples:
  hipfire start
  hipfire start --model Qwen3.5-30B-A3B --port 11435
  hipfire start --host 0.0.0.0


###### **Options:**

* `--host <HOST>` — Override bind host for the background server
* `-p`, `--port <PORT>` — Override bind port for the background server
* `-m`, `--model <MODEL>` — Pre-load a model on startup by name, shorthand, alias, or path
* `--debug-chat` — Log full raw chat requests and raw model replies
* `--wait-secs <WAIT_SECS>` — Seconds to wait for /health before returning. Default 0 returns immediately

  Default value: `0`
* `--json` — Emit a machine-readable JSON object instead of the human summary



## `hipfire stop`

Stop the background hipfire server

**Usage:** `hipfire stop [OPTIONS]`

Examples:
  hipfire stop
  hipfire stop --force


###### **Options:**

* `-f`, `--force` — Skip the graceful wait and send SIGKILL immediately
* `--json` — Emit a machine-readable JSON object instead of the human summary



## `hipfire restart`

Restart the background hipfire server

**Usage:** `hipfire restart [OPTIONS]`

Examples:
  hipfire restart
  hipfire restart --model Qwen3.5-30B-A3B


###### **Options:**

* `--host <HOST>` — Override bind host for the restarted background server
* `-p`, `--port <PORT>` — Override bind port for the restarted background server
* `-m`, `--model <MODEL>` — Pre-load a model on startup by name, shorthand, alias, or path
* `--debug-chat` — Log full raw chat requests and raw model replies
* `--wait-secs <WAIT_SECS>` — Seconds to wait for /health before returning. Default 0 returns immediately

  Default value: `0`
* `--json` — Emit a machine-readable JSON object instead of the human summary



## `hipfire status`

Show background server status

**Usage:** `hipfire status [OPTIONS]`

Examples:
  hipfire status


###### **Options:**

* `--json` — Emit a machine-readable JSON object instead of the human summary



## `hipfire daemon`

Run the inference daemon in the foreground (JSON-lines over stdin/stdout)

**Usage:** `hipfire daemon [ARGS]...`

Examples:
  hipfire daemon
  hipfire daemon --listen
  hipfire daemon --listen /run/hipfire.sock
  hipfire daemon --precompile


###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the daemon



## `hipfire download`

Download a model repository (`org/name`) into the local store

**Usage:** `hipfire download [OPTIONS] <REPO>`

Examples:
  hipfire download Qwen/Qwen3.5-9B
  hipfire download Qwen/Qwen3.5-9B --revision <sha>
  hipfire download Zyphra/ZAYA1-8B --include '*.safetensors'
  hipfire download Qwen/Qwen3.5-9B --raw          # HuggingFace cache tree

Streams into ~/.hipfire/models/models--Org--Name.hfa, encoding as it
downloads so the raw checkpoint is never staged. An interrupted run
leaves <archive>.hfa.part and resumes on the next download.

###### **Arguments:**

* `<REPO>` — Repository to fetch, as `org/name`.

   HuggingFace is the only source today. When a second one exists it joins as `--source <name>` rather than a new subcommand.

###### **Options:**

* `--revision <REVISION>` — Revision to pin: a commit sha, or `main`

  Default value: `main`
* `--include <INCLUDE>` — Only fetch paths matching this glob
* `--dest <DEST>` — Destination root. Defaults to `~/.hipfire/models` (or `$HF_HOME` with `--raw`)
* `--output <OUTPUT>` — Write the archive to this exact path instead of deriving it from the repo
* `--force` — Replace an existing archive. Without this an existing file is never overwritten — these are routinely the only copy of a model on an array with no redundancy
* `--raw` — Fetch a HuggingFace cache tree instead of encoding to a `.hfa` archive
* `--jobs <JOBS>` — Parallel connections: whole files in raw mode, ranged windows within a file in archive mode

  Default value: `4`
* `--detach` — Queue the fetch as a background job instead of downloading here, and return its id. Monitor it with `hipfire jobs watch <id>`.

   The job is a file in `~/.hipfire/jobs/deferred/queued`, so this works whether or not the server is running — an unclaimed job simply waits.



## `hipfire induct`

Bring an external model into a named `.hfq` — calibrate, quantize, fold sidecars. Accepts a HuggingFace `org/name` or a local safetensors dir

**Usage:** `hipfire induct [OPTIONS] [SOURCE]`

###### **Arguments:**

* `<SOURCE>` — Model source: a HuggingFace repo id (`org/name`) or a local safetensors directory. Omit to be prompted

###### **Options:**

* `--format <FORMAT>` — Quant format token (e.g. `oq4++`, `mq4`, `qtip3`, `bf16`). Omit to be prompted from the known list
* `--detach` — Queue the induction as a background job instead of running it here, and return its id. Monitor it with `hipfire jobs watch <id>`.

   Both `source` and `--format` are required with this flag: a detached job has no terminal to prompt on.



## `hipfire import`

Import an external checkpoint (GGUF, safetensors) into a `.hfq`

**Usage:** `hipfire import <COMMAND>`

###### **Subcommands:**

* `gguf` — Import a GGUF checkpoint into a `.hfq`
* `safetensors` — Import a HuggingFace safetensors directory into a `.hfq`



## `hipfire import gguf`

Import a GGUF checkpoint into a `.hfq`

**Usage:** `hipfire import gguf [OPTIONS] --input <INPUT> --output <OUTPUT> --format <FORMAT>`

###### **Options:**

* `--input <INPUT>` — Source `.gguf`
* `--output <OUTPUT>` — Destination `.hfq`
* `--format <FORMAT>` — Target quant format token
* `--no-kmap` — Disable the k-map, quantizing uniformly
* `--kmap-dense` — Dense k-map
* `--kmap-mode <KMAP_MODE>` — k-map mode: `full`, `alternating`/`alt`, or `typed`

  Default value: `alternating`



## `hipfire import safetensors`

Import a HuggingFace safetensors directory into a `.hfq`

**Usage:** `hipfire import safetensors [OPTIONS] --input <INPUT> --output <OUTPUT>`

###### **Options:**

* `--input <INPUT>` — Source HuggingFace directory
* `--output <OUTPUT>` — Destination `.hfq`
* `--arch <ARCH>` — Architecture family override



## `hipfire export`

Export a `.hfq` back to an external format

**Usage:** `hipfire export <COMMAND>`

###### **Subcommands:**

* `safetensors` — Export a `.hfq` back to a HuggingFace safetensors directory



## `hipfire export safetensors`

Export a `.hfq` back to a HuggingFace safetensors directory

**Usage:** `hipfire export safetensors [OPTIONS] --input <INPUT> --output <OUTPUT>`

###### **Options:**

* `--input <INPUT>` — Source `.hfq`
* `--output <OUTPUT>` — Destination directory
* `--arch <ARCH>` — Architecture family override
* `--shard-size <SHARD_SIZE>` — Shard size, e.g. `5G`



## `hipfire repack`

Pack a HuggingFace directory into a `.hfa` archive, or restore/verify one

NOT `optimize`: that rewrites a `.hfq` into an arch-optimal weight layout. This is the lossless container round-trip.

**Usage:** `hipfire repack [OPTIONS] --input <INPUT>`

Examples:
  hipfire repack --input <hf_dir> --output <archive.hfa>   # pack, lossless
  hipfire repack --input <archive.hfa> --output <hf_dir>   # restore, byte-identical
  hipfire repack --input <archive.hfa> --check             # verify stored checksums

Not to be confused with `hipfire optimize`, which rewrites a .hfq into
an arch-optimal weight layout.

###### **Options:**

* `--input <INPUT>` — Source: a HuggingFace directory to pack, or a `.hfa` to restore/check
* `--output <OUTPUT>` — Destination. Omit with `--check`
* `--verify <VERIFY>` — Verify the restored tree against this directory
* `--check` — Verify stored checksums without writing anything
* `--upgrade` — Upgrade an older archive in place



## `hipfire lora`

Derive, merge or convert a steering adapter

**Usage:** `hipfire lora <COMMAND>`

###### **Subcommands:**

* `export` — Derive a steering adapter from contrastive prompt sets
* `merge` — Merge an adapter into a base `.hfq`
* `convert` — Convert an adapter between `.hfq` and `.json` forms



## `hipfire lora export`

Derive a steering adapter from contrastive prompt sets

**Usage:** `hipfire lora export [OPTIONS] --hfq <HFQ> --data-dir <DATA_DIR> --out <OUT>`

###### **Options:**

* `--hfq <HFQ>` — Base model to derive the adapter against
* `--data-dir <DATA_DIR>` — Directory holding `good_prompts.txt` and `bad_prompts.txt`
* `--out <OUT>` — Destination adapter (`.lora.hfq` or `.lora.json`)
* `--limit <LIMIT>` — Prompts to read from each set

  Default value: `16`
* `--strength <STRENGTH>` — Steering strength

  Default value: `0.2`
* `--max-seq <MAX_SEQ>` — Max sequence length during capture

  Default value: `2048`
* `--no-orthogonalize` — Skip orthogonalisation of the derived directions



## `hipfire lora merge`

Merge an adapter into a base `.hfq`

**Usage:** `hipfire lora merge --hfq <HFQ> --adapter <ADAPTER> --out <OUT>`

###### **Options:**

* `--hfq <HFQ>` — Base `.hfq`
* `--adapter <ADAPTER>` — Adapter to merge
* `--out <OUT>` — Destination merged `.hfq`



## `hipfire lora convert`

Convert an adapter between `.hfq` and `.json` forms

**Usage:** `hipfire lora convert --in <INPUT> --out <OUT>`

###### **Options:**

* `--in <INPUT>` — Source adapter. (Spelled `--in`, matching the existing tool.)
* `--out <OUT>` — Destination adapter



## `hipfire artifact`

Audit and compare calibration / residual artifacts

**Usage:** `hipfire artifact <COMMAND>`

###### **Subcommands:**

* `audit-calibration` — Check a `.calib.hfq` for structural and coverage problems
* `compare-calibration` — Compare two calibration artifacts numerically
* `compare-calibration-stability` — Compare a lower-capacity calibration against a higher-capacity one
* `compare-residuals` — Compare two residual-probe artifacts
* `moe-router-profile` — Report routed-expert activation distribution from a calibration artifact



## `hipfire artifact audit-calibration`

Check a `.calib.hfq` for structural and coverage problems

**Usage:** `hipfire artifact audit-calibration --input <INPUT>`

###### **Options:**

* `--input <INPUT>` — Calibration artifact to audit



## `hipfire artifact compare-calibration`

Compare two calibration artifacts numerically

**Usage:** `hipfire artifact compare-calibration [OPTIONS] --reference <REFERENCE> --candidate <CANDIDATE>`

###### **Options:**

* `--reference <REFERENCE>` — Reference artifact
* `--candidate <CANDIDATE>` — Candidate artifact
* `--atol <ATOL>` — Absolute tolerance
* `--rtol <RTOL>` — Relative tolerance
* `--max-reports <MAX_REPORTS>` — Cap on reported mismatches
* `--allow-unproven-provenance` — Compare even when provenance cannot be proven equal



## `hipfire artifact compare-calibration-stability`

Compare a lower-capacity calibration against a higher-capacity one

**Usage:** `hipfire artifact compare-calibration-stability --reference <REFERENCE> --candidate <CANDIDATE>`

###### **Options:**

* `--reference <REFERENCE>` — Reference (higher-capacity) artifact
* `--candidate <CANDIDATE>` — Candidate (lower-capacity) artifact



## `hipfire artifact compare-residuals`

Compare two residual-probe artifacts

**Usage:** `hipfire artifact compare-residuals [OPTIONS] --reference <REFERENCE> --candidate <CANDIDATE>`

###### **Options:**

* `--reference <REFERENCE>` — Reference residuals artifact
* `--candidate <CANDIDATE>` — Candidate residuals artifact
* `--atol <ATOL>` — Absolute tolerance
* `--rtol <RTOL>` — Relative tolerance
* `--max-reports <MAX_REPORTS>` — Cap on reported mismatches



## `hipfire artifact moe-router-profile`

Report routed-expert activation distribution from a calibration artifact

**Usage:** `hipfire artifact moe-router-profile [OPTIONS] --input <INPUT>`

###### **Options:**

* `--input <INPUT>` — Calibration artifact to profile
* `--layer <LAYER>` — Restrict to one layer
* `--top <TOP>` — Report the top N experts
* `--min-activations <MIN_ACTIVATIONS>` — Ignore experts below this activation count
* `--tokenizer <TOKENIZER>` — Tokenizer for naming, when available
* `--json` — Emit JSON instead of a table



## `hipfire calibrate`

Capture activation statistics into a `.calib.hfq`

**Usage:** `hipfire calibrate [OPTIONS] --model <MODEL> --corpus <CORPUS> --output <OUTPUT>`

Defaults, validation and the `auto|N` forms are owned by the
        calibration parser, not by this command -- so they are applied but not
        listed here. `hipfire calibrate --help-flags` prints the full reference.

###### **Options:**

* `--model <MODEL>` — Model to calibrate: a safetensors dir or cache root
* `--corpus <CORPUS>` — Calibration corpus
* `--output <OUTPUT>` — Destination `.calib.hfq`
* `--sequences <SEQUENCES>`
* `--context <CONTEXT>`
* `--sampling-seed <SAMPLING_SEED>`
* `--sequence-batch <SEQUENCE_BATCH>` — `auto` or a row count
* `--time-tile <TIME_TILE>` — `auto` or a tile size
* `--max-rows <MAX_ROWS>`
* `--min-expert-activations <MIN_EXPERT_ACTIVATIONS>`
* `--expert-capture-target <EXPERT_CAPTURE_TARGET>`
* `--expert-capture-tile-rows <EXPERT_CAPTURE_TILE_ROWS>`
* `--required-expert-fraction <REQUIRED_EXPERT_FRACTION>`
* `--expert-coverage-policy <EXPERT_COVERAGE_POLICY>` — `strict` or `preserve-undercovered`
* `--kldref` — Capture a KLD reference
* `--no-kldref` — Skip the KLD reference
* `--kldref-topk <KLDREF_TOPK>`
* `--kldref-rows <KLDREF_ROWS>`
* `--layer-prefetch-bytes <LAYER_PREFETCH_BYTES>`
* `--boundary-ram` — Hold boundary rows in RAM instead of on disk
* `--boundary-dir <BOUNDARY_DIR>`
* `--resume` — Resume an interrupted run (the default)
* `--no-resume` — Start fresh, discarding any spool
* `--finalize-completed` — Publish an already-complete resumed spool without executing a layer
* `--pause-after-layers <PAUSE_AFTER_LAYERS>`
* `--residual-probe-output <RESIDUAL_PROBE_OUTPUT>`
* `--residual-probe-rows <RESIDUAL_PROBE_ROWS>`
* `--cask-output <CASK_OUTPUT>` — Also write a CASK (TriAttention centers) sidecar
* `--cask-only` — CASK only: the calibration artifact is scratch and removed after
* `--dry-run` — Plan without executing
* `--help-flags` — Print the calibration parser's own flag reference and exit



## `hipfire two-pass`

Calibrate then quantize in one run

**Usage:** `hipfire two-pass [OPTIONS] --model <MODEL> --calib <CALIB> --output <OUTPUT> [-- <QUANT_ARGS>...]`

Arguments after `--` are forwarded verbatim to the quantizer.

###### **Arguments:**

* `<QUANT_ARGS>` — Quantizer arguments, after `--`

###### **Options:**

* `--model <MODEL>` — Model to calibrate then quantize
* `--calib <CALIB>` — Calibration artifact to write or reuse (`.calib.hfq`)
* `--output <OUTPUT>` — Destination quantized artifact
* `--format <FORMAT>` — Quant format token
* `--corpus <CORPUS>` — Calibration corpus
* `--skip-calib` — Reuse an existing calibration instead of capturing one
* `--dry-run` — Plan without executing



## `hipfire npu`

NPU artifact tooling (linux only)

**Usage:** `hipfire npu <COMMAND>`

###### **Subcommands:**

* `pair-hfp` — Pair a whole-scaled `.hfp` into the paired layout (linux only)



## `hipfire npu pair-hfp`

Pair a whole-scaled `.hfp` into the paired layout (linux only)

**Usage:** `hipfire npu pair-hfp --in <INPUT> --out <OUTPUT>`

###### **Options:**

* `--in <INPUT>` — Source `.hfp`. (Spelled `--in`, matching the existing tool.)
* `--out <OUTPUT>` — Destination `.hfp`



## `hipfire jobs`

Submit, watch and cancel background jobs (downloads, training)

**Usage:** `hipfire jobs <COMMAND>`

Examples:
  hipfire download Qwen/Qwen3.5-9B --detach   # submit, return immediately
  hipfire jobs list
  hipfire jobs watch <id>
  hipfire jobs cancel <id>

A cancelled download can be resubmitted: the archive is written under a
.part marker with a .manifest sidecar, so it resumes rather than restarts.

###### **Subcommands:**

* `list` — List jobs in every state
* `status` — Show one job, with the tail of its log
* `watch` — Follow one job until it finishes
* `cancel` — Ask a running job to stop, or drop a queued one



## `hipfire jobs list`

List jobs in every state

**Usage:** `hipfire jobs list [OPTIONS]`

###### **Options:**

* `--json` — Emit JSON instead of a table



## `hipfire jobs status`

Show one job, with the tail of its log

**Usage:** `hipfire jobs status [OPTIONS] <ID>`

###### **Arguments:**

* `<ID>` — Job id, as printed by `list`

###### **Options:**

* `--json` — Emit JSON instead of text



## `hipfire jobs watch`

Follow one job until it finishes

**Usage:** `hipfire jobs watch [OPTIONS] <ID>`

###### **Arguments:**

* `<ID>` — Job id, as printed by `list`

###### **Options:**

* `--json` — Emit JSON instead of text



## `hipfire jobs cancel`

Ask a running job to stop, or drop a queued one

**Usage:** `hipfire jobs cancel [OPTIONS] <ID>`

###### **Arguments:**

* `<ID>` — Job id, as printed by `list`

###### **Options:**

* `--json` — Emit JSON instead of text



## `hipfire list`

List locally available models

**Usage:** `hipfire list [OPTIONS]`

Examples:
  hipfire list
  hipfire list --json
  hipfire list --local


###### **Options:**

* `--json` — Emit a machine-readable JSON array instead of the table
* `--local` — Skip the secondary (network) model store even when one is configured



## `hipfire inspect`

Detail the contents of a .hfq artefact (arch, shape, quant histogram, tensors)

Diffusion containers are detected automatically and additionally report their pipeline summary (class, components, weight format, runtime support) — what `hipfire diffusion inspect` used to print separately.

**Usage:** `hipfire inspect [OPTIONS] <TARGET>`

###### **Arguments:**

* `<TARGET>` — Container to inspect: a `.hfq` file path or a local model alias

###### **Options:**

* `--tensors` — List every tensor (name, quant type, shape, group size, size)
* `--json` — Emit a machine-readable JSON object (includes the full tensor array and the raw metadata verbatim); ignores `--tensors`



## `hipfire quantize`

Quantize a model artefact

**Usage:** `hipfire quantize [ARGS]...`

Examples:
  hipfire quantize --input model.hfa --output model--oq4.hfq --quant oq4
  hipfire quantize --help


###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the quantizer



## `hipfire convert`

Convert model artefacts (drafters, MTP heads)

**Usage:** `hipfire convert <COMMAND>`

###### **Subcommands:**

* `dflash` — Build a DFlash drafter sidecar from a model
* `dspark` — Build a DSpark drafter sidecar from a model
* `draft-mq4` — Convert a drafter to mq4
* `mtp-extract` — Extract an MTP head into its own artefact
* `mtp-merge` — Merge an MTP head into an mq4 artefact



## `hipfire convert dflash`

Build a DFlash drafter sidecar from a model

**Usage:** `hipfire convert dflash [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the tool



## `hipfire convert dspark`

Build a DSpark drafter sidecar from a model

**Usage:** `hipfire convert dspark [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the tool



## `hipfire convert draft-mq4`

Convert a drafter to mq4

**Usage:** `hipfire convert draft-mq4 [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the tool



## `hipfire convert mtp-extract`

Extract an MTP head into its own artefact

**Usage:** `hipfire convert mtp-extract [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the tool



## `hipfire convert mtp-merge`

Merge an MTP head into an mq4 artefact

**Usage:** `hipfire convert mtp-merge [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the tool



## `hipfire eval`

Run the quant admission/model evaluation harness

**Usage:** `hipfire eval [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded to hipfire-eval. Use positional <model>; common flags include --compare, --reference, --battery, --suite, --benchmark, --runs, --force, and --regenerate



## `hipfire monitor`

Live terminal monitor for GPU, memory, and daemon state

**Usage:** `hipfire monitor`



## `hipfire atlas`

Kernel Atlas: inspect, count, and render Atlas rows

**Usage:** `hipfire atlas [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the tool



## `hipfire steer`

Steering-vector harness

**Usage:** `hipfire steer [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the tool



## `hipfire hneurons-probe`

Harmful-neuron probe

**Usage:** `hipfire hneurons-probe [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the tool



## `hipfire hfq`

Inspect a .hfq artefact (verify, list, extract, meta-get/set, rearch)

**Usage:** `hipfire hfq [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded verbatim to the tool



## `hipfire bench`

Quick daemon benchmark: load time, TTFT, pp512 prefill t/s, tg128 decode t/s

**Usage:** `hipfire bench [OPTIONS] [MODEL]`

Examples:
  hipfire bench Qwen3.5-30B-A3B
  hipfire bench --pp-tokens 512 --tg-tokens 128 --repetitions 5
  hipfire bench Qwen3.5-30B-A3B --json


###### **Arguments:**

* `<MODEL>` — Model name, shorthand, alias, or path. Falls back to default_model

###### **Options:**

* `--pp-tokens <PP_TOKENS>` — Target prompt/prefill token count. The daemon reports the actual count

  Default value: `512`
* `--tg-tokens <TG_TOKENS>` — Generated token count for the decode-throughput sample

  Default value: `128`
* `-r`, `--repetitions <REPETITIONS>` — Number of measured repetitions, matching llama-bench's default

  Default value: `5`
* `--no-warmup` — Skip warmup runs before measuring
* `--json` — Print JSON instead of a compact text report



## `hipfire doctor`

Diagnose the local Hipfire install, runtime, daemon, and monitoring prerequisites

**Usage:** `hipfire doctor [OPTIONS]`

Examples:
  hipfire doctor
  hipfire doctor --json
  hipfire doctor --fix


###### **Options:**

* `--fix` — Apply safe user-space fixes and invoke hipfire-priv-helper for privileged fixes
* `--json` — Emit the full report as JSON



## `hipfire host-profile`

Measure host, GPU-copy, and model storage bandwidth

**Usage:** `hipfire host-profile [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded to hipfire-host-profile



## `hipfire collect-artifacts`

Collect Tier-1 calibration artifacts (Hessian/imatrix/router-histogram) in one model load

**Usage:** `hipfire collect-artifacts [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded to the collect_artifacts runner



## `hipfire optimize`

Reshuffle a canonical .hfq into an arch-optimal layout (<model>.<arch>.hfq)

The `repack` alias is gone: `repack` is a DIFFERENT operation — the HF-dir <-> `.hfa` archive round-trip — and one name for two things is worse than a longer name for one.

**Usage:** `hipfire optimize [ARGS]...`

###### **Arguments:**

* `<ARGS>` — Arguments forwarded to the optimize runner



## `hipfire model`

Compose/decompose .hfq packaging: bundle a base + role/feature sidecars into one container, or split a bundle back into its component files

**Usage:** `hipfire model <COMMAND>`

###### **Subcommands:**

* `compose` — Merge a base `.hfq` and its role/feature sidecars into one bundled container (records a provenance manifest so `decompose` is lossless)
* `decompose` — Split a bundled `.hfq` back into its base + sidecar files
* `induct` — Interactive wizard: bring an external model (HuggingFace repo or local safetensors dir) into a named `.hfq` — calibrate, quantize, fold sidecars
* `inspect` — Detail the contents of a `.hfq` container. Alias of top-level `hipfire inspect`; same arguments, same output



## `hipfire model compose`

Merge a base `.hfq` and its role/feature sidecars into one bundled container (records a provenance manifest so `decompose` is lossless)

**Usage:** `hipfire model compose [OPTIONS] <INPUTS> <INPUTS>...`

###### **Arguments:**

* `<INPUTS>` — Base container first, then one or more sidecars (file paths or model aliases)

###### **Options:**

* `-o`, `--output <OUTPUT>` — Output bundle path. Default: the base name with the sidecar feature dot-groups inserted before the quant token, each marked `+` because the role is now embedded rather than standalone (e.g. `Model--mq4.hfq` + `Model.mtp.hfq` -> `Model--+mtp.mq4.hfq`)
* `--check` — Validate component roles, formats, architectures, geometry, lengths, digests, and reserved namespaces without writing a bundle
* `--json` — Emit a machine-readable JSON report
* `--overwrite` — Replace an existing output bundle. Without this flag compose fails closed when the destination exists



## `hipfire model decompose`

Split a bundled `.hfq` back into its base + sidecar files

**Usage:** `hipfire model decompose [OPTIONS] <BUNDLE> <OUTPUT_DIR>`

###### **Arguments:**

* `<BUNDLE>` — Bundle container to split (file path or model alias)
* `<OUTPUT_DIR>` — Directory to write the reconstructed component files into

###### **Options:**

* `--infer` — Heuristically split a bundle that has no `hipfire_compose` manifest, using the filename's role dot-groups + tensor-name prefixes. Legacy bundles with a plain filename fall back to inferring roles from tensor names alone. Lossy: output files are not byte-identical to any originals. Bundles that DO carry a manifest still take the exact, lossless path
* `--json` — Emit a machine-readable JSON report
* `--overwrite` — Replace existing reconstructed component files. Without this flag decompose fails closed before replacing a destination



## `hipfire model induct`

Interactive wizard: bring an external model (HuggingFace repo or local safetensors dir) into a named `.hfq` — calibrate, quantize, fold sidecars

**Usage:** `hipfire model induct [OPTIONS] [SOURCE]`

###### **Arguments:**

* `<SOURCE>` — Model source: a HuggingFace repo id (`org/name`) or a local safetensors directory. Omit to be prompted

###### **Options:**

* `--format <FORMAT>` — Quant format token (e.g. `oq4++`, `mq4`, `qtip3`, `bf16`). Omit to be prompted from the known list
* `--detach` — Queue the induction as a background job instead of running it here, and return its id. Monitor it with `hipfire jobs watch <id>`.

   Both `source` and `--format` are required with this flag: a detached job has no terminal to prompt on.



## `hipfire model inspect`

Detail the contents of a `.hfq` container. Alias of top-level `hipfire inspect`; same arguments, same output

**Usage:** `hipfire model inspect [OPTIONS] <TARGET>`

###### **Arguments:**

* `<TARGET>` — Container to inspect: a `.hfq` file path or a local model alias

###### **Options:**

* `--tensors` — List every tensor (name, quant type, shape, group size, size)
* `--json` — Emit a machine-readable JSON object (includes the full tensor array and the raw metadata verbatim); ignores `--tensors`



## `hipfire diffusion`

Import, generate, and quantize diffusion models stored as .hfq artifacts

Inspection is not here: `hipfire inspect <artefact>` autodetects a diffusion container and prints its pipeline summary.

Runtime note: runnable `.hfq` diffusion artifacts still perform CLIP tokenization as host-side setup. `txt2img`, `img2img`, and `smoke` can opt into `--rocm-device-id` to route currently GPU-backed generation boundaries through ROCm. Diffusers cache discovery also lists transformer-denoiser pipelines such as Flux, Krea, Qwen Image, and Qwen Image Edit so clients can see convertible models, but native serving still requires a runnable `.hfq` artifact and a matching diffusion runtime.

`hipfire serve --model <diffusion.hfq>` pre-warms the resolved diffusion pipeline cache directly instead of routing the artifact through the chat daemon loader. The server exposes the same hybrid path through the Stable Diffusion API extension fields `rocm_device_id` or `hipfire_rocm_device_id` on `/sdapi/v1/txt2img` and `/sdapi/v1/img2img` requests, through the same keys in `override_settings`, or through the persisted `/sdapi/v1/options` value `hipfire_rocm_device_id`. Persisted `/sdapi/v1/options` values for `send_images`, `save_images`, `outdir_samples`, `outdir_txt2img_samples`, `outdir_img2img_samples`, `outdir_grids`, `outdir_txt2img_grids`, and `outdir_img2img_grids` act as generation defaults unless the request or `override_settings` supplies a more specific value.

`/sdapi/v1/progress` tracks active SDAPI sampling steps and updates `current_image` with live PNG previews decoded from intermediate latents, then leaves the final generated PNG there after a successful HFQ diffusion request completes; WebUI's `skip_current_image=true` progress query suppresses only that response's preview payload. The `/sdapi/v1/skip` endpoint records WebUI-compatible skip state without interrupting the whole request; `/sdapi/v1/interrupt` is the cancellation path. `/sdapi/v1/memory` returns WebUI-shaped host RAM stats and marks CUDA memory stats unavailable because Hipfire uses HIP/ROCm. WebUI's create/train embedding and hypernetwork endpoints are registered for client compatibility and return an `info` response explaining that native training is not implemented by the SDAPI layer. WebUI's optional server command endpoints (`server-kill`, `server-restart`, and `server-stop`) are registered as disabled compatibility no-ops so SDAPI clients do not see 404s, but external clients cannot stop or restart the Hipfire process through them.

SDAPI sampler fields follow WebUI's split controls: full scheduler names such as `DDIM`, `DPM++ 2M`, and `DPM++ 3M` are accepted directly, while schedule modifiers such as `Automatic` and `Karras` combine with `sampler_name` or `sampler_index` (for example `Euler` + `Karras` becomes `Euler Karras`).

SDAPI img2img and inpaint support WebUI resize modes 0 (stretch), 1 (crop and resize), 2 (resize and fill), and 3 (latent upscale). Modes 0-2 resize init and mask images before VAE encoding; mode 3 keeps the init image at its source dimensions, VAE-encodes it, then resizes the latent tensor to the requested output shape; `/sdapi/v1/latent-upscale-modes` advertises Hipfire's nearest-neighbor latent resize aliases. `seed_resize_from_w` and `seed_resize_from_h` generate the initial noise at the requested source dimensions and resize it to the target latent shape before sampling. Hipfire also accepts common WebUI generation fields such as `styles`, `restore_faces`, `tiling`, `eta`, `s_churn`, `s_tmin`, `s_tmax`, `s_noise`, `override_settings_restore_afterwards`, `disable_extra_networks`, and `comments`; fields that do not affect the native runtime are returned in response `parameters` and listed in `info.ignored_fields` when active. `do_not_save_samples` suppresses disk writes even when `save_images` is true. `return_grid` appends a generated batch grid to the response image list for multi-image outputs, and `do_not_save_grid` suppresses grid disk writes independently of sample writes. Masked img2img also honors WebUI's `inpainting_mask_invert`, `mask_blur`, `mask_blur_x`, `mask_blur_y`, `mask_round`, and `inpainting_fill` options; default fill (0) is applied in image space before VAE encode, original (1) leaves init pixels unchanged, and latent noise (2) / latent nothing (3) additionally alter masked latents. WebUI's `inpaint_full_res` and `inpaint_full_res_padding` crop masked regions for processing and composite the generated crop back onto the init image. SDAPI requests can also import common WebUI `infotext` fields when those fields are not explicitly set in JSON. Non-empty `script_name` and `script_args` payloads are rejected because Hipfire exposes no SDAPI selectable scripts. `alwayson_scripts` accepts empty or disabled default extension payloads, but active script payloads are rejected. Txt2img high-res generation is implemented as a batched first-pass txt2img generation followed by a second-pass img2img generation at the high-res target dimensions. SDAPI high-res requests accept `enable_hr`, `firstphase_width`, `firstphase_height`, `hr_scale`, `hr_upscaler`, `hr_resize_x`, `hr_resize_y`, `hr_second_pass_steps`, `hr_checkpoint_name`, `hr_prompt`, `hr_negative_prompt`, `hr_sampler_name`, and `hr_scheduler`; `hr_checkpoint_name` may point to another resolvable diffusion HFQ artifact for the second pass.

The runtime accepts Q4F16_G64, f16, bf16, f32, Q8F16, Q4_K, HFQ4G128, HFQ4G256, and HFQ6G256 tensor payloads. Other packed payloads require a matching diffusion dequantizer/runtime implementation.

**Usage:** `hipfire diffusion <COMMAND>`

###### **Subcommands:**

* `import` — Convert a Diffusers snapshot or single-file checkpoint into a Hipfire .hfq artifact
* `preflight` — Plan HIP diffusion buffers and optionally run a ROCm device preflight
* `txt2img` — Generate PNG images directly from a diffusion .hfq artifact
* `img2img` — Generate PNG images from init images with a diffusion .hfq artifact
* `smoke` — Run an end-to-end diffusion admission smoke and validate output PNGs
* `quantize` — Re-encode the weight tensors of a source .hfq into a packed quant format
* `calibrate` — Run an activation-calibration pass and write a .calib.hfq sidecar
* `quant-diff` — Compare per-tensor weight reconstruction error between two diffusion .hfq artifacts (e.g. a bf16 reference vs its quantized derivative)
* `calib-eval` — Quantify the activation-aware clip calibration ("+") on the fold format: for each fold-eligible transformer linear, report RTN vs clip weight-space error using a `.calib.hfq` imatrix. Weight-space only (no GPU)



## `hipfire diffusion import`

Convert a Diffusers snapshot or single-file checkpoint into a Hipfire .hfq artifact.

The importer extracts tensors from common Diffusers single-file and sharded safetensors layouts first, then falls back to legacy PyTorch .bin archives or opaque source weight entries when a component cannot be indexed yet.

**Usage:** `hipfire diffusion import [OPTIONS] --output <OUTPUT> <SOURCE>`

###### **Arguments:**

* `<SOURCE>` — Diffusers snapshot directory containing model_index.json, or a .safetensors/.ckpt checkpoint

###### **Options:**

* `-o`, `--output <OUTPUT>` — Output .hfq artifact path
* `--model-name <MODEL_NAME>` — Model name to store in the diffusion metadata; defaults to the source directory name
* `--max-batch <MAX_BATCH>` — Maximum batch size declared by the artifact. Runtime kernels may cap this lower initially

  Default value: `1`
* `--metadata-only` — Import configs/tokenizers only and skip weight indexing for fast planning/inspection



## `hipfire diffusion preflight`

Plan HIP diffusion buffers and optionally run a ROCm device preflight

The preflight command prints a deterministic memory plan for the requested resolution, batch, scheduler, and prompt set. When a `--device-id` is given it also initializes the selected HIP device, allocates the planned buffer classes, runs a host/device roundtrip probe, and launches the diffusion kernel probes against CPU references.

**Usage:** `hipfire diffusion preflight [OPTIONS] --model <MODEL>`

###### **Options:**

* `-m`, `--model <MODEL>` — Diffusion .hfq artifact to inspect by name, shorthand, alias, or path
* `-p`, `--prompt <PROMPT>` — Prompt text. Repeat for batched planning, or use --batch-size with one prompt

  Default value: `hipfire diffusion preflight`
* `--negative-prompt <NEGATIVE_PROMPT>` — Negative prompt text. Omit for empty negatives, pass once to reuse, or repeat per prompt
* `--width <WIDTH>` — Output image width in pixels

  Default value: `512`
* `--height <HEIGHT>` — Output image height in pixels

  Default value: `512`
* `--steps <STEPS>` — Denoising steps

  Default value: `20`
* `--cfg-scale <CFG_SCALE>` — Classifier-free guidance scale

  Default value: `7`
* `--distilled-guidance-scale <DISTILLED_GUIDANCE_SCALE>` — Guidance-distilled model scale, separate from classifier-free guidance
* `--scheduler <SCHEDULER>` — Scheduler/sampler name

  Default value: `Automatic`
* `--seed <SEED>` — Seed. Omit for zero, pass once to reuse, or repeat per prompt
* `--subseed <SUBSEED>` — Optional subseed. Pass once to reuse or repeat per prompt
* `--subseed-strength <SUBSEED_STRENGTH>` — Blend strength for subseed latents

  Default value: `0`
* `--batch-size <BATCH_SIZE>` — Batch size when a single prompt is supplied

  Default value: `1`
* `--device-id <DEVICE_ID>` — ROCm device id to initialize and preflight (omit for plan-only)

  Default value: `0`



## `hipfire diffusion txt2img`

Generate PNG images directly from a diffusion .hfq artifact

With `--enable-hr`, the command first generates the requested base batch, decodes those PNGs as init images, then runs an img2img second pass at `--hr-scale` or the `--hr-resize-x`/`--hr-resize-y` target.

**Usage:** `hipfire diffusion txt2img [OPTIONS] --model <MODEL> --prompt <PROMPT> --output <OUTPUT>`

###### **Options:**

* `-m`, `--model <MODEL>` — Diffusion .hfq artifact to run by name, shorthand, alias, or path
* `-p`, `--prompt <PROMPT>` — Prompt text. Repeat for batched generation, or use --batch-size with one prompt
* `--negative-prompt <NEGATIVE_PROMPT>` — Negative prompt text. Omit for empty negatives, pass once to reuse, or repeat per prompt
* `-o`, `--output <OUTPUT>` — Output PNG file for one image, or output directory for batches
* `--preview-dir <PREVIEW_DIR>` — Directory to write a per-step preview PNG (step_00.png, step_01.png, ...) by decoding the intermediate latent after each denoise pass. Useful for a webui progress strip; adds one VAE decode per step. Single-image runs only
* `--width <WIDTH>` — Output image width in pixels

  Default value: `512`
* `--height <HEIGHT>` — Output image height in pixels

  Default value: `512`
* `--firstphase-width <FIRSTPHASE_WIDTH>` — First-pass high-res width before upscale; preserves --width/--height aspect when used alone
* `--firstphase-height <FIRSTPHASE_HEIGHT>` — First-pass high-res height before upscale; preserves --width/--height aspect when used alone
* `--steps <STEPS>` — Denoising steps

  Default value: `20`
* `--cfg-scale <CFG_SCALE>` — Classifier-free guidance scale

  Default value: `7`
* `--distilled-guidance-scale <DISTILLED_GUIDANCE_SCALE>` — Guidance-distilled model scale, separate from classifier-free guidance
* `--scheduler <SCHEDULER>` — Scheduler/sampler name, such as Automatic, Euler, Euler Karras, DDIM, DPM++ 2M Karras, or DPM++ 3M Karras

  Default value: `Automatic`
* `--seed <SEED>` — Seed. Omit for zero, pass once to reuse, or repeat per prompt
* `--subseed <SUBSEED>` — Optional subseed. Pass once to reuse or repeat per prompt
* `--subseed-strength <SUBSEED_STRENGTH>` — Blend strength for subseed latents

  Default value: `0`
* `--batch-size <BATCH_SIZE>` — Batch size when a single prompt is supplied

  Default value: `1`
* `--enable-hr` — Run a high-res second pass by feeding first-pass txt2img results through img2img
* `--hr-scale <HR_SCALE>` — High-res scale when --hr-resize-x/--hr-resize-y are both omitted or zero

  Default value: `2`
* `--hr-resize-x <HR_RESIZE_X>` — Exact high-res target width, or aspect-preserving width when used alone
* `--hr-resize-y <HR_RESIZE_Y>` — Exact high-res target height, or aspect-preserving height when used alone
* `--hr-second-pass-steps <HR_SECOND_PASS_STEPS>` — Denoising steps for the high-res second pass; defaults to --steps
* `--hr-denoising-strength <HR_DENOISING_STRENGTH>` — Img2img denoising strength for the high-res second pass

  Default value: `0.75`
* `--rocm-device-id <ROCM_DEVICE_ID>` — Use ROCm for currently GPU-routed generation stages on this device id ROCm device to generate on. Omit to auto-detect (a single GPU is used silently; the first of several with a warning). The CPU reference oracle is opt-in via the HIPFIRE_DIFFUSION_CPU_REFERENCE environment variable
* `--mrflow <MRFLOW>` — Enable MrFlow staged sampling: a fast low-resolution pass, pixel-space super-resolution, re-encode, and a short direct-sigma refine. --width and --height are the final resolution; the low-res pass runs at those divided by the upscale factor. Flow-match backbones only (FLUX / Qwen-Image / Z-Image / Krea-2). Overrides --enable-hr

  Possible values:
  - `zit-9plus1`:
    Z-Image Turbo, 9 low-res + 1 refine, sigma 0.11, no CFG (paper demo)
  - `krea2-12plus1`:
    Krea-2 base, 12 low-res + 1 refine, sigma 0.12, cfg 4.0
  - `krea2-20plus1`:
    Krea-2 base, 20 low-res + 1 refine, sigma 0.15, cfg 4.0
  - `krea2-turbo-8plus1`:
    Krea-2 Turbo, 8 low-res + 1 refine, sigma 0.11, no CFG

* `--mrflow-total-steps <MRFLOW_TOTAL_STEPS>` — Override the total MrFlow denoise budget across the low-resolution and refine passes. The preset's refine count is reserved first; for example, 8 total steps with a 1-step refine runs 7+1
* `--mrflow-refine-sigma <MRFLOW_REFINE_SIGMA>` — Override the MrFlow refine start sigma (preset default). Larger values (0.16-0.20) can improve text-heavy generations
* `--mrflow-upscale <MRFLOW_UPSCALE>` — Override the MrFlow pixel-space upscale factor (preset default 2.0)
* `--mrflow-shifted` — Use the flow-match shifted interior refine schedule (only affects refine passes with more than one step)
* `--mrflow-sr <MRFLOW_SR>` — RealESRGAN RRDBNet super-resolution .hfq (from `hipfire-coexistence`) for the MrFlow Stage-2 upscale. Without it, Stage 2 falls back to a plain cover-resize (much softer output)



## `hipfire diffusion img2img`

Generate PNG images from init images with a diffusion .hfq artifact

**Usage:** `hipfire diffusion img2img [OPTIONS] --model <MODEL> --prompt <PROMPT> --init-image <INIT_IMAGE> --output <OUTPUT>`

###### **Options:**

* `-m`, `--model <MODEL>` — Diffusion .hfq artifact to run by name, shorthand, alias, or path
* `-p`, `--prompt <PROMPT>` — Prompt text. Repeat for batched generation, or use --batch-size with one prompt
* `--negative-prompt <NEGATIVE_PROMPT>` — Negative prompt text. Omit for empty negatives, pass once to reuse, or repeat per prompt
* `--init-image <INIT_IMAGE>` — Input image path. Repeat for an image batch, or pass once to reuse across prompts
* `--mask <MASK>` — Optional mask image path for inpaint-capable artifacts
* `-o`, `--output <OUTPUT>` — Output PNG file for one image, or output directory for batches
* `--width <WIDTH>` — Output image width in pixels. Defaults to the init image width
* `--height <HEIGHT>` — Output image height in pixels. Defaults to the init image height
* `--steps <STEPS>` — Denoising steps

  Default value: `20`
* `--cfg-scale <CFG_SCALE>` — Classifier-free guidance scale

  Default value: `7`
* `--distilled-guidance-scale <DISTILLED_GUIDANCE_SCALE>` — Guidance-distilled model scale, separate from classifier-free guidance
* `--scheduler <SCHEDULER>` — Scheduler/sampler name, such as Automatic, Euler, Euler Karras, DDIM, DPM++ 2M Karras, or DPM++ 3M Karras

  Default value: `Automatic`
* `--seed <SEED>` — Seed. Omit for zero, pass once to reuse, or repeat per prompt
* `--subseed <SUBSEED>` — Optional subseed. Pass once to reuse or repeat per prompt
* `--subseed-strength <SUBSEED_STRENGTH>` — Blend strength for subseed latents

  Default value: `0`
* `--batch-size <BATCH_SIZE>` — Batch size when a single prompt is supplied

  Default value: `1`
* `--denoising-strength <DENOISING_STRENGTH>` — Img2img denoising strength in [0, 1]

  Default value: `0.75`
* `--rocm-device-id <ROCM_DEVICE_ID>` — Use ROCm for currently GPU-routed generation stages on this device id ROCm device to generate on. Omit to auto-detect (a single GPU is used silently; the first of several with a warning). The CPU reference oracle is opt-in via the HIPFIRE_DIFFUSION_CPU_REFERENCE environment variable



## `hipfire diffusion smoke`

Run an end-to-end diffusion admission smoke and validate output PNGs

**Usage:** `hipfire diffusion smoke [OPTIONS] --model <MODEL>`

###### **Options:**

* `-m`, `--model <MODEL>` — Diffusion .hfq artifact to run by name, shorthand, alias, or path
* `-p`, `--prompt <PROMPT>` — Prompt text for the smoke run

  Default value: `hipfire diffusion smoke test`
* `--negative-prompt <NEGATIVE_PROMPT>` — Negative prompt text

  Default value: ``
* `--output-dir <OUTPUT_DIR>` — Output directory for smoke PNGs

  Default value: `/tmp/hipfire-diffusion-smoke`
* `--width <WIDTH>` — Output image width in pixels

  Default value: `64`
* `--height <HEIGHT>` — Output image height in pixels

  Default value: `64`
* `--steps <STEPS>` — Denoising steps

  Default value: `1`
* `--cfg-scale <CFG_SCALE>` — Classifier-free guidance scale

  Default value: `1`
* `--distilled-guidance-scale <DISTILLED_GUIDANCE_SCALE>` — Guidance-distilled model scale, separate from classifier-free guidance
* `--scheduler <SCHEDULER>` — Scheduler/sampler name

  Default value: `Euler`
* `--seed <SEED>` — Seed

  Default value: `0`
* `--batch-size <BATCH_SIZE>` — Batch size for each smoke leg

  Default value: `1`
* `--denoising-strength <DENOISING_STRENGTH>` — Img2img denoising strength

  Default value: `0.5`
* `--rocm-device-id <ROCM_DEVICE_ID>` — Use ROCm for currently GPU-routed generation stages on this device id ROCm device to generate on. Omit to auto-detect (a single GPU is used silently; the first of several with a warning). The CPU reference oracle is opt-in via the HIPFIRE_DIFFUSION_CPU_REFERENCE environment variable
* `--txt2img-only` — Only run txt2img; skip the img2img leg
* `--skip-masked-img2img` — Skip the masked img2img leg



## `hipfire diffusion quantize`

Re-encode the weight tensors of a source .hfq into a packed quant format

Reads an existing diffusion .hfq (weights stored as f32/f16/bf16 source), re-encodes the large 2D+ `.weight` tensors into the requested format, and copies every other entry (biases, norms, configs, tokenizers) verbatim. Decoding is per-tensor by quant_type, so the output loads unchanged.

**Usage:** `hipfire diffusion quantize [OPTIONS] --output <OUTPUT> <SOURCE>`

###### **Arguments:**

* `<SOURCE>` — Source diffusion .hfq artifact (typically `weight_format: source`)

###### **Options:**

* `-o`, `--output <OUTPUT>` — Output quantized .hfq artifact path
* `--format <FORMAT>` — Quant format: q8, q4, q4k, q4+, oq4/oq4+/oq4++/oq8 (rotated), oq4p/oq8p (plain), a decimal plain-Opus target such as oq4.25, or oq4-mixed for the legacy data-free heuristic. Plain Opus uses int8 activations

  Default value: `q8`
* `--calib <CALIB>` — Optional .calib.hfq sidecar (from `diffusion calibrate`); enables oq4++ LDLQ
* `--mix-fraction <MIX_FRACTION>` — For plain-Opus mixed precision: fraction (0.0–1.0) of quantized parameters to place at int8 (highest fan-in first), the rest int4. Overrides the format to mixed; achieved average ≈ 4 + 4·fraction bits. The output name is rewritten to the achieved `oq<avg>` token
* `--arch-importance` — Rank the int8 promotion by the arch's structural importance prior (embedders/attention/modulation/output over the FFN bulk) instead of the default highest-fan-in heuristic. Same bit budget; different tensor selection. Only affects `--mix-fraction` (plain-Opus mixed)



## `hipfire diffusion calibrate`

Run an activation-calibration pass and write a .calib.hfq sidecar

Generates a few instrumented denoise steps over sample prompts, capturing per-weight activation statistics (imatrix + per-linear Hessian). The resulting .calib.hfq feeds `quantize --format oq4++ --calib`.

**Usage:** `hipfire diffusion calibrate [OPTIONS] --output <OUTPUT> <MODEL>`

###### **Arguments:**

* `<MODEL>` — Source diffusion .hfq artifact to calibrate

###### **Options:**

* `-o`, `--output <OUTPUT>` — Output .calib.hfq sidecar path
* `-p`, `--prompt <PROMPTS>` — Calibration prompts (repeatable); defaults to a small built-in set
* `--steps <STEPS>` — Denoise steps per prompt

  Default value: `4`
* `--width <WIDTH>`

  Default value: `256`
* `--height <HEIGHT>`

  Default value: `256`
* `--cfg-scale <CFG_SCALE>` — CFG scale (>1 captures both conditional and unconditional activations)

  Default value: `7.5`
* `--hessian-max-k <HESSIAN_MAX_K>` — Max linear input dim K to capture a full [K,K] Hessian for (else imatrix only)

  Default value: `2048`
* `--rocm-device-id <ROCM_DEVICE_ID>` — ROCm device used for instrumented resident calibration



## `hipfire diffusion quant-diff`

Compare per-tensor weight reconstruction error between two diffusion .hfq artifacts (e.g. a bf16 reference vs its quantized derivative)

Decodes every quantizable `transformer/tensors/*.weight` from both artifacts to f32 and reports per-tensor error, ranked by relative L2. This is the sampler-independent quant-quality check: if the worst tensor is near-lossless, any rendered-image drift is trajectory divergence, not weight corruption. Pairs with `scripts/flux2_trajectory_divergence.py`.

**Usage:** `hipfire diffusion quant-diff [OPTIONS] <REFERENCE> <CANDIDATE>`

###### **Arguments:**

* `<REFERENCE>` — Reference artifact (typically the bf16 / p0 source .hfq)
* `<CANDIDATE>` — Candidate artifact (typically the quantized .hfq, e.g. --oq8.hfq)

###### **Options:**

* `--top <TOP>` — Print the N worst tensors by relative L2 error

  Default value: `20`
* `--rel-rms-threshold <REL_RMS_THRESHOLD>` — Relative-L2 threshold above which a tensor is flagged as real corruption

  Default value: `0.05`
* `--json` — Emit the full per-tensor diff as JSON instead of a table



## `hipfire diffusion calib-eval`

Quantify the activation-aware clip calibration ("+") on the fold format: for each fold-eligible transformer linear, report RTN vs clip weight-space error using a `.calib.hfq` imatrix. Weight-space only (no GPU)

**Usage:** `hipfire diffusion calib-eval [OPTIONS] <SOURCE> <CALIB>`

###### **Arguments:**

* `<SOURCE>` — Source diffusion .hfq (bf16 weights)
* `<CALIB>` — Calibration sidecar (.calib.hfq) with per-tensor imatrix

###### **Options:**

* `--bits <BITS>` — Fold bit width to evaluate (1/2/4)

  Default value: `4`
* `--json` — Emit JSON instead of a table



## `hipfire admin`

Query the running hipfire admin API for scripts and agents

**Usage:** `hipfire admin [OPTIONS] <COMMAND>`

###### **Subcommands:**

* `status` — Combined status snapshot for scripts and agents
* `chat` — Send one non-streaming chat request through /v1/chat/completions
* `health` — Raw /health payload
* `models` — Local model registry from the admin API
* `config` — Resolved runtime config
* `training` — Training run summaries or one run detail
* `diagnostics` — Filesystem, binary, kernel-cache, lock, and log diagnostics
* `logs` — Tail known hipfire logs
* `get` — GET an arbitrary admin/server path, e.g. /admin/training/runs
* `set-password` — Set the /admin console password (argon2id hash -> ~/.hipfire/admin.passwd)

###### **Options:**

* `--host <HOST>` — Override admin API host. Defaults to config host, with 0.0.0.0 mapped to 127.0.0.1
* `--port <PORT>` — Override admin API port. Defaults to config port



## `hipfire admin status`

Combined status snapshot for scripts and agents

**Usage:** `hipfire admin status`



## `hipfire admin chat`

Send one non-streaming chat request through /v1/chat/completions

**Usage:** `hipfire admin chat [OPTIONS] <PROMPT>...`

###### **Arguments:**

* `<PROMPT>` — User prompt text

###### **Options:**

* `--model <MODEL>` — Model name, shorthand, alias, or path. Defaults to server config when omitted
* `--system <SYSTEM>` — Optional system message
* `--max-tokens <MAX_TOKENS>` — Max tokens to generate
* `--temperature <TEMPERATURE>` — Sampling temperature
* `--top-p <TOP_P>` — Nucleus sampling top-p
* `--text` — Print only the assistant message text



## `hipfire admin health`

Raw /health payload

**Usage:** `hipfire admin health`



## `hipfire admin models`

Local model registry from the admin API

**Usage:** `hipfire admin models`



## `hipfire admin config`

Resolved runtime config

**Usage:** `hipfire admin config [OPTIONS]`

###### **Options:**

* `--model <MODEL>` — Resolve config for a specific model name, shorthand, alias, or path



## `hipfire admin training`

Training run summaries or one run detail

**Usage:** `hipfire admin training [OPTIONS] [ID]`

###### **Arguments:**

* `<ID>` — Optional run ID

###### **Options:**

* `--events` — Return full events for the run ID



## `hipfire admin diagnostics`

Filesystem, binary, kernel-cache, lock, and log diagnostics

**Usage:** `hipfire admin diagnostics`



## `hipfire admin logs`

Tail known hipfire logs

**Usage:** `hipfire admin logs [OPTIONS]`

###### **Options:**

* `--lines <LINES>` — Number of lines per log file

  Default value: `120`



## `hipfire admin get`

GET an arbitrary admin/server path, e.g. /admin/training/runs

**Usage:** `hipfire admin get <PATH>`

###### **Arguments:**

* `<PATH>` — Absolute or relative server path



## `hipfire admin set-password`

Set the /admin console password (argon2id hash -> ~/.hipfire/admin.passwd)

**Usage:** `hipfire admin set-password [PASSWORD]`

###### **Arguments:**

* `<PASSWORD>` — New password. If omitted, read once from stdin (no echo when a TTY)



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>
