# AGENTS.md - hipfire-daemon

The daemon owns runtime resource leases before HIP initialization. Keep resource
locking deterministic and compatible with CLI GPU-lock workflows.

## Resource Locks

- `hipfire-daemon` acquires `flock(2)` GPU/NPU/CPU leases before HIP init.
- Single-GPU coordination must share the same inode as the `hipfire lock` CLI:
  `gpu_resource_lock_path()` = `resource_lock_path("hip-gpu-0")` →
  `~/.hipfire/locks/hip-gpu-0.lock`. The former separate `gpu_lock_path()` /
  `$HIPFIRE_GPU_LOCKFILE` file is **removed** — it did not share an inode with
  this lease, so it never actually coordinated with the daemon.
- Multi-GPU, NPU, and CPU leases use `<resource_lock_root()>/<resource>.lock`,
  i.e. `$HIPFIRE_RESOURCE_LOCK_DIR`, else `~/.hipfire/locks`, else
  `<tmpdir>/hipfire-resource-locks` only when `$HOME` is unset.
- `HIPFIRE_RESOURCE_LOCK_WAIT_MS>0` waits for a busy lease; `0` fails fast.
- Do not add stale-lock cleanup that fights `flock`; the kernel releases the
  lock on process death.
