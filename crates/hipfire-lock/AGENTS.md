# AGENTS.md - hipfire-lock

This crate owns the shared `flock(2)` primitive and the single-GPU lockfile
contract used by `hipfire lock` and non-daemon GPU workflows.

## Lock Contract

- The single-GPU mutex path is `gpu_resource_lock_path()`, which is exactly
  `resource_lock_path("hip-gpu-0")` → `~/.hipfire/locks/hip-gpu-0.lock`.
- Per-resource lockfiles live under `resource_lock_root()`:
  `$HIPFIRE_RESOURCE_LOCK_DIR`, else `~/.hipfire/locks`, else
  `<tmpdir>/hipfire-resource-locks` only when `$HOME` is unset.
- **There is no `gpu_lock_path()` and no `$HIPFIRE_GPU_LOCKFILE`.** They were a
  *separate* GPU lockfile that did not share an inode with the daemon's
  `hip-gpu-0` resource lease, so the two never coordinated. Both are removed.
  This matters more than a renamed path: `flock` keys on the **inode**, so a
  caller pointed at the wrong file still acquires successfully and simply fails
  to exclude anyone — a silent loss of coordination, not an error.
- Keep the lockfile inode stable. Do not unlink a flocked lockfile as a release
  mechanism.
- `flock` releases on process death, including crash or SIGKILL. Avoid stale-lock
  cleanup schemes that assume pid files are authoritative.
- Changes here must stay compatible with `crates/hipfire-cli` and tests/scripts
  that call `hipfire lock {acquire,release,status}` (`gpu-lock` is only an alias).
- The holder line written *into* the lockfile is file contents, not kernel state,
  so it outlives its writer — and it is what a contention error names. A stale
  line makes a real conflict point at a dead pid; `hipfire lock kill` clears it
  when it can prove the writer is gone.
