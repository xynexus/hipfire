// Count NPU command submissions by interposing ioctl() on the amdxdna driver.
//
// Memory scanning proved FLM's per-layer dispatch is not resident as a
// transaction binary (docs/npu/flm-refe-log.md, 2026-07-31). What is still
// unknown is how MANY commands FLM submits per token, which distinguishes two
// very different pictures:
//
//   ~1 submission per layer  -> a real fused per-layer instruction stream
//                               exists and reaches the device by some path that
//                               never passes through readable userspace.
//   ~9-10 per layer          -> there is no fused stream; a "layer" is repeated
//                               re-invocation of the generic DDR<->memtile
//                               staging ladder with patched buffer addresses,
//                               while the compute cores run persistent programs
//                               loaded once from the PDI.
//
// llama3.2:1b is 16 layers and streams 38.0 MB of weights per layer; the
// 8-column staging transaction moves 4 MB. So the second picture predicts
// roughly 9-10 x 16 = ~150 submissions per token.
//
//   gcc -shared -fPIC -O2 -o npu_ioctl_count.so npu_ioctl_count.c -ldl \
//       -I ~/xdna-driver/include/uapi
//   NPU_COUNT_OUT=/tmp/counts.txt LD_PRELOAD=$PWD/npu_ioctl_count.so flm run llama3.2:1b
//
// The include path must be .../include/uapi, not .../include. With the latter,
// <drm/amdxdna_accel.h> resolves to the SYSTEM header /usr/include/drm/, which
// is older and has no DRM_IOCTL_AMDXDNA_WAIT_CMD -- the build fails on that one
// symbol while EXEC_CMD resolves fine, which makes it look like a typo rather
// than the wrong header.
//
// Counters are printed at exit, and on SIGUSR1 so a long-running process can be
// sampled without killing it. Read-only: every call is forwarded unmodified.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <drm/amdxdna_accel.h>

static int (*real_ioctl)(int, unsigned long, ...);

// Relaxed (not atomic): exact ordering does not matter for a count, and we must
// not perturb the timing of the thing being measured.
static volatile unsigned long n_exec, n_wait, n_sync, n_create_bo, n_other_drm;
static volatile unsigned long n_cmds;   // sum of cmd_count over all EXEC_CMD
static volatile unsigned long n_args;   // sum of arg_count, ~ buffers per command

static void report(const char *why) {
  const char *path = getenv("NPU_COUNT_OUT");
  FILE *f = stderr;
  if (path) {
    FILE *g = fopen(path, "a");
    if (g) f = g;
  }
  fprintf(f,
          "[npu_ioctl_count %s pid=%d] EXEC_CMD=%lu (cmds=%lu args=%lu) "
          "WAIT=%lu SYNC_BO=%lu CREATE_BO=%lu other_drm=%lu\n",
          why, (int)getpid(), n_exec, n_cmds, n_args, n_wait, n_sync,
          n_create_bo, n_other_drm);
  fflush(f);
  if (f != stderr) fclose(f);
}

static void on_sigusr1(int sig) { (void)sig; report("sample"); }

// Per-submission trace: NPU_TRACE_OUT=<path> [NPU_TRACE_N=<count>].
// Columns: seq hwctx type cmd_count arg_count
static FILE *trace;
static unsigned long trace_n = 4000;

// Handle -> (size, type) map, populated from CREATE_BO, so a submission's
// argument list can be reported as buffer sizes rather than opaque handles.
// Open-addressed fixed table: no allocation on the traced path, and a lost
// entry degrades to "size unknown" rather than misreporting.
#define BOMAP_BITS 16
#define BOMAP_SIZE (1u << BOMAP_BITS)
static struct { uint32_t handle, type; uint64_t size; } bomap[BOMAP_SIZE];

static void bo_put(uint32_t h, uint64_t sz, uint32_t ty) {
  uint32_t i = (h * 2654435761u) & (BOMAP_SIZE - 1);
  for (uint32_t n = 0; n < 64; n++, i = (i + 1) & (BOMAP_SIZE - 1)) {
    if (bomap[i].handle == 0 || bomap[i].handle == h) {
      bomap[i].handle = h; bomap[i].size = sz; bomap[i].type = ty;
      return;
    }
  }
}

static int bo_get(uint32_t h, uint64_t *sz, uint32_t *ty) {
  uint32_t i = (h * 2654435761u) & (BOMAP_SIZE - 1);
  for (uint32_t n = 0; n < 64; n++, i = (i + 1) & (BOMAP_SIZE - 1)) {
    if (bomap[i].handle == h) { *sz = bomap[i].size; *ty = bomap[i].type; return 1; }
    if (bomap[i].handle == 0) return 0;
  }
  return 0;
}

// Dump the argument buffer list of the first N submissions of a given
// arg_count, so the 50-argument fused decode command can be decomposed.
static FILE *argf;
static unsigned long argdump_want, argdump_left;

__attribute__((constructor)) static void init(void) {
  real_ioctl = dlsym(RTLD_NEXT, "ioctl");
  signal(SIGUSR1, on_sigusr1);
  const char *tp = getenv("NPU_TRACE_OUT");
  if (tp) {
    trace = fopen(tp, "w");
    const char *tn = getenv("NPU_TRACE_N");
    if (tn) trace_n = strtoul(tn, NULL, 10);
  }
  const char *ap = getenv("NPU_ARGS_OUT");
  if (ap) {
    argf = fopen(ap, "w");
    const char *an = getenv("NPU_ARGS_MATCH");   // only dump submissions with this arg_count
    argdump_want = an ? strtoul(an, NULL, 10) : 0;
    const char *ac = getenv("NPU_ARGS_COUNT");   // how many such submissions
    argdump_left = ac ? strtoul(ac, NULL, 10) : 2;
  }
}

__attribute__((destructor)) static void fini(void) {
  report("exit");
  if (trace) { fflush(trace); fclose(trace); trace = NULL; }
}

int ioctl(int fd, unsigned long request, ...) {
  va_list ap;
  va_start(ap, request);
  void *arg = va_arg(ap, void *);
  va_end(ap);

  if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");

  // CREATE_BO fills in the handle on return, so it has to be recorded after the
  // call, not before.
  if (request == DRM_IOCTL_AMDXDNA_CREATE_BO) {
    int rc = real_ioctl(fd, request, arg);
    struct amdxdna_drm_create_bo *c = (struct amdxdna_drm_create_bo *)arg;
    n_create_bo++;
    if (rc == 0 && c) bo_put(c->handle, c->size, c->type);
    return rc;
  }

  switch (request) {
  case DRM_IOCTL_AMDXDNA_EXEC_CMD: {
    struct amdxdna_drm_exec_cmd *e = (struct amdxdna_drm_exec_cmd *)arg;
    n_exec++;
    if (e) {
      // cmd_count is the number of command handles in one submission, so a
      // command chain counts as its length, not as one.
      n_cmds += e->cmd_count ? e->cmd_count : 1;
      n_args += e->arg_count;
      // Trace the shape of each submission. Two commands per token could be two
      // invocations of one kernel or one invocation each of two different
      // kernels; a repeating (hwctx, type, arg_count) pattern distinguishes
      // them, and arg_count is a proxy for how many buffers a phase touches.
      if (trace && n_exec <= trace_n)
        fprintf(trace, "%lu %u %u %u %u\n", n_exec, e->hwctx, e->type,
                e->cmd_count, e->arg_count);

      // The argument list is a user pointer to arg_count u32 BO handles
      // (amdxdna_ctx.c: copy_from_user(arg_bo_hdls, ..., arg_count * sizeof(u32))).
      if (argf && argdump_left && e->args && e->arg_count &&
          (!argdump_want || e->arg_count == argdump_want)) {
        const uint32_t *h = (const uint32_t *)(uintptr_t)e->args;
        fprintf(argf, "# submission %lu hwctx=%u arg_count=%u\n",
                n_exec, e->hwctx, e->arg_count);
        for (uint32_t k = 0; k < e->arg_count; k++) {
          uint64_t sz = 0; uint32_t ty = 0;
          int known = bo_get(h[k], &sz, &ty);
          fprintf(argf, "%u %u %llu %u %s\n", k, h[k],
                  (unsigned long long)sz, ty, known ? "" : "unknown");
        }
        fflush(argf);
        argdump_left--;
      }
    }
    break;
  }
  case DRM_IOCTL_AMDXDNA_WAIT_CMD:   n_wait++; break;
  case DRM_IOCTL_AMDXDNA_SYNC_BO:    n_sync++; break;
  default:
    // Cheap check for "some other DRM ioctl", to confirm we are on the right fd
    // even when the counters above stay at zero.
    if (((request >> 8) & 0xFF) == 'd') n_other_drm++;
    break;
  }

  return real_ioctl(fd, request, arg);
}
