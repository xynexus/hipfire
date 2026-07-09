// SPDX-License-Identifier: Apache-2.0
//! AdamW optimizer (fp32, decoupled weight decay) for the training path.
//!
//! Owns the per-parameter `m`/`v` moment buffers and the step counter, computes
//! bias corrections, and applies `Gpu::adamw_step`. Matches `sft.py`'s
//! AdamW(β1=0.9, β2=0.999, eps=1e-8, wd=0). `set_lr` supports a schedule.

use hipfire_rdna::{DType, Gpu, GpuTensor, HipResult};

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    t: i32,
    m: Vec<GpuTensor>,
    v: Vec<GpuTensor>,
    numel: Vec<usize>,
}

impl AdamW {
    /// Allocate zeroed moment state for params of the given element counts.
    /// The order of `sizes` fixes the param order used by `step`.
    pub fn new(
        gpu: &mut Gpu,
        sizes: &[usize],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) -> HipResult<Self> {
        let mut m = Vec::with_capacity(sizes.len());
        let mut v = Vec::with_capacity(sizes.len());
        for &n in sizes {
            m.push(gpu.zeros(&[n], DType::F32)?);
            v.push(gpu.zeros(&[n], DType::F32)?);
        }
        Ok(Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m,
            v,
            numel: sizes.to_vec(),
        })
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    /// Global-norm gradient clipping: if the L2 norm over ALL `grads` exceeds
    /// `max_norm`, scale every grad by `max_norm / norm` in place. Returns the
    /// pre-clip global norm (for logging). `max_norm <= 0` disables (returns 0).
    ///
    /// Essential for a bigger drafter whose conditioning input (target hidden
    /// states) carries gemma-style ~1e6 activation outliers — without it the
    /// fc-ingest gradient explodes and training diverges at every LR.
    pub fn clip_grad_global_norm(
        &self,
        gpu: &mut Gpu,
        grads: &[&GpuTensor],
        max_norm: f32,
    ) -> HipResult<f32> {
        if !(max_norm > 0.0) {
            return Ok(0.0);
        }
        let acc = gpu.zeros(&[1], DType::F32)?;
        for g in grads {
            gpu.sum_sq_accum_f32(g, &acc)?;
        }
        let total = gpu.download_f32(&acc)?[0];
        gpu.free_tensor(acc)?;
        let norm = total.sqrt();
        let clipped = norm.is_finite() && norm > max_norm;
        if clipped {
            let factor = max_norm / norm;
            for g in grads {
                gpu.scale_f32(g, factor)?;
            }
        }
        if std::env::var("HIPFIRE_TRAIN_GRAD_LOG").is_ok() {
            eprintln!(
                "  grad_global_norm = {norm:.4}  sumsq = {total:.4e}  clip@{max_norm} -> {}",
                if clipped { "CLIPPED" } else { "no-op" }
            );
        }
        Ok(norm)
    }

    /// One step over all params. `params[i]` is updated in place from
    /// `grads[i]`; both must match the construction order/sizes.
    pub fn step(
        &mut self,
        gpu: &mut Gpu,
        params: &[&GpuTensor],
        grads: &[&GpuTensor],
    ) -> HipResult<()> {
        assert_eq!(params.len(), self.numel.len());
        assert_eq!(grads.len(), self.numel.len());
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);
        for i in 0..params.len() {
            gpu.adamw_step(
                params[i],
                grads[i],
                &self.m[i],
                &self.v[i],
                self.numel[i],
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                bc1,
                bc2,
            )?;
        }
        Ok(())
    }

    pub fn step_count(&self) -> i32 {
        self.t
    }

    /// Download the optimizer state (per-param `m`, `v`, and the step counter) to
    /// host — for checkpointing. `m[i]`/`v[i]` match the construction order.
    pub fn save_state(&self, gpu: &mut Gpu) -> HipResult<(Vec<Vec<f32>>, Vec<Vec<f32>>, i32)> {
        let m = self
            .m
            .iter()
            .map(|t| gpu.download_f32(t))
            .collect::<HipResult<Vec<_>>>()?;
        let v = self
            .v
            .iter()
            .map(|t| gpu.download_f32(t))
            .collect::<HipResult<Vec<_>>>()?;
        Ok((m, v, self.t))
    }

    /// Restore optimizer state from host buffers (resume). Sizes/order must match
    /// construction. Overwrites the existing device moment buffers in place.
    pub fn load_state(
        &mut self,
        gpu: &mut Gpu,
        m: &[Vec<f32>],
        v: &[Vec<f32>],
        t: i32,
    ) -> HipResult<()> {
        assert_eq!(m.len(), self.m.len(), "AdamW m count mismatch on resume");
        assert_eq!(v.len(), self.v.len(), "AdamW v count mismatch on resume");
        for (i, h) in m.iter().enumerate() {
            assert_eq!(h.len(), self.numel[i], "AdamW m[{i}] size mismatch");
            gpu.memcpy_htod_auto(&self.m[i].buf, bytemuck_f32(h))?;
        }
        for (i, h) in v.iter().enumerate() {
            assert_eq!(h.len(), self.numel[i], "AdamW v[{i}] size mismatch");
            gpu.memcpy_htod_auto(&self.v[i].buf, bytemuck_f32(h))?;
        }
        self.t = t;
        Ok(())
    }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
