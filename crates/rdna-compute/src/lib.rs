//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

mod compiler;
mod dispatch;
mod kernels;
mod mq4_i4_dot8;
pub mod pool;
pub mod profile;
pub mod profiler;

pub use compiler::KernelCompiler;
pub use dispatch::{
    gemv_dp4a_enabled, mq4_i4_dot8_down_enabled, mq4_i4_dot8_gemv_enabled,
    mq4_i4_dot8_residual_enabled, mq4_i4_dot8_wo_enabled, DType, Gpu, GpuTensor,
};
pub use kernels::GEMV_SRC;
