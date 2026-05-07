//! rdna-compute: Kernel compilation, caching, and dispatch for RDNA GPUs.

mod compiler;
mod dispatch;
pub mod gemv_graph_cache;
mod kernels;
pub mod pool;
pub mod profile;
pub mod profiler;

pub use compiler::KernelCompiler;
pub use dispatch::{DType, Gpu, GpuTensor};
pub use gemv_graph_cache::{
    DispatchOutcome as GemvGraphDispatchOutcome, GemvGraphCache, GemvGraphStats, GemvShape,
    ScalarTy as GemvGraphScalarTy,
};
pub use kernels::GEMV_SRC;
