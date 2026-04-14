//! engine: GGUF model loading and LLaMA inference on RDNA GPUs.

pub mod gguf;
pub mod hfq;
pub mod llama;
#[cfg(feature = "deltanet")]
pub mod qwen35;
#[cfg(feature = "deltanet")]
pub mod qwen35_vl;
#[cfg(feature = "deltanet")]
pub mod speculative;
// Gemma 4: standard hybrid attention (sliding + full), not DeltaNet. The
// scale_f32 and rope_partial_halved dispatches it uses are ungated; the
// module itself is always available.
pub mod gemma4;
pub mod gemma4_vision;
pub mod image;
pub mod tokenizer;
