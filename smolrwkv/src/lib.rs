/// Quantized model/evaluation (just u8 for now)
pub mod quantized;

/// Simple model/evaluation (f32)
pub mod simple;

// GGML model/evaluation.
#[cfg(feature = "ggml")]
pub mod ggml;

/// Traits representing the components involved in evaluating RWKV.
pub mod model_traits;

/// Utility functions.
pub mod util;

/// Functions related to loading models.
pub mod loader;
