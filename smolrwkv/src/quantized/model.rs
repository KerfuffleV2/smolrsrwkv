#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, Array2, ArrayD};

/// Attention type (and for items that need a full float).
pub type ATy = f32;
/// Quantized type.
pub type WTy = u8;

#[derive(Debug, Clone, PartialEq)]
pub struct TensorQ2 {
    pub weight: Array2<WTy>,
    pub mx: ArrayD<ATy>,
    pub my: ArrayD<ATy>,
    pub rx: Array1<ATy>,
    pub ry: Array2<ATy>,
}
