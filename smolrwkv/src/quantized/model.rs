#![allow(clippy::upper_case_acronyms)]
use ndarray::{Array1, Array2, ArrayD, AsArray, Axis, Ix2, IxDyn};
use tracing::instrument;

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

fn amin<'a, A: AsArray<'a, ATy, Ix2>>(arr: A, axis: Axis) -> Array1<ATy> {
    arr.into()
        .axis_iter(axis)
        .map(|a| a.iter().copied().fold(ATy::INFINITY, |a, b| a.min(b)))
        .collect::<Array1<ATy>>()
}

fn amax<'a, A: AsArray<'a, ATy, Ix2>>(arr: A, axis: Axis) -> Array1<ATy> {
    arr.into()
        .axis_iter(axis)
        .map(|a| a.iter().copied().fold(ATy::NEG_INFINITY, |a, b| a.max(b)))
        .collect::<Array1<ATy>>()
}

impl From<Array2<ATy>> for TensorQ2 {
    #[instrument(skip_all, name = "convert_tensorq2", level = "DEBUG")]
    fn from(mut value: Array2<ATy>) -> Self {
        let shape = value.shape();
        let (mx, my) = if shape[0] > shape[1] {
            let miny = amin(&value, Axis(0)).insert_axis(Axis(1));
            value -= &miny;
            let miny = miny
                .into_dimensionality::<IxDyn>()
                .expect("miny failed dimensionality conversion!");
            let minx = amin(&value, Axis(1));
            value -= &minx;
            let minx = minx
                .into_dimensionality::<IxDyn>()
                .expect("minx failed dimensionality conversion!");
            (minx, miny)
        } else {
            let miny = amin(&value, Axis(1));
            value -= &miny;
            let miny = miny
                .into_dimensionality::<IxDyn>()
                .expect("miny failed dimensionality conversion!");
            let minx = amin(&value, Axis(0)).insert_axis(Axis(1));
            value -= &minx;
            let minx = minx
                .into_dimensionality::<IxDyn>()
                .expect("minx failed dimensionality conversion!");

            (minx, miny)
        };
        let rx = amax(&value, Axis(1));
        value /= &rx;
        let ry = amax(&value, Axis(0)).insert_axis(Axis(1));
        value /= &ry;
        let weight = value.mapv_into_any(|el| (el * 256.0).floor().clamp(0.0, 255.0) as u8);
        Self {
            weight,
            mx,
            my,
            rx: rx / 16.0,
            ry: ry / 16.0,
        }
    }
}
