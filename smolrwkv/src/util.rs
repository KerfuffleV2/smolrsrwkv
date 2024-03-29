#![allow(clippy::deprecated_semver)]
use std::ops::{Add, Sub};

use anyhow::{anyhow, ensure, Result};
use half::slice::HalfFloatSliceExt;
use memmap2::{Mmap, MmapOptions};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, NdFloat, ScalarOperand, Zip};
use num_traits::FromPrimitive;
use tracing::instrument;

use crate::loader::{TensorData, TensorType};
use crate::quantized::model::{ATy, TensorQ2};

/// Basically all the math stuff ndarray supports and we need for evaluating
/// RWKV
pub trait ReqOps: Sized + Default + Clone
where
    Self: NdFloat + ScalarOperand + FromPrimitive,
    Self: for<'a> Sub<&'a Array1<Self>, Output = Array1<Self>>,
    Self: for<'a> Add<&'a Array1<Self>, Output = Array1<Self>>,
{
}

impl ReqOps for f32 {}
impl ReqOps for f64 {}

/// Converting bfloat16 format tensors. The exact type is pretty flexible.
/// For non-quantized models, it'll just be converting to an array. However
/// for quantized models the trait can be used to convert to a structure.
/// In the case of 8bit quantized models, this would be TensorQ2 for 2D
/// tensors.
pub trait ConvertBF16Tensor<T>: Sized {
    fn convert_tensor(tensor: &TensorData<'_>) -> Result<T>;
}

impl ConvertBF16Tensor<Array1<Self>> for f32 {
    fn convert_tensor(tensor: &TensorData<'_>) -> Result<Array1<Self>> {
        ensure!(tensor.dtype == TensorType::BFloat16, "Bad tensor type");
        Ok(Array1::from(bf16_tensor_to_f32(tensor)))
    }
}

impl ConvertBF16Tensor<Array2<Self>> for f32 {
    fn convert_tensor(tensor: &TensorData<'_>) -> Result<Array2<Self>> {
        ensure!(tensor.dtype == TensorType::BFloat16, "Bad tensor type");
        // Squeeze all the things.
        let shp = tensor
            .shape
            .iter()
            .copied()
            .filter(|i| i != &1)
            .collect::<Vec<usize>>();
        anyhow::ensure!(shp.len() == 2, "Bad shape");
        Array2::from_shape_vec((shp[0], shp[1]), bf16_tensor_to_f32(tensor))
            .map_err(|e| anyhow!("Failed to build tensor in tensor_to_array2: {e}"))
    }
}

impl ConvertBF16Tensor<TensorQ2> for f32 {
    fn convert_tensor(tensor: &TensorData<'_>) -> Result<TensorQ2> {
        Ok(
            <Self as ConvertBF16Tensor<Array2<Self>>>::convert_tensor(tensor)
                .map_err(|e| anyhow!("Failed to build tensor in tensor_to_array2: {e}"))?
                .into(),
        )
    }
}

/// Helper function for opening a file and mmaping it.
pub fn mmap_file(s: &str) -> Result<Mmap> {
    let fp = std::fs::File::open(s)?;
    let m = unsafe { MmapOptions::new().map(&fp)? };
    #[cfg(unix)]
    m.advise(memmap2::Advice::DontDump)?;
    Ok(m)
}

/// Uses a pool to run a function with a limited number of threads.
pub fn run_threadlimited<R, F>(max_threads: usize, f: F) -> R
where
    R: Send,
    F: FnOnce() -> R + Send,
{
    rayon::ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build()
        .expect("Building thread pool failed!")
        .install(f)
}

/// Helper function to convert a SafeTensors TensorData into a flat
/// vector of f32. The number of dimensions doesn't matter at this
/// point.
fn bf16_tensor_to_f32(tensor: &TensorData<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype, TensorType::BFloat16, "Expected BF16 tensor");
    assert_ne!(tensor.data.len() & 1, 1, "Odd size for BF16 tensor input");
    tensor
        .data
        .chunks(2)
        .map(|i| half::bf16::from_le_bytes([i[0], i[1]]).to_f32())
        .collect::<Vec<f32>>()
}

/// Helper function to convert a SafeTensors TensorData into a flat
/// vector of f32. The number of dimensions doesn't matter at this
/// point.
pub fn bf16_tensor_to_f32_buf(tensor: &TensorData<'_>, buf: &mut Vec<f32>) {
    use half::slice::HalfBitsSliceExt;

    assert_eq!(tensor.dtype, TensorType::BFloat16, "Expected BF16 tensor");
    assert_ne!(tensor.data.len() & 1, 1, "Odd size for BF16 tensor input");
    let elcount = tensor.data.len() / 2;

    buf.clear();
    buf.reserve(elcount);
    let src: &[half::bf16] = bytemuck::cast_slice::<_, u16>(tensor.data).reinterpret_cast();
    let dst = unsafe {
        // This is only ever written to.
        std::mem::transmute::<&mut [std::mem::MaybeUninit<f32>], &mut [f32]>(
            &mut buf.spare_capacity_mut()[0..elcount],
        )
    };
    src.convert_to_f32_slice(dst);
    unsafe { buf.set_len(elcount) };
}

#[instrument(level = "DEBUG", fields(x = ?x.shape()))]
pub fn sigmoid<T: ReqOps>(x: Array1<T>) -> Array1<T> {
    let o = T::one();
    x.map(|val| o / (o + (-(*val)).exp()))
}

#[instrument(level = "DEBUG", fields(x = ?x.shape()))]
pub fn softmax<T: ReqOps>(x: &ArrayView1<T>) -> Array1<T> {
    let x_exp = x.mapv(T::exp);
    &x_exp / x_exp.sum()
}

pub trait ParDot {
    type Output;
    fn pardot(&self, rhs: &Self::Output) -> Self::Output;
}

impl<T: ReqOps> ParDot for Array2<T> {
    type Output = Array1<T>;
    #[instrument(level = "DEBUG",skip(self), fields(lhs = ?self.shape(), rhs = ?rhs.shape()))]
    fn pardot(&self, rhs: &Self::Output) -> Self::Output {
        dumdot::pardotv_simple(&self.view(), &rhs.view())
    }
}

impl ParDot for TensorQ2 {
    type Output = Array1<ATy>;
    #[instrument(level = "DEBUG", skip(self), fields(lhs = ?self.weight.shape(), rhs = ?rhs.shape()))]
    fn pardot(&self, rhs: &Self::Output) -> Self::Output {
        dumdot::pardot8(self, rhs)
    }
}

#[allow(dead_code)]
mod dumdot {
    use super::{Array1, Array2, ArrayView1, ArrayView2, ReqOps, Zip};
    use crate::quantized::model::{ATy, TensorQ2};
    use ndarray::{parallel::prelude::*, Axis};

    /// The simple implementation of parallel matrix-vector multiplication using Rayon.
    /// Presumably this calculates every single row separately which could add some overhead.
    pub fn pardotv_simple<T: ReqOps>(lhs: &ArrayView2<T>, rhs: &ArrayView1<T>) -> Array1<T> {
        Zip::from(lhs.outer_iter()).par_map_collect(|row| row.dot(rhs))
    }

    pub fn pardot8(lhs: &TensorQ2, rhs: &Array1<ATy>) -> Array1<ATy> {
        let rx = &lhs.rx;
        let mx = lhs
            .mx
            .broadcast(lhs.weight.raw_dim())
            .expect("Impossible? Broadcast mx failed!");
        let my = lhs
            .my
            .broadcast(lhs.weight.raw_dim())
            .expect("Impossible? Broadcast mx failed!");

        Zip::from(lhs.weight.rows())
            .and(lhs.ry.rows())
            .and(my.rows())
            .and(mx.rows())
            .par_map_collect(|row, ry, mx, my| {
                (row.map(|el| *el as ATy + 0.5) * ry[0] * rx + my + mx).dot(rhs)
            })
    }

    // Sadly seems to be slower.
    pub fn pardot8_manualdot(lhs: &TensorQ2, rhs: &Array1<ATy>) -> Array1<ATy> {
        let rx = &lhs.rx;
        let mx = lhs
            .mx
            .broadcast(lhs.weight.raw_dim())
            .expect("Impossible? Broadcast mx failed!");
        let my = lhs
            .my
            .broadcast(lhs.weight.raw_dim())
            .expect("Impossible? Broadcast mx failed!");

        Zip::from(lhs.weight.rows())
            .and(lhs.ry.rows())
            .and(my.rows())
            .and(mx.rows())
            .par_map_collect(|row, ry, mx, my| {
                let ry = ry[0];
                Zip::from(row).and(rx).and(mx).and(my).and(rhs).fold(
                    0.0,
                    |acc, el, rx, mx, my, rhs| {
                        let el = (*el as ATy) + 0.5;
                        let dqv = ((el * (ry * rx)) + (mx + my)) * rhs;
                        acc + dqv
                    },
                )
            })
    }

    /// Chunked parallel matrix-vector multiplication. However, it requires copying results
    /// around. Intuitively you might think it's better but just eyeballing the speed of the results
    /// looks about the same as the other function.
    pub fn pardotv_chunked<T: ReqOps>(lhs: &ArrayView2<T>, rhs: &ArrayView1<T>) -> Array1<T> {
        lhs.axis_chunks_iter(Axis(0), 64)
            .into_par_iter()
            .flat_map_iter(|x| x.dot(rhs))
            .collect::<Vec<_>>()
            .into()
    }

    pub fn pardot<T: ReqOps>(lhs: &Array2<T>, rhs: &Array1<T>) -> Array1<T> {
        pardotv_chunked(&lhs.view(), &rhs.view())
    }
}
pub use dumdot::{pardot, pardot8};
