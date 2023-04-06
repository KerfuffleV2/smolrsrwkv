use std::ops::{Add, Sub};

use anyhow::{anyhow, Result};
use mmap_rs::{MmapFlags, MmapOptions};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, NdFloat, ScalarOperand, Zip};
use num_traits::FromPrimitive;

use crate::loader::{TensorData, TensorType};

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

/// Converting bfloat16 format tensors to 1D or 2D arrays of float (only implemented for f32).
/// You could implement it for f64, but there's no practical reason to do so. Unfortunately,
/// you can't easily implement it for smaller types (16bit, 8bit, etc).
pub trait ConvertBF16Tensor: ReqOps {
    fn tensor_to_array1(tensor: &TensorData<'_>) -> Array1<Self>;
    fn tensor_to_array2(tensor: &TensorData<'_>) -> Result<Array2<Self>>;
}

impl ConvertBF16Tensor for f32 {
    fn tensor_to_array1(tensor: &TensorData<'_>) -> Array1<Self> {
        Array1::from(bf16_tensor_to_f32(tensor))
    }

    fn tensor_to_array2(tensor: &TensorData<'_>) -> Result<Array2<Self>> {
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

/// Helper function for opening a file and mmaping it.
pub fn mmap_file(s: &str) -> Result<mmap_rs::Mmap> {
    let fp = std::fs::File::open(s)?;
    let flen = fp.metadata()?.len();
    unsafe {
        MmapOptions::new(flen as usize)
            .and_then(|mo| {
                mo.with_file(fp, 0)
                    .with_flags(MmapFlags::NO_CORE_DUMP)
                    .map()
            })
            .map_err(|e| anyhow!(e))
    }
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

/// Magical stuff I don't understand too well. It takes the list of probabilities
/// and chooses a reasonable tokenid based on that.
pub fn sample_probs<T: ReqOps + num_traits::AsPrimitive<f32>>(
    rng: &mut impl rand::Rng,
    probs: &ArrayView1<T>,
    forever: bool, // Never select EndOfDocument token.
    temperature: f32,
    top_p: f32,
) -> usize {
    use rand::distributions::{Distribution, WeightedError, WeightedIndex};

    let mut sorted_probs = probs.as_slice().unwrap().to_vec();

    sorted_probs.sort_by(|a, b| T::partial_cmp(a, b).unwrap().reverse());
    let mut cumulative_probs = Vec::with_capacity(sorted_probs.len());
    let _ = sorted_probs.iter().fold(T::zero(), |acc, i| {
        let newcum = acc + *i;
        cumulative_probs.push(newcum);
        newcum
    });
    let cutoffidx = cumulative_probs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| v.as_() > top_p)
        .map(|i| i.0)
        .unwrap_or_default();
    let cutoff = sorted_probs[cutoffidx].as_();
    let mut probs = probs.map(|i| {
        let i: f32 = i.as_();
        if i < cutoff {
            0.0
        } else {
            i
        }
    });
    if forever {
        probs[0] = 0.0;
    }
    let probs = &probs / probs.sum();
    let dist = match WeightedIndex::new(probs.iter().map(|val| val.powf(1.0 / temperature))) {
        Ok(dist) => dist,
        Err(WeightedError::AllWeightsZero) => {
            // Sorry if you wanted tokens forever, but this is how it's got to be.
            return 0;
        }
        e => e.expect("I didn't sign up for this! (Bad weight in generated probability list.)"),
    };
    dist.sample(rng)
}

pub fn sigmoid<T: ReqOps>(x: Array1<T>) -> Array1<T> {
    let o = T::one();
    x.map(|val| o / (o + (-(*val)).exp()))
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
                (row.map(|el| *el as f32 + 0.5) * ry[0] * rx + my + mx).dot(rhs)
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
                        let el = (*el as f32) + 0.5;
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
pub use dumdot::pardot;
pub use dumdot::pardot8;
