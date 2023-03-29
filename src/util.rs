use std::ops::Sub;

use anyhow::{anyhow, Result};
use mmap_rs::{MmapFlags, MmapOptions};
use ndarray::{Array1, Array2, ArrayView1, NdFloat, ScalarOperand};
use num_traits::FromPrimitive;
use safetensors::tensor::TensorView;

pub trait ReqOps: Sized + Default + Clone
where
    Self: NdFloat + ScalarOperand + FromPrimitive,
    Self: for<'a> Sub<&'a Array1<Self>, Output = Array1<Self>>,
{
}

impl ReqOps for f32 {}
impl ReqOps for f64 {}

pub trait ConvertBF16Tensor: ReqOps {
    fn tensor_to_array1(tensor: &TensorView<'_>) -> Array1<Self>;
    fn tensor_to_array2(tensor: &TensorView<'_>) -> Result<Array2<Self>>;
}

impl ConvertBF16Tensor for f32 {
    fn tensor_to_array1(tensor: &TensorView<'_>) -> Array1<Self> {
        Array1::from(bf16_tensor_to_f32(tensor))
    }

    fn tensor_to_array2(tensor: &TensorView<'_>) -> Result<Array2<Self>> {
        // Squeeze all the things.
        let shp = tensor
            .shape()
            .iter()
            .copied()
            .filter(|i| i != &1)
            .collect::<Vec<usize>>();
        anyhow::ensure!(shp.len() == 2, "Bad shape");
        Ok(Array2::from_shape_vec(
            (shp[0], shp[1]),
            bf16_tensor_to_f32(tensor),
        )?)
    }
}
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

pub fn sigmoid<T: ReqOps>(x: &Array1<T>) -> Array1<T> {
    x.map(|val| T::one() / (T::one() + (-(*val)).exp()))
}

fn bf16_tensor_to_f32(tensor: &TensorView<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype(), safetensors::Dtype::BF16);
    tensor
        .data()
        .chunks(2)
        .map(|i| half::bf16::from_le_bytes([i[0], i[1]]).to_f32())
        .collect::<Vec<f32>>()
}

pub fn sample_probs<T>(
    rng: &mut impl rand::Rng,
    probs: &ArrayView1<T>,
    temperature: f32,
    top_p: f32,
) -> usize
where
    T: ReqOps + num_traits::AsPrimitive<f32> + rand::distributions::uniform::SampleUniform,
    for<'a> &'a T: rand::distributions::uniform::SampleBorrow<T>,
{
    use rand::distributions::Distribution;
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
    let probs = probs.map(|i| {
        let i: f32 = i.as_();
        if i < cutoff {
            0.0
        } else {
            i.powf(1.0 / temperature)
        }
    });
    let probs = &probs / probs.sum();
    let dist = rand::distributions::WeightedIndex::new(probs.iter())
        .expect("I didn't sign up for this! (Bad weight in generated probability list.)");
    dist.sample(rng)
}
