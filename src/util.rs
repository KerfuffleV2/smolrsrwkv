use anyhow::{anyhow, Result};
use mmap_rs::{Mmap, MmapFlags, MmapOptions};
use ndarray::prelude::*;
use safetensors::{tensor::TensorView, SafeTensors};

use crate::model::{Ty, RWKV};

pub fn sigmoid(x: &Array1<Ty>) -> Array1<Ty> {
    x.map(|val| 1.0 / (1.0 + (-val).exp()))
}

fn bf16_tensor_to_f32(tensor: &TensorView<'_>) -> Vec<f32> {
    assert_eq!(tensor.dtype(), safetensors::Dtype::BF16);
    tensor
        .data()
        .chunks(2)
        .map(|i| half::bf16::from_le_bytes([i[0], i[1]]).to_f32())
        .collect::<Vec<f32>>()
}

pub fn bf16_tensor_to_array1(tensor: &TensorView<'_>) -> Array1<f32> {
    Array1::from(bf16_tensor_to_f32(tensor))
}

pub fn bf16_tensor_to_array2(tensor: &TensorView<'_>) -> Result<Array2<f32>> {
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

pub fn load_rwkv(m: Mmap) -> Result<RWKV<f32>> {
    let st = SafeTensors::deserialize(m.as_slice())?;
    RWKV::from_safetensors(&st)
}

pub fn sample_probs(
    rng: &mut impl rand::Rng,
    probs: &ArrayView1<f32>,
    temp: f32,
    top_p: f32,
) -> usize {
    use rand::distributions::Distribution;
    let mut sorted_probs = probs.as_slice().unwrap().to_vec();
    sorted_probs.sort_by(|a, b| f32::total_cmp(a, b).reverse());
    let mut cumulative_probs = Vec::with_capacity(sorted_probs.len());
    let _ = sorted_probs.iter().fold(0.0, |acc, i| {
        let newcum = acc + *i;
        cumulative_probs.push(newcum);
        newcum
    });
    let cutoffidx = cumulative_probs
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| *v > top_p)
        .map(|i| i.0)
        .unwrap_or_default();
    let cutoff = sorted_probs[cutoffidx];
    let probs = probs.map(|i| if *i < cutoff { 0.0 } else { i.powf(1.0 / temp) });
    let probs = &probs / probs.sum();
    let dist =
        rand::distributions::WeightedIndex::new(probs.iter()).expect("I didn't sign up for this!");
    dist.sample(rng)
}
