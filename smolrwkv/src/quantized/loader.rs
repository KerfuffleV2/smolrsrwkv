use std::{
    collections::HashMap,
    io::{stdout, Write},
};

use anyhow::{anyhow, Error, Ok, Result};
use ndarray::{Array1, Array2, AsArray, Axis, Ix2, IxDyn};
use rayon::{iter::ParallelIterator, prelude::IntoParallelIterator};
use tracing::{info, instrument};

use crate::{
    loader::{TensorData, TensorDataMap},
    model_traits::RunLayerNorm,
    quantized::model::{ATy, TensorQ2},
    simple::model::*,
    util::ConvertBF16Tensor,
};

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

/// LayerMap helper type to avoid repetition.
type LM<'a> = HashMap<String, TensorData<'a>>;

/// Helper function for extracting a tensor from the HashMap by string key.
/// Takes a closure to convert from the TensorData struct to a usable format.
fn gk<O>(m: &LM, k: &str, f: impl Fn(&TensorData<'_>) -> O) -> Result<O> {
    m.get(k).map(f).ok_or_else(|| anyhow!("Bad format"))
}

impl TryFrom<&LM<'_>> for Attention<ATy, TensorQ2> {
    type Error = Error;

    #[instrument(skip_all, name = "convert_attention", level = "DEBUG")]
    fn try_from(lm: &LM<'_>) -> Result<Self> {
        let key_weight = gk(lm, "att.key.weight", ATy::tensor_to_array2)??.into();
        let value_weight = gk(lm, "att.value.weight", ATy::tensor_to_array2)??.into();
        let output_weight = gk(lm, "att.output.weight", ATy::tensor_to_array2)??.into();
        let receptance_weight = gk(lm, "att.receptance.weight", ATy::tensor_to_array2)??.into();
        Ok(Self {
            time: AttTime::try_from(lm)?,
            key_weight,
            value_weight,
            output_weight,
            receptance_weight,
        })
    }
}

impl TryFrom<&LM<'_>> for FeedForwardNetwork<ATy, TensorQ2> {
    type Error = Error;

    #[instrument(skip_all, name = "convert_ffn", level = "DEBUG")]
    fn try_from(lm: &LM<'_>) -> Result<Self> {
        let key_weight = gk(lm, "ffn.key.weight", ATy::tensor_to_array2)??.into();
        let value_weight = gk(lm, "ffn.value.weight", ATy::tensor_to_array2)??.into();
        let receptance_weight = gk(lm, "ffn.receptance.weight", ATy::tensor_to_array2)??.into();
        Ok(Self {
            time: FFNTime::try_from(lm)?,
            key_weight,
            value_weight,
            receptance_weight,
        })
    }
}

impl TryFrom<&LM<'_>> for RWKVLayer<ATy, TensorQ2> {
    type Error = Error;

    #[instrument(skip_all, name = "convert_layer", level = "DEBUG")]
    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(Self {
            ln_tm: LayerNorm::try_from((1, lm))?,
            ln_cm: LayerNorm::try_from((2, lm))?,
            att: Attention::try_from(lm)?,
            ffn: FeedForwardNetwork::try_from(lm)?,
        })
    }
}

impl TryFrom<TensorDataMap<'_>> for RWKV<ATy, TensorQ2> {
    type Error = Error;

    #[instrument(skip_all, name = "load_model")]
    fn try_from(tensors: TensorDataMap<'_>) -> Result<Self> {
        // This builds a HashMap of HashMaps.
        // The top level is None for non-layer tensors like "emb.weight" and
        // Some(layer_index) for each layer. The second level is just String key to TensorData.
        //
        // Worth noting is the fact that the model file gets mmaped but the actual keys/values
        // could be in any order. This means if you're loading from a spinny disky it could require
        // seeking all around the file rather than just reading sequentially.

        info!("Discovering model structure for model type Q8.");
        let mut layers = Vec::with_capacity(32);
        let mut nlm = HashMap::default();
        tensors.into_iter().try_for_each(|(mut name, tensor)| {
            if let Some(rest) = name.strip_prefix("blocks.") {
                let result = rest.split_once('.').ok_or_else(|| anyhow!("Bad format"))?;
                let lnum: usize = result.0.parse()?;
                if lnum >= layers.len() {
                    layers.resize_with(lnum + 1, HashMap::default);
                }

                name = result.1.to_string();
                layers[lnum].insert(name, tensor);
                Ok(())
            } else {
                nlm.insert(name, tensor);
                Ok(())
            }
        })?;

        anyhow::ensure!(!layers.is_empty(), "Not even one measly layer?");
        anyhow::ensure!(
            layers.iter().all(|lm| !lm.is_empty()),
            "Unexpected empty layers!"
        );
        anyhow::ensure!(!nlm.is_empty(), "Missing non-layer tensors!");
        let n_layers = layers.len();

        let ln0 = { LayerNorm::try_from((0, &layers[0]))? };

        info!("Loading {n_layers} layer(s):");
        let layers = layers
            .into_par_iter()
            .map(|lm| {
                print!(".");
                stdout().flush().ok();
                RWKVLayer::try_from(&lm)
            })
            .collect::<Result<Vec<_>, _>>()?;

        println!();
        info!("Precomputing embedding.");

        // It's possible to just precompute the embeddings in advance.
        let mut emb = gk(&nlm, "emb.weight", ATy::tensor_to_array2)??;
        let n_embed = emb.shape()[1];
        let n_vocab = emb.shape()[0];

        (0..n_vocab).for_each(|idx| {
            let idxemb = emb
                .index_axis_mut(Axis(0), idx)
                .into_slice_memory_order()
                .expect("Impossible: into_slice_memory_order failed!");
            idxemb.copy_from_slice(&ln0.norm(&idxemb).into_raw_vec());
        });
        drop(ln0);

        info!("Loading non-layer tensors.");

        Ok(RWKV {
            emb,
            head_weight: gk(&nlm, "head.weight", ATy::tensor_to_array2)??.into(),
            ln_out: LayerNorm {
                bias: gk(&nlm, "ln_out.bias", ATy::tensor_to_array1)?,
                weight: gk(&nlm, "ln_out.weight", ATy::tensor_to_array1)?,
            },
            layers,
            n_layers,
            n_embed,
            n_vocab,
        })
    }
}
