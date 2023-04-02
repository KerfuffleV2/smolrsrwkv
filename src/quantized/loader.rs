use anyhow::{anyhow, Error, Result};
use mmap_rs::Mmap;
use rayon::{iter::ParallelIterator, prelude::IntoParallelIterator};
use safetensors::{tensor::TensorView, SafeTensors};

use ndarray::Axis;
use std::{
    collections::HashMap,
    io::{stdout, Write},
};

use crate::{
    model_traits::RunLayerNorm, quantized::model::*, simple::model as S, util::ConvertBF16Tensor,
};

/// LayerMap helper type to avoid repetition.
type LM<'a> = HashMap<String, safetensors::tensor::TensorView<'a>>;

/// Helper function for extracting a tensor from the HashMap by string key.
/// Takes a closure to convert from the SafeTensors TensorView struct to
/// a usable format.
fn gk<O>(m: &LM, k: &str, f: impl Fn(&TensorView) -> O) -> Result<O> {
    m.get(k).map(f).ok_or_else(|| anyhow!("Bad format"))
}

/// Convert from a mmap (just a chunk of bytes) to the RWKV<T> struct
/// Requires the ConvertBF16Tensor trait (from `crate::utils`) due to
/// tensors being stored in bfloat16 format which isn't suitable for
/// actual calculation.
impl TryFrom<Mmap> for RWKV {
    type Error = Error;

    fn try_from(value: Mmap) -> std::result::Result<Self, Self::Error> {
        // Note that this actually just reads the metadata and not
        // the tensor data itself.
        let st = SafeTensors::deserialize(value.as_slice())?;
        // Use the TryFrom instance to convert from SafeTensors to RWKV<T>.
        (&st).try_into()
    }
}

impl TryFrom<&LM<'_>> for Attention {
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(Attention {
            key_weight: gk(lm, "att.key.weight", ATy::tensor_to_array2)??.into(),
            value_weight: gk(lm, "att.value.weight", ATy::tensor_to_array2)??.into(),
            output_weight: gk(lm, "att.output.weight", ATy::tensor_to_array2)??.into(),
            receptance_weight: gk(lm, "att.receptance.weight", ATy::tensor_to_array2)??.into(),
            time: S::AttTime::try_from(lm)?,
        })
    }
}

impl TryFrom<&LM<'_>> for FeedForwardNetwork {
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(FeedForwardNetwork {
            key_weight: gk(lm, "ffn.key.weight", ATy::tensor_to_array2)??.into(),
            value_weight: gk(lm, "ffn.value.weight", ATy::tensor_to_array2)??.into(),
            receptance_weight: gk(lm, "ffn.receptance.weight", ATy::tensor_to_array2)??.into(),
            time: S::FFNTime::try_from(lm)?,
        })
    }
}

impl TryFrom<&SafeTensors<'_>> for RWKV {
    type Error = Error;

    fn try_from(tensors: &SafeTensors<'_>) -> Result<Self> {
        let mut n_layers = 0usize;
        // This builds a HashMap of HashMaps.
        // The top level is None for non-layer tensors like "emb.weight" and
        // Some(layer_index) for each layer. The second level is just String key to TensorView.
        //
        // Worth noting is the fact that the model file gets mmaped but the actual keys/values
        // could be in any order. This means if you're loading from a spinny disky it could require
        // seeking all around the file rather than just reading sequentially.

        println!("* Discovering model structure.");
        let tm = tensors.tensors().into_iter().try_fold(
            HashMap::<Option<u32>, HashMap<String, TensorView>>::new(),
            |mut tm, (mut name, tensor)| {
                let (layer_num, ktv) = if let Some(rest) = name.strip_prefix("blocks.") {
                    let result = rest.split_once('.').ok_or_else(|| anyhow!("Bad format"))?;
                    let lnum: usize = result.0.parse()?;
                    n_layers = n_layers.max(lnum + 1);
                    name = result.1.to_string();
                    (Some(lnum as u32), tensor)
                } else {
                    (None, tensor)
                };

                tm.entry(layer_num)
                    .or_insert_with(Default::default)
                    .insert(name, ktv);
                Result::<_, Error>::Ok(tm)
            },
        )?;

        println!("* Precomputing embedding.");
        let nlm = tm
            .get(&None)
            .ok_or_else(|| anyhow!("Missing non-layer tensors!"))?;
        let l0m = tm.get(&Some(0)).unwrap();
        // It's possible to just precompute the embeddings in advance.
        let ln0 = S::LayerNorm::try_from((0, l0m))?;
        let mut emb = gk(nlm, "emb.weight", ATy::tensor_to_array2)??;
        let n_embed = emb.shape()[1];
        let n_vocab = emb.shape()[0];
        (0..n_vocab).for_each(|idx| {
            let idxemb = emb
                .index_axis_mut(Axis(0), idx)
                .into_slice_memory_order()
                .expect("Impossible: into_slice_memory_order failed!");
            idxemb.copy_from_slice(&ln0.norm(&idxemb).into_raw_vec());
        });

        anyhow::ensure!(n_layers > 0, "Not even one measly layer?");

        print!("* Loading {n_layers} layer(s): ");
        stdout().flush().ok();
        let layers = (0..n_layers)
            .into_par_iter()
            .map(|lnum| {
                print!(".");
                stdout().flush().ok();
                let lm = tm
                    .get(&Some(lnum as u32))
                    .expect("Impossible layer missing");
                Result::<_, Error>::Ok(RWKVLayer {
                    ln_tm: S::LayerNorm::try_from((1, lm))?,
                    ln_cm: S::LayerNorm::try_from((2, lm))?,
                    att: Attention::try_from(lm)?,
                    ffn: FeedForwardNetwork::try_from(lm)?,
                })
            })
            .collect::<Result<Vec<RWKVLayer>, _>>()?;

        println!("\n* Loading non-layer tensors.");

        Ok(RWKV {
            emb,
            head: gk(nlm, "head.weight", ATy::tensor_to_array2)??.into(),
            ln_out: S::LayerNorm {
                bias: gk(nlm, "ln_out.bias", ATy::tensor_to_array1)?,
                weight: gk(nlm, "ln_out.weight", ATy::tensor_to_array1)?,
            },
            layers,
            n_layers,
            n_embed,
            n_vocab,
        })
    }
}
