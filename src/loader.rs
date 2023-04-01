use anyhow::{anyhow, Error, Result};
use mmap_rs::Mmap;
use safetensors::{tensor::TensorView, SafeTensors};

use ndarray::{Axis, Ix1};
use std::{
    collections::HashMap,
    io::{stdout, Write},
};

use crate::{
    model::*,
    model_traits::RunLayerNorm,
    util::{Conv2, ConvertBF16Tensor, FloatTensor},
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
impl<T: ConvertBF16Tensor> TryFrom<Mmap> for RWKV<T, T>
where
    FloatTensor<T, Ix1>: Conv2,
{
    type Error = Error;

    fn try_from(value: Mmap) -> std::result::Result<Self, Self::Error> {
        // Note that this actually just reads the metadata and not
        // the tensor data itself.
        let st = SafeTensors::deserialize(value.as_slice())?;
        // Use the TryFrom instance to convert from SafeTensors to RWKV<T>.
        (&st).try_into()
    }
}

impl<T: ConvertBF16Tensor> TryFrom<(usize, &LM<'_>)> for LayerNorm<T>
where
    FloatTensor<T, Ix1>: Conv2,
{
    type Error = Error;

    fn try_from((idx, lm): (usize, &HashMap<String, TensorView<'_>>)) -> Result<Self> {
        Ok(Self {
            bias: T::tensor_to_array1(
                lm.get(&format!("ln{idx}.bias"))
                    .ok_or_else(|| anyhow!("Bad format"))?,
            ),
            weight: T::tensor_to_array1(
                lm.get(&format!("ln{idx}.weight"))
                    .ok_or_else(|| anyhow!("Bad format"))?,
            ),
        })
    }
}

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for AttTime<T, T>
where
    FloatTensor<T, Ix1>: Conv2,
{
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(AttTime {
            first: gk(lm, "att.time_first", FloatTensor::tensor_to)??,
            // Time decay can be precomputed to simplify inference.
            decay: gk(lm, "att.time_decay", T::tensor_to_array1)?.mapv(|el| (-el.exp()).exp()),
            mix_k: Mix(gk(lm, "att.time_mix_k", T::tensor_to_array1)?),
            mix_v: Mix(gk(lm, "att.time_mix_v", T::tensor_to_array1)?),
            mix_r: Mix(gk(lm, "att.time_mix_r", T::tensor_to_array1)?),
        })
    }
}

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for Attention<T, T>
where
    FloatTensor<T, Ix1>: Conv2,
{
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(Attention {
            key_weight: gk(lm, "att.key.weight", T::tensor_to_array2)??,
            value_weight: gk(lm, "att.value.weight", T::tensor_to_array2)??,
            output_weight: gk(lm, "att.output.weight", T::tensor_to_array2)??,
            receptance_weight: gk(lm, "att.receptance.weight", T::tensor_to_array2)??,
            time: AttTime::try_from(lm)?,
        })
    }
}

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for FFNTime<T> {
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(FFNTime {
            mix_k: Mix(gk(lm, "ffn.time_mix_k", T::tensor_to_array1)?),
            mix_r: Mix(gk(lm, "ffn.time_mix_r", T::tensor_to_array1)?),
        })
    }
}

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for FeedForwardNetwork<T, T> {
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(FeedForwardNetwork {
            key_weight: gk(lm, "ffn.key.weight", T::tensor_to_array2)??,
            value_weight: gk(lm, "ffn.value.weight", T::tensor_to_array2)??,
            receptance_weight: gk(lm, "ffn.receptance.weight", T::tensor_to_array2)??,
            time: FFNTime::try_from(lm)?,
        })
    }
}

impl<T: ConvertBF16Tensor> TryFrom<&SafeTensors<'_>> for RWKV<T, T>
where
    FloatTensor<T, Ix1>: Conv2,
{
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
        let ln0 = LayerNorm::try_from((0, l0m))?;
        let mut emb = gk(nlm, "emb.weight", T::tensor_to_array2)??;
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

        let mut stdout = stdout().lock();
        print!("* Loading {n_layers} layer(s): ");
        stdout.flush().ok();
        let layers = (0..n_layers)
            .map(|lnum| {
                print!("{}", lnum + 1);
                stdout.flush().ok();
                // println!("-   Loading layer {}/{n_layers}", lnum + 1);
                let lm = tm
                    .get(&Some(lnum as u32))
                    .expect("Impossible layer missing");
                let result = Result::<_, Error>::Ok(RWKVLayer {
                    ln1: LayerNorm::try_from((1, lm))?,
                    ln2: LayerNorm::try_from((2, lm))?,
                    att: Attention::try_from(lm)?,
                    ffn: FeedForwardNetwork::try_from(lm)?,
                });
                if lnum < n_layers - 1 {
                    print!(", ");
                    stdout.flush().ok();
                }
                result
            })
            .collect::<Result<Vec<RWKVLayer<T, T>>, _>>()?;

        println!("\n* Loading non-layer tensors.");

        Ok(RWKV {
            emb,
            head: gk(nlm, "head.weight", T::tensor_to_array2)??,
            ln_out: LayerNorm {
                bias: gk(nlm, "ln_out.bias", T::tensor_to_array1)?,
                weight: gk(nlm, "ln_out.weight", T::tensor_to_array1)?,
            },
            layers,
            n_layers,
            n_embed,
            n_vocab,
        })
    }
}
