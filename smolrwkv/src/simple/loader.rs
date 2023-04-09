use std::{
    collections::HashMap,
    io::{stdout, Write},
};

use anyhow::{anyhow, Error, Ok, Result};
use ndarray::{Array2, Axis};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    loader::{TensorData, TensorDataMap},
    model_traits::RunLayerNorm,
    simple::model::*,
    util::ConvertBF16Tensor,
};

/// LayerMap helper type to avoid repetition.
type LM<'a> = HashMap<String, TensorData<'a>>;

/// Helper function for extracting a tensor from the HashMap by string key.
/// Takes a closure to convert from the TensorData struct to a usable format.
fn gk<O>(m: &LM, k: &str, f: impl Fn(&TensorData<'_>) -> O) -> Result<O> {
    m.get(k).map(f).ok_or_else(|| anyhow!("Bad format"))
}

impl<T: ConvertBF16Tensor> TryFrom<(usize, &LM<'_>)> for LayerNorm<T> {
    type Error = Error;

    fn try_from((idx, lm): (usize, &HashMap<String, TensorData<'_>>)) -> Result<Self> {
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

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for AttTime<T> {
    type Error = Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        let mut decay = gk(lm, "att.time_decay", T::tensor_to_array1)?;
        // Time decay can be precomputed to simplify inference.
        decay.mapv_inplace(|el| -el.exp());

        Ok(AttTime {
            first: gk(lm, "att.time_first", T::tensor_to_array1)?,
            decay,
            mix_k: Mix(gk(lm, "att.time_mix_k", T::tensor_to_array1)?),
            mix_v: Mix(gk(lm, "att.time_mix_v", T::tensor_to_array1)?),
            mix_r: Mix(gk(lm, "att.time_mix_r", T::tensor_to_array1)?),
        })
    }
}

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for Attention<T, Array2<T>> {
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

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for FeedForwardNetwork<T, Array2<T>> {
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

impl<T: ConvertBF16Tensor> TryFrom<TensorDataMap<'_>> for RWKV<T, Array2<T>> {
    type Error = Error;

    fn try_from(tensors: TensorDataMap<'_>) -> Result<Self> {
        let mut n_layers = 0usize;
        // This builds a HashMap of HashMaps.
        // The top level is None for non-layer tensors like "emb.weight" and
        // Some(layer_index) for each layer. The second level is just String key to TensorData.
        //
        // Worth noting is the fact that the model file gets mmaped but the actual keys/values
        // could be in any order. This means if you're loading from a spinny disky it could require
        // seeking all around the file rather than just reading sequentially.

        println!("* Discovering model structure for model type float32.");
        let tm = tensors.into_iter().try_fold(
            HashMap::<Option<u32>, HashMap<String, TensorData<'_>>>::new(),
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
                Ok(tm)
            },
        )?;

        println!("* Precomputing embedding.");
        let nlm = tm
            .get(&None)
            .ok_or_else(|| anyhow!("Missing non-layer tensors!"))?;
        let l0m = tm.get(&Some(0)).expect("Missing first layer!");
        // It's possible to just precompute the embeddings in advance.
        let mut emb = gk(nlm, "emb.weight", T::tensor_to_array2)??;
        let n_embed = emb.shape()[1];
        let n_vocab = emb.shape()[0];
        let ln0 = LayerNorm::try_from((0, l0m))?;
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
                Ok(RWKVLayer {
                    ln_tm: LayerNorm::try_from((1, lm))?,
                    ln_cm: LayerNorm::try_from((2, lm))?,
                    att: Attention::try_from(lm)?,
                    ffn: FeedForwardNetwork::try_from(lm)?,
                })
            })
            .collect::<Result<Vec<RWKVLayer<T, Array2<T>>>, _>>()?;

        println!("\n* Loading non-layer tensors.");

        Ok(RWKV {
            emb,
            head_weight: gk(nlm, "head.weight", T::tensor_to_array2)??,
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
