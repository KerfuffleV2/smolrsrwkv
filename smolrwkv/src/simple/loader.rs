use std::{
    collections::HashMap,
    io::{stdout, Write},
};

use anyhow::{anyhow, Error, Ok, Result};
use ndarray::{Array1, Array2, Axis};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tracing::{info, instrument};

use crate::{
    loader::{TensorData, TensorDataMap},
    model_traits::RunLayerNorm,
    simple::model::*,
    util::{ConvertBF16Tensor, ReqOps},
};

/// LayerMap helper type to avoid repetition.
type LM<'a> = HashMap<String, TensorData<'a>>;

/// Helper function for extracting a tensor from the HashMap by string key.
/// Takes a closure to convert from the TensorData struct to a usable format.
fn gk<O>(m: &LM, k: &str, f: impl Fn(&TensorData<'_>) -> O) -> Result<O> {
    m.get(k).map(f).ok_or_else(|| anyhow!("Bad format"))
}

impl<T: ConvertBF16Tensor<Array1<T>>> TryFrom<(usize, &LM<'_>)> for LayerNorm<T> {
    type Error = Error;

    #[instrument(skip_all, name = "convert_layer_norm", level = "DEBUG")]
    fn try_from((idx, lm): (usize, &HashMap<String, TensorData<'_>>)) -> Result<Self> {
        Ok(Self {
            bias: T::convert_tensor(
                lm.get(&format!("ln{idx}.bias"))
                    .ok_or_else(|| anyhow!("Bad format"))?,
            )?,
            weight: T::convert_tensor(
                lm.get(&format!("ln{idx}.weight"))
                    .ok_or_else(|| anyhow!("Bad format"))?,
            )?,
        })
    }
}

impl<T: ConvertBF16Tensor<Array1<T>> + ReqOps> TryFrom<&LM<'_>> for AttTime<T> {
    type Error = Error;

    #[instrument(skip_all, name = "convert_attn_time_mix", level = "DEBUG")]
    fn try_from(lm: &LM<'_>) -> Result<Self> {
        let mut decay = gk(lm, "att.time_decay", T::convert_tensor)??;
        // Time decay can be precomputed to simplify inference.
        decay.mapv_inplace(|el| -el.exp());

        Ok(AttTime {
            first: gk(lm, "att.time_first", T::convert_tensor)??,
            decay,
            mix_k: Mix(gk(lm, "att.time_mix_k", T::convert_tensor)??),
            mix_v: Mix(gk(lm, "att.time_mix_v", T::convert_tensor)??),
            mix_r: Mix(gk(lm, "att.time_mix_r", T::convert_tensor)??),
        })
    }
}

impl<WT, T: ConvertBF16Tensor<Array1<T>> + ConvertBF16Tensor<WT> + ReqOps> TryFrom<&LM<'_>>
    for Attention<T, WT>
{
    type Error = Error;

    #[instrument(skip_all, name = "convert_attention", level = "DEBUG")]
    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(Attention {
            key_weight: gk(lm, "att.key.weight", T::convert_tensor)??,
            value_weight: gk(lm, "att.value.weight", T::convert_tensor)??,
            output_weight: gk(lm, "att.output.weight", T::convert_tensor)??,
            receptance_weight: gk(lm, "att.receptance.weight", T::convert_tensor)??,
            time: AttTime::try_from(lm)?,
        })
    }
}

impl<T: ConvertBF16Tensor<Array1<T>>> TryFrom<&LM<'_>> for FFNTime<T> {
    type Error = Error;

    #[instrument(skip_all, name = "convert_ffn_time_mix", level = "DEBUG")]
    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(FFNTime {
            mix_k: Mix(gk(lm, "ffn.time_mix_k", T::convert_tensor)??),
            mix_r: Mix(gk(lm, "ffn.time_mix_r", T::convert_tensor)??),
        })
    }
}

impl<WT, T: ConvertBF16Tensor<Array1<T>> + ConvertBF16Tensor<WT> + ReqOps> TryFrom<&LM<'_>>
    for FeedForwardNetwork<T, WT>
{
    type Error = Error;

    #[instrument(skip_all, name = "convert_ffn", level = "DEBUG")]
    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(FeedForwardNetwork {
            key_weight: gk(lm, "ffn.key.weight", T::convert_tensor)??,
            value_weight: gk(lm, "ffn.value.weight", T::convert_tensor)??,
            receptance_weight: gk(lm, "ffn.receptance.weight", T::convert_tensor)??,
            time: FFNTime::try_from(lm)?,
        })
    }
}

impl<WT, T: ConvertBF16Tensor<Array1<T>> + ConvertBF16Tensor<WT> + ReqOps> TryFrom<&LM<'_>>
    for RWKVLayer<T, WT>
{
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

impl<
        WT: Send,
        T: ConvertBF16Tensor<Array1<T>>
            + ConvertBF16Tensor<Array2<T>>
            + ConvertBF16Tensor<WT>
            + ReqOps,
    > TryFrom<TensorDataMap<'_>> for RWKV<T, WT>
{
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

        info!("Discovering model structure.");
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
        let mut emb: Array2<_> = gk(&nlm, "emb.weight", T::convert_tensor)??;
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
            head_weight: gk(&nlm, "head.weight", T::convert_tensor)??,
            ln_out: LayerNorm {
                bias: gk(&nlm, "ln_out.bias", T::convert_tensor)??,
                weight: gk(&nlm, "ln_out.weight", T::convert_tensor)??,
            },
            layers,
            n_layers,
            n_embed,
            n_vocab,
        })
    }
}
