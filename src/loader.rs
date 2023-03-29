use anyhow::{anyhow, Result};

use safetensors::{tensor::TensorView, SafeTensors};
use std::collections::HashMap;

use crate::{model::*, util::ConvertBF16Tensor};

type LM<'a> = HashMap<String, safetensors::tensor::TensorView<'a>>;

fn gk<O>(m: &LM, k: &str, f: impl Fn(&TensorView) -> O) -> Result<O> {
    m.get(k).map(f).ok_or_else(|| anyhow!("Bad format"))
}

impl<T: ConvertBF16Tensor> TryFrom<mmap_rs::Mmap> for RWKV<T> {
    type Error = anyhow::Error;

    fn try_from(value: mmap_rs::Mmap) -> std::result::Result<Self, Self::Error> {
        let st = SafeTensors::deserialize(value.as_slice())?;
        (&st).try_into()
    }
}

impl<T: ConvertBF16Tensor> TryFrom<(usize, &LM<'_>)> for LayerNorm<T> {
    type Error = anyhow::Error;

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

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for AttTime<T> {
    type Error = anyhow::Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(AttTime {
            first: gk(lm, "att.time_first", T::tensor_to_array1)?,
            decay: gk(lm, "att.time_decay", T::tensor_to_array1)?,
            mix_k: Mix(gk(lm, "att.time_mix_k", T::tensor_to_array1)?),
            mix_v: Mix(gk(lm, "att.time_mix_v", T::tensor_to_array1)?),
            mix_r: Mix(gk(lm, "att.time_mix_r", T::tensor_to_array1)?),
        })
    }
}

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for Attention<T> {
    type Error = anyhow::Error;

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
    type Error = anyhow::Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(FFNTime {
            mix_k: Mix(gk(lm, "ffn.time_mix_k", T::tensor_to_array1)?),
            mix_r: Mix(gk(lm, "ffn.time_mix_r", T::tensor_to_array1)?),
        })
    }
}

impl<T: ConvertBF16Tensor> TryFrom<&LM<'_>> for FeedForwardNetwork<T> {
    type Error = anyhow::Error;

    fn try_from(lm: &LM<'_>) -> Result<Self> {
        Ok(FeedForwardNetwork {
            key_weight: gk(lm, "ffn.key.weight", T::tensor_to_array2)??,
            value_weight: gk(lm, "ffn.value.weight", T::tensor_to_array2)??,
            receptance_weight: gk(lm, "ffn.receptance.weight", T::tensor_to_array2)??,
            time: FFNTime::try_from(lm)?,
        })
    }
}

impl<T: ConvertBF16Tensor> TryFrom<&SafeTensors<'_>> for RWKV<T> {
    type Error = anyhow::Error;

    fn try_from(tensors: &SafeTensors<'_>) -> Result<Self> {
        let mut n_layers = 0;
        let tm = tensors.tensors().into_iter().try_fold(
            HashMap::<Option<u32>, HashMap<String, TensorView>>::new(),
            |mut tm, (mut name, tensor)| {
                let (layer_num, ktv) = if let Some(rest) = name.strip_prefix("blocks.") {
                    let result = rest.split_once('.').ok_or_else(|| anyhow!("Bad format"))?;
                    let lnum = result.0.parse()?;
                    n_layers = n_layers.max(lnum + 1);
                    name = result.1.to_string();
                    (Some(lnum), tensor)
                } else {
                    (None, tensor)
                };

                tm.entry(layer_num)
                    .or_insert_with(Default::default)
                    .insert(name, ktv);
                Result::<_, anyhow::Error>::Ok(tm)
            },
        )?;
        anyhow::ensure!(n_layers > 0, "Not even one measly layer?");

        let layers = (0..n_layers)
            .map(|lnum| {
                let lm = tm.get(&Some(lnum)).expect("Impossible layer missing");
                Result::<_, anyhow::Error>::Ok(Layer {
                    ln: [LayerNorm::try_from((1, lm))?, LayerNorm::try_from((2, lm))?],
                    att: Attention::try_from(lm)?,
                    ffn: FeedForwardNetwork::try_from(lm)?,
                })
                //
            })
            .collect::<Result<Vec<Layer<T>>, _>>()?;
        let l0m = tm.get(&Some(0)).unwrap();
        let nlm = tm
            .get(&None)
            .ok_or_else(|| anyhow!("Missing non-layer tensors!"))?;
        Ok(RWKV {
            emb: gk(nlm, "emb.weight", T::tensor_to_array2)??,
            head: gk(nlm, "head.weight", T::tensor_to_array2)??,
            ln_out: LayerNorm {
                bias: gk(nlm, "ln_out.bias", T::tensor_to_array1)?,
                weight: gk(nlm, "ln_out.weight", T::tensor_to_array1)?,
            },
            ln0: LayerNorm::try_from((0, l0m))?,
            layers,
        })
    }
}
