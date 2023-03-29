use anyhow::{anyhow, Result};

use safetensors::{tensor::TensorView, SafeTensors};
use std::collections::HashMap;

use crate::model::*;
use crate::util::{bf16_tensor_to_array1, bf16_tensor_to_array2};

impl TryFrom<mmap_rs::Mmap> for RWKV<Ty> {
    type Error = anyhow::Error;

    fn try_from(value: mmap_rs::Mmap) -> std::result::Result<Self, Self::Error> {
        let st = SafeTensors::deserialize(value.as_slice())?;
        (&st).try_into()
    }
}

impl TryFrom<(usize, &HashMap<String, TensorView<'_>>)> for LayerNorm<Ty> {
    type Error = anyhow::Error;

    fn try_from((idx, lm): (usize, &HashMap<String, TensorView<'_>>)) -> Result<Self> {
        Ok(Self {
            bias: bf16_tensor_to_array1(
                lm.get(&format!("ln{idx}.bias"))
                    .ok_or_else(|| anyhow!("Bad format"))?,
            ),
            weight: bf16_tensor_to_array1(
                lm.get(&format!("ln{idx}.weight"))
                    .ok_or_else(|| anyhow!("Bad format"))?,
            ),
        })
    }
}

impl TryFrom<&SafeTensors<'_>> for RWKV<Ty> {
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
        fn gk<O>(
            m: &HashMap<String, TensorView>,
            k: &str,
            f: impl Fn(&TensorView) -> O,
        ) -> Result<O> {
            m.get(k).map(f).ok_or_else(|| anyhow!("Bad format"))
        }
        let layers = (0..n_layers)
            .map(|lnum| {
                let lm = tm.get(&Some(lnum)).expect("Impossible layer missing");
                Result::<_, anyhow::Error>::Ok(Layer {
                    ln: [LayerNorm::try_from((1, lm))?, LayerNorm::try_from((2, lm))?],
                    att: Attention {
                        key_weight: gk(lm, "att.key.weight", bf16_tensor_to_array2)??,
                        value_weight: gk(lm, "att.value.weight", bf16_tensor_to_array2)??,
                        output_weight: gk(lm, "att.output.weight", bf16_tensor_to_array2)??,
                        receptance_weight: gk(lm, "att.receptance.weight", bf16_tensor_to_array2)??,
                        time: AttTime {
                            first: gk(lm, "att.time_first", bf16_tensor_to_array1)?,
                            decay: gk(lm, "att.time_decay", bf16_tensor_to_array1)?,
                            mix_k: Mix(gk(lm, "att.time_mix_k", bf16_tensor_to_array1)?),
                            mix_v: Mix(gk(lm, "att.time_mix_v", bf16_tensor_to_array1)?),
                            mix_r: Mix(gk(lm, "att.time_mix_r", bf16_tensor_to_array1)?),
                        },
                    },
                    ffn: FeedForwardNetwork {
                        key_weight: gk(lm, "ffn.key.weight", bf16_tensor_to_array2)??,
                        value_weight: gk(lm, "ffn.value.weight", bf16_tensor_to_array2)??,
                        receptance_weight: gk(lm, "ffn.receptance.weight", bf16_tensor_to_array2)??,
                        time: FFNTime {
                            mix_k: Mix(gk(lm, "ffn.time_mix_k", bf16_tensor_to_array1)?),
                            mix_r: Mix(gk(lm, "ffn.time_mix_r", bf16_tensor_to_array1)?),
                        },
                    },
                })
                //
            })
            .collect::<Result<Vec<Layer<Ty>>, _>>()?;
        let l0m = tm.get(&Some(0)).unwrap();
        let nlm = tm
            .get(&None)
            .ok_or_else(|| anyhow!("Missing non-layer tensors!"))?;
        Ok(RWKV {
            emb: gk(nlm, "emb.weight", bf16_tensor_to_array2)??,
            head: gk(nlm, "head.weight", bf16_tensor_to_array2)??,
            ln_out: LayerNorm {
                bias: gk(nlm, "ln_out.bias", bf16_tensor_to_array1)?,
                weight: gk(nlm, "ln_out.weight", bf16_tensor_to_array1)?,
            },
            ln0: LayerNorm::try_from((0, l0m))?,
            layers,
        })
    }
}
