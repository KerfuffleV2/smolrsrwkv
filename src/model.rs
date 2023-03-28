#![allow(clippy::upper_case_acronyms)]
use std::collections::HashMap;

use anyhow::{anyhow, Result};
use ndarray::{prelude::*, LinalgScalar};
use safetensors::{tensor::TensorView, SafeTensors};

use crate::util::{bf16_tensor_to_array1, bf16_tensor_to_array2, sigmoid};

pub type Ty = f32;

#[derive(Debug, Clone, PartialEq)]
pub struct Mix<T>(pub Array1<T>);

#[derive(Debug, Clone, PartialEq)]
pub struct LayerNorm<T> {
    pub bias: Array1<T>,
    pub weight: Array1<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttTime<T> {
    pub decay: Array1<T>,
    pub mix_k: Mix<T>,
    pub mix_v: Mix<T>,
    pub mix_r: Mix<T>,
    pub first: Array1<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FFNTime<T> {
    pub mix_k: Mix<T>,
    pub mix_r: Mix<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Attention<T> {
    pub key_weight: Array2<T>,
    pub value_weight: Array2<T>,
    pub output_weight: Array2<T>,
    pub receptance_weight: Array2<T>,
    pub time: AttTime<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeedForwardNetwork<T> {
    pub key_weight: Array2<T>,
    pub value_weight: Array2<T>,
    pub receptance_weight: Array2<T>,
    pub time: FFNTime<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Layer<T> {
    pub ln: [LayerNorm<T>; 2],
    pub att: Attention<T>,
    pub ffn: FeedForwardNetwork<T>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RWKV<T> {
    pub emb: Array2<T>,
    pub head: Array2<T>,
    pub ln_out: LayerNorm<T>,
    pub ln0: LayerNorm<T>,
    pub layers: Vec<Layer<T>>,
}

#[derive(Clone, PartialEq)]
pub struct RWKVLayerState<T> {
    pub tm_state: Array1<T>,
    pub tm_num: Array1<T>,
    pub tm_den: Array1<T>,
    pub cm_state: Array1<T>,
}

impl<T: Clone + LinalgScalar> RWKVLayerState<T> {
    pub fn new(n_embed: usize) -> Self {
        let zs = Array1::zeros(n_embed);
        Self {
            tm_state: zs.clone(),
            tm_num: zs.clone(),
            tm_den: zs.clone(),
            cm_state: zs,
        }
    }

    pub fn update(
        &mut self,
        tm_state: Array1<T>,
        tm_num: Array1<T>,
        tm_den: Array1<T>,
        cm_state: Array1<T>,
    ) {
        *self = Self {
            tm_state,
            tm_num,
            tm_den,
            cm_state,
        }
    }
}

impl Mix<Ty> {
    pub fn mix(&self, x: &ArrayView1<Ty>, last_x: &ArrayView1<Ty>) -> Array1<Ty> {
        x * &self.0 + last_x * (1.0 - &self.0)
    }
}

impl Attention<Ty> {
    pub fn time_mixing(
        &self,
        x: &ArrayView1<Ty>,
        state: &RWKVLayerState<Ty>,
    ) -> (Array1<Ty>, (Array1<Ty>, Array1<Ty>)) {
        let last_x = &state.tm_state.view();
        let last_num = &state.tm_num.view();
        let last_den = &state.tm_den.view();

        let k = self.key_weight.dot(&self.time.mix_k.mix(x, last_x));
        let v = self.value_weight.dot(&self.time.mix_v.mix(x, last_x));
        let r = self.receptance_weight.dot(&self.time.mix_r.mix(x, last_x));

        let exp_k = k.mapv(|el| el.exp());
        let exp_decay = self.time.decay.mapv(|el| (-el.exp()).exp());

        let wkv = {
            let e = (&self.time.first + &k).mapv(|el| el.exp());
            (last_num + &e * &v) / (last_den + e)
        };
        let rwkv = sigmoid(&r) * wkv;

        let num = &exp_decay * last_num + &exp_k * &v;
        let den = &exp_decay * last_den + &exp_k;
        (self.output_weight.dot(&rwkv), (num, den))
    }
}

impl FeedForwardNetwork<Ty> {
    pub fn channel_mixing(&self, x: &ArrayView1<Ty>, state: &RWKVLayerState<Ty>) -> Array1<Ty> {
        let last_x = &state.cm_state.view();
        let k = self.key_weight.dot(&self.time.mix_k.mix(x, last_x));
        let r = self.receptance_weight.dot(&self.time.mix_r.mix(x, last_x));
        let vk = self.value_weight.dot(&k.mapv(|val| val.max(0.0).powi(2)));
        sigmoid(&r) * &vk
    }
}

impl LayerNorm<Ty> {
    pub fn norm(&self, x: &ArrayView1<Ty>) -> Array1<Ty> {
        let mean = x.mean().unwrap();
        let std = x.std(0.0);
        (x - mean) / std * &self.weight + &self.bias
    }
}

impl LayerNorm<f32> {
    fn from_layermap(lm: &HashMap<String, TensorView>, idx: usize) -> Result<Self> {
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

impl RWKV<f32> {
    pub fn evaluate(&self, token: usize, state: &mut [RWKVLayerState<f32>]) -> Array1<f32> {
        let x = self.emb.index_axis(Axis(0), token);
        let mut x = self.ln0.norm(&x);

        for (lnum, layer) in self.layers.iter().enumerate() {
            let layer_state = &mut state[lnum];
            let x_ln1 = layer.ln[0].norm(&x.view());
            let (dx, (tm_num, tm_den)) = layer.att.time_mixing(&x_ln1.view(), layer_state);
            x += &dx;

            let x_ln2 = layer.ln[1].norm(&x.view());
            let dx = layer.ffn.channel_mixing(&x_ln2.view(), layer_state);
            x += &dx;

            layer_state.update(x_ln1, tm_num, tm_den, x_ln2);
        }
        let x = self.ln_out.norm(&x.view());
        let x = self.head.dot(&x);
        let x_max = x.fold(Ty::MIN, |acc, el| acc.max(*el));

        let e_x = (x - x_max).mapv(|el| el.exp());
        &e_x / e_x.sum()
    }

    pub fn from_safetensors(tensors: &SafeTensors) -> Result<Self> {
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
                    ln: [
                        LayerNorm::from_layermap(lm, 1)?,
                        LayerNorm::from_layermap(lm, 2)?,
                    ],
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
            .collect::<Result<Vec<Layer<f32>>, _>>()?;
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
            ln0: LayerNorm::from_layermap(l0m, 0)?,
            layers,
        })
    }
}
