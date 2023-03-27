#![allow(clippy::upper_case_acronyms)]
use std::{collections::HashMap, io::Write};

use anyhow::{anyhow, Result};
use mmap_rs::{Mmap, MmapFlags, MmapOptions};
use ndarray::prelude::*;
use safetensors::{tensor::TensorView, SafeTensors};

const TESTSTR: &str = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";
const MODEL: &str = "./RWKV-4-Pile-430M-20220808-8066.safetensors";
const TOKENIZER: &str = "./20B_tokenizer.json";

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

pub struct RWKVState<T>(Array3<T>);

impl RWKVState<Ty> {
    pub fn new(n_layers: usize, n_embed: usize) -> Self {
        Self(Array3::zeros((n_layers, 4, n_embed)))
    }

    pub fn update(&mut self, layer_num: usize, a: &[Array1<Ty>; 4]) {
        self.0.slice_mut(s![layer_num, 0, ..]).assign(&a[0]);
        self.0.slice_mut(s![layer_num, 1, ..]).assign(&a[1]);
        self.0.slice_mut(s![layer_num, 2, ..]).assign(&a[2]);
        self.0.slice_mut(s![layer_num, 3, ..]).assign(&a[3]);
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
        lnum: usize,
        x: &ArrayView1<Ty>,
        state: &RWKVState<Ty>,
    ) -> (Array1<Ty>, (Array1<Ty>, Array1<Ty>)) {
        let last_x = state.0.slice(s![lnum, 0, ..]);
        let last_num = state.0.slice(s![lnum, 1, ..]);
        let last_den = state.0.slice(s![lnum, 2, ..]);

        let k = self.key_weight.dot(&self.time.mix_k.mix(x, &last_x));
        let v = self.value_weight.dot(&self.time.mix_v.mix(x, &last_x));
        let r = self.receptance_weight.dot(&self.time.mix_r.mix(x, &last_x));

        let exp_k = k.mapv(|el| el.exp());
        let exp_decay = self.time.decay.mapv(|el| (-el.exp()).exp());

        let wkv = {
            let e = (&self.time.first + &k).mapv(|el| el.exp());
            (&last_num + &e * &v) / (&last_den + e)
        };
        let rwkv = sigmoid(&r) * wkv;

        let num = &exp_decay * &last_num + &exp_k * &v;
        let den = &exp_decay * &last_den + &exp_k;
        (self.output_weight.dot(&rwkv), (num, den))
    }
}

impl FeedForwardNetwork<Ty> {
    pub fn channel_mixing(
        &self,
        lnum: usize,
        x: &ArrayView1<Ty>,
        state: &RWKVState<Ty>,
    ) -> Array1<Ty> {
        let last_x = state.0.slice(s![lnum, 3, ..]);
        let k = self.key_weight.dot(&self.time.mix_k.mix(x, &last_x));
        let r = self.receptance_weight.dot(&self.time.mix_r.mix(x, &last_x));
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
    pub fn evaluate(&self, token: usize, state: &mut RWKVState<f32>) -> Array1<f32> {
        let x = self.emb.index_axis(Axis(0), token);
        let mut x = self.ln0.norm(&x);

        for (lnum, layer) in self.layers.iter().enumerate() {
            let x_ln1 = layer.ln[0].norm(&x.view());
            let (dx, (tm_num, tm_den)) = layer.att.time_mixing(lnum, &x_ln1.view(), state);
            x += &dx;

            let x_ln2 = layer.ln[1].norm(&x.view());
            let dx = layer.ffn.channel_mixing(lnum, &x_ln2.view(), state);
            x += &dx;

            state.update(lnum, &[x_ln1, tm_num, tm_den, x_ln2]);
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

fn sigmoid(x: &Array1<Ty>) -> Array1<Ty> {
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

fn bf16_tensor_to_array1(tensor: &TensorView<'_>) -> Array1<f32> {
    Array1::from(bf16_tensor_to_f32(tensor))
}

fn bf16_tensor_to_array2(tensor: &TensorView<'_>) -> Result<Array2<f32>> {
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

fn mmap_file(s: &str) -> Result<mmap_rs::Mmap> {
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

fn load_rwkv(m: Mmap) -> Result<RWKV<f32>> {
    let st = SafeTensors::deserialize(m.as_slice())?;
    RWKV::from_safetensors(&st)
}

fn sample_probs(rng: &mut impl rand::Rng, probs: &ArrayView1<f32>, temp: f32, top_p: f32) -> usize {
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

fn main() -> Result<()> {
    let mut rng = rand::thread_rng();
    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER).map_err(|e| anyhow!(e))?;
    let rwkv = load_rwkv(mmap_file(MODEL)?)?;
    println!(
        "** Loaded: layers={}, embed={:?}",
        rwkv.layers.len(),
        rwkv.emb.shape()
    );
    let mut state = RWKVState::new(rwkv.layers.len(), rwkv.emb.shape()[1]);
    let toks = tokenizer.encode(TESTSTR, false).map_err(|e| anyhow!(e))?;
    let mut probs = Array1::<f32>::zeros(rwkv.emb.shape()[0]);

    toks.get_ids().iter().for_each(|tid| {
        probs = rwkv.evaluate(*tid as usize, &mut state);
        let tokstr = tokenizer.decode(vec![*tid], false).unwrap();
        print!("{}", tokstr);
        std::io::stdout().flush().ok();
    });
    loop {
        let tokid = sample_probs(&mut rng, &probs.view(), 1.0, 0.85);
        if tokid == 0 {
            println!(" [end of text]");
            break;
        }
        let tokstr = tokenizer.decode(vec![tokid as u32], false).unwrap();
        print!("{}", tokstr);
        std::io::stdout().flush().ok();
        probs = rwkv.evaluate(tokid, &mut state);
    }
    println!("Hokay.");
    Ok(())
}
