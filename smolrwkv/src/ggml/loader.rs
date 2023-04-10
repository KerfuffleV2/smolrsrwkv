#![allow(unused_imports, unused_variables, dead_code)]
use std::{
    collections::HashMap,
    io::{stdout, Write},
    ops::Deref,
};

use anyhow::{anyhow, bail, ensure, Error, Ok as AOk, Result};
use ndarray::{Array1, Array2};
use tracing::{info, instrument};

use ggml::{Context, Tensor};

use super::model::*;
use crate::{
    loader::{TensorData, TensorDataMap, TensorType},
    util::ConvertBF16Tensor,
};

type Ty = f32;
const GT32: ggml::Type = ggml::Type::F32;

/// LayerMap helper type to avoid repetition.
type BuildCtx<'a, 'b> = (&'a Context, &'a HashMap<String, TensorData<'b>>);

#[repr(transparent)]
struct Tents(Tensor);

impl From<Tensor> for Tents {
    fn from(value: Tensor) -> Self {
        Self(value)
    }
}

/// Helper function for extracting a tensor from the HashMap by string key.
/// Takes a closure to convert from the TensorData struct to a usable format.
fn gk<O>((_, m): &BuildCtx, k: &str, f: impl Fn(&TensorData<'_>) -> O) -> Result<O> {
    m.get(k).map(f).ok_or_else(|| anyhow!("Bad format"))
}

fn gk1(ctx: &Context, lm: &HashMap<String, TensorData<'_>>, key: &str) -> Result<Tensor> {
    let Tents(t) = (
        ctx,
        <Ty as ConvertBF16Tensor<Array1<Ty>>>::convert_tensor(
            lm.get(key).ok_or_else(|| anyhow!("Bad format"))?,
        )?,
    )
        .into();
    Ok(t)
}

fn gk2(ctx: &Context, lm: &HashMap<String, TensorData<'_>>, key: &str) -> Result<Tensor> {
    let Tents(t) = (
        ctx,
        <Ty as ConvertBF16Tensor<Array2<Ty>>>::convert_tensor(
            lm.get(key).ok_or_else(|| anyhow!("Bad format"))?,
        )?,
    )
        .into();
    Ok(t)
}

impl From<(&Context, Array1<Ty>)> for Tents {
    fn from((ctx, arr): (&Context, Array1<Ty>)) -> Self {
        let shp = arr.shape();
        let t = ctx.new_tensor_1d(GT32, shp[0]);
        unsafe { (t.data() as *mut f32).copy_from_nonoverlapping(arr.as_ptr(), arr.len()) }
        Self(t)
    }
}

impl From<(&Context, Array2<Ty>)> for Tents {
    fn from((ctx, arr): (&Context, Array2<Ty>)) -> Self {
        let shp = arr.shape();
        let t = ctx.new_tensor_2d(GT32, shp[0], shp[1]);
        unsafe { (t.data() as *mut f32).copy_from_nonoverlapping(arr.as_ptr(), arr.len()) }
        Self(t)
    }
}

impl TryFrom<(usize, BuildCtx<'_, '_>)> for LayerNorm {
    type Error = Error;

    #[instrument(skip_all, name = "convert_layer_norm", level = "DEBUG")]
    fn try_from((idx, (ctx, lm)): (usize, BuildCtx<'_, '_>)) -> Result<Self> {
        Ok(Self {
            bias: gk1(ctx, lm, &format!("ln{idx}.weight"))?,
            weight: gk1(ctx, lm, &format!("ln{idx}.bias"))?,
        })
    }
}

impl TryFrom<BuildCtx<'_, '_>> for AttTime {
    type Error = Error;

    #[instrument(skip_all, err, name = "convert_attn_time_mix", level = "DEBUG")]
    fn try_from(bctx @ (ctx, lm): BuildCtx<'_, '_>) -> Result<Self> {
        let Tents(decay) = (
            ctx,
            <Ty as ConvertBF16Tensor<Array1<Ty>>>::convert_tensor(
                lm.get("att.time_decay")
                    .ok_or_else(|| anyhow!("Bad format"))?,
            )?,
        )
            .into();
        Ok(Self {
            first: gk1(ctx, lm, "att.time_first")?,
            decay,
            mix_k: Mix(gk1(ctx, lm, "att.time_mix_k")?),
            mix_v: Mix(gk1(ctx, lm, "att.time_mix_v")?),
            mix_r: Mix(gk1(ctx, lm, "att.time_mix_r")?),
        })
    }
}

impl TryFrom<BuildCtx<'_, '_>> for Attention {
    type Error = Error;

    #[instrument(skip_all, name = "convert_att", level = "DEBUG")]
    fn try_from(bctx @ (ctx, lm): BuildCtx<'_, '_>) -> Result<Self> {
        Ok(Self {
            key_weight: gk2(ctx, lm, "att.key.weight")?,
            value_weight: gk2(ctx, lm, "att.value.weight")?,
            output_weight: gk2(ctx, lm, "att.output.weight")?,
            receptance_weight: gk2(ctx, lm, "att.receptance.weight")?,
            time: AttTime::try_from(bctx)?,
        })
    }
}

impl TryFrom<BuildCtx<'_, '_>> for FFNTime {
    type Error = Error;

    #[instrument(skip_all, name = "convert_ffn_time_mix", level = "DEBUG")]
    fn try_from(bctx @ (ctx, lm): BuildCtx<'_, '_>) -> Result<Self> {
        Ok(Self {
            mix_k: Mix(gk1(ctx, lm, "att.time_mix_k")?),
            mix_r: Mix(gk1(ctx, lm, "att.time_mix_r")?),
        })
    }
}

impl TryFrom<BuildCtx<'_, '_>> for FeedForwardNetwork {
    type Error = Error;

    #[instrument(skip_all, name = "convert_ffn", level = "DEBUG")]
    fn try_from(bctx @ (ctx, lm): BuildCtx<'_, '_>) -> Result<Self> {
        Ok(FeedForwardNetwork {
            key_weight: gk2(ctx, lm, "ffn.key.weight")?,
            value_weight: gk2(ctx, lm, "ffn.value.weight")?,
            receptance_weight: gk2(ctx, lm, "ffn.receptance.weight")?,
            time: FFNTime::try_from(bctx)?,
        })
    }
}

impl TryFrom<BuildCtx<'_, '_>> for RWKVLayer {
    type Error = Error;

    #[instrument(skip_all, name = "convert_layer", level = "DEBUG")]
    fn try_from(bctx: BuildCtx<'_, '_>) -> Result<Self> {
        Ok(Self {
            ln_tm: LayerNorm::try_from((1, bctx))?,
            ln_cm: LayerNorm::try_from((2, bctx))?,
            att: Attention::try_from(bctx)?,
            ffn: FeedForwardNetwork::try_from(bctx)?,
        })
    }
}

impl TryFrom<TensorDataMap<'_>> for RWKV {
    type Error = Error;

    #[instrument(skip_all, name = "load_model")]
    fn try_from(tensors: TensorDataMap<'_>) -> Result<Self> {
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
                AOk(())
            } else {
                nlm.insert(name, tensor);
                Ok(())
            }
        })?;
        let n_layers = layers.len();

        anyhow::ensure!(n_layers > 0, "Not even one measly layer?");
        anyhow::ensure!(
            layers.iter().all(|lm| !lm.is_empty()),
            "Unexpected empty layers!"
        );
        anyhow::ensure!(!nlm.is_empty(), "Missing non-layer tensors!");

        // FIXME; Real stuff here.
        let ctx_size = 10 * 1024 * 1024 * 1024;

        let ctx = ggml::Context::init(ctx_size);

        let ln0 = LayerNorm::try_from((0, (&ctx, &layers[0])))?;

        info!("Loading {n_layers} layer(s):");
        let layers = layers
            .into_iter()
            .map(|lm| {
                print!(".");
                stdout().flush().ok();
                RWKVLayer::try_from((&ctx, &lm))
            })
            .collect::<Result<Vec<_>, _>>()?;

        println!();
        info!("Precomputing embedding... psyche!");

        // It's possible to just precompute the embeddings in advance.
        let emb = gk2(&ctx, &nlm, "emb.weight")?;
        let embne = emb.get_ne();

        let n_embed = embne[1] as usize;
        let n_vocab = embne[0] as usize;
        info!("Loaded? {n_embed} -- {n_vocab}");

        // (0..n_vocab).for_each(|idx| {
        //     let idxemb = emb
        //         .index_axis_mut(Axis(0), idx)
        //         .into_slice_memory_order()
        //         .expect("Impossible: into_slice_memory_order failed!");
        //     idxemb.copy_from_slice(&ln0.norm(&idxemb).into_raw_vec());
        // });
        // drop(ln0);

        info!("Loading non-layer tensors.");

        Ok(RWKV {
            emb,
            head_weight: gk2(&ctx, &nlm, "head.weight")?,
            ln_out: LayerNorm {
                weight: gk1(&ctx, &nlm, "ln_out.weight")?,
                bias: gk1(&ctx, &nlm, "ln_out.bias")?,
            },
            layers,
            n_layers,
            n_embed,
            n_vocab,
            ctx,
        })
    }
}
