use std::{
    borrow::Cow,
    collections::HashMap,
    io::{stdout, Write},
};

use anyhow::{anyhow, bail, ensure, Error, Result};
use ndarray::{Array1, Array2};
use num_traits::ToPrimitive;
use tracing::{info, instrument};

use ggml_sys_bleedingedge as ggml_sys;
use rusty_ggml::{
    context::{GgmlContext as Context, GgmlContextBuilder},
    dims::*,
    tensor::{GgmlElementType as GT, GgmlElementType, GgmlTensor as Tensor},
};

use super::model::*;
use crate::{
    loader::{GenericLoader, TensorDataMap},
    util::bf16_tensor_to_f32_buf,
};

type ATy = f32;
const GT32: GT = GT::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RwkvGgmlType {
    Float32,
    Q4_0,
    Q4_1,
    Q4_2,
    Q4_3,
}

#[allow(clippy::from_over_into)]
// Note: Only Into here because can't handle all GGML types.
impl Into<GT> for RwkvGgmlType {
    fn into(self) -> GT {
        match self {
            RwkvGgmlType::Float32 => GT::F32,
            RwkvGgmlType::Q4_0 => GT::Q4_0,
            RwkvGgmlType::Q4_1 => GT::Q4_1,
            RwkvGgmlType::Q4_2 => GT::Q4_2,
            RwkvGgmlType::Q4_3 => GT::Q4_3,
        }
    }
}

struct BuildCtx<'a, 'b> {
    lnum: Option<usize>,
    ctx: &'a Context,
    lm: RWKVLoadMap<'b>,
}

#[derive(Default)]
pub struct RWKVLoader<'a> {
    _marker: std::marker::PhantomData<&'a ()>,
}

#[allow(clippy::new_without_default)]
impl<'a> RWKVLoader<'a> {
    const WTENSORS: &'static [&'static str] = &[
        "att.key.weight",
        "att.value.weight",
        "att.output.weight",
        "att.receptance.weight",
        "ffn.key.weight",
        "ffn.value.weight",
        "ffn.receptance.weight",
    ];
    const DECAY: &'static str = "att.time_decay";
}

pub enum RWKVLoadedTensorData<'a> {
    Float32(Cow<'a, Vec<f32>>),
    U8(Cow<'a, Vec<u8>>),
}
pub struct RWKVLoadedTensor<'a> {
    layer: Option<u32>,
    name: String,
    shape: [usize; 2],
    typ: RwkvGgmlType,
    data: RWKVLoadedTensorData<'a>,
}

type RWKVLoadMap<'a> = HashMap<(Option<u32>, String), RWKVLoadedTensor<'a>>;

impl<'a> GenericLoader for RWKVLoader<'a> {
    type Context = RWKVLoadMap<'a>;
    type ItemDefinition = RWKVLoaderItemDefinition<Vec<f32>>;
    type LoadedItem = RWKVLoadedTensor<'a>;

    fn use_parallel(&self, itemdef: &Self::ItemDefinition) -> bool {
        itemdef.use_parallel
    }

    fn load_one_item(&self, mut itemdef: Self::ItemDefinition) -> Result<Self::LoadedItem> {
        if itemdef.name == Self::DECAY {
            itemdef.data.iter_mut().for_each(|el| {
                *el = -el.exp();
            })
        }
        match &itemdef.out_type {
            RwkvGgmlType::Q4_0 | RwkvGgmlType::Q4_1 | RwkvGgmlType::Q4_2 | RwkvGgmlType::Q4_3 => {
                info!(
                    "Quantizing {}{}{:?} ({:?})",
                    itemdef
                        .layer
                        .as_ref()
                        .map(|l| format!("{l}."))
                        .unwrap_or_default(),
                    itemdef.name,
                    itemdef.shape,
                    itemdef.out_type
                );
                Ok(RWKVLoadedTensor {
                    layer: itemdef.layer,
                    name: itemdef.name,
                    typ: itemdef.out_type,
                    data: RWKVLoadedTensorData::U8(Cow::Owned(quantize_simple(
                        &itemdef.shape,
                        itemdef.out_type,
                        itemdef.data,
                    )?)),
                    shape: itemdef.shape,
                })
            }
            RwkvGgmlType::Float32 => {
                let data = RWKVLoadedTensorData::Float32(Cow::Owned(itemdef.data));
                Ok(RWKVLoadedTensor {
                    layer: itemdef.layer,
                    name: itemdef.name,
                    typ: itemdef.out_type,
                    data,
                    shape: itemdef.shape,
                })
            }
        }
    }

    fn loaded_item(&self, context: &mut Self::Context, item: Self::LoadedItem) -> Result<()> {
        context.insert((item.layer, item.name.clone()), item);
        Ok(())
    }
}

fn quantize_simple(shape: &[usize], wtype: RwkvGgmlType, buf: Vec<f32>) -> Result<Vec<u8>> {
    let nels = buf.len();

    // FIXME: Verify this is safe, but 32bit -> 4bit shouldn't take more than 8bits per element
    // plus maybe an extra block. Riiight?
    let wt: GgmlElementType = wtype.into();
    let mut qbuf = Vec::with_capacity(
        nels + unsafe { ggml_sys::ggml_blck_size(wt.to_u32().unwrap()) as usize },
    );
    let mut hist = [0i64; 16];
    let out_size = unsafe {
        match wtype {
            RwkvGgmlType::Q4_0 => ggml_sys::ggml_quantize_q4_0(
                buf.as_ptr(),
                qbuf.as_mut_ptr() as *mut std::ffi::c_void,
                nels as i32,
                shape[1] as i32,
                hist.as_mut_ptr(),
            ),
            RwkvGgmlType::Q4_1 => ggml_sys::ggml_quantize_q4_1(
                buf.as_ptr(),
                qbuf.as_mut_ptr() as *mut std::ffi::c_void,
                nels as i32,
                shape[1] as i32,
                hist.as_mut_ptr(),
            ),
            RwkvGgmlType::Q4_2 => ggml_sys::ggml_quantize_q4_2(
                buf.as_ptr(),
                qbuf.as_mut_ptr() as *mut std::ffi::c_void,
                nels as i32,
                shape[1] as i32,
                hist.as_mut_ptr(),
            ),
            RwkvGgmlType::Q4_3 => ggml_sys::ggml_quantize_q4_3(
                buf.as_ptr(),
                qbuf.as_mut_ptr() as *mut std::ffi::c_void,
                nels as i32,
                shape[1] as i32,
                hist.as_mut_ptr(),
            ),
            _ => bail!("Bad weight type!"),
        }
    };
    unsafe { qbuf.set_len(out_size) };
    Ok(qbuf)
}

/// Helper function for extracting a tensor from the HashMap by string key.
fn gk<const DIMS: usize>(bctx: &mut BuildCtx<'_, '_>, key: &str) -> Result<Tensor<DIMS>>
where
    Dim<DIMS>: DimValid,
    DimPair<DIMS, 4>: DimLt,
{
    let ltensor = bctx
        .lm
        .remove(&(bctx.lnum.map(|i| i as u32), key.to_string()))
        .map_or_else(
            || Err(anyhow!("Missing tensor: {key} (layer: {:?})", bctx.lnum)),
            Ok,
        )?;
    let shp = ltensor
        .shape
        .iter()
        .copied()
        .filter(|i| *i != 1)
        .collect::<Vec<_>>();
    ensure!(shp.len() == DIMS, "Unexpected shape for tensor {key}");
    ensure!((1..=2).contains(&DIMS), "Unsupport dimensions for {key}");
    let gtyp = ltensor.typ.into();
    let mut shape = [0; DIMS];
    shape.iter_mut().zip(shp.iter()).for_each(|(d, s)| *d = *s);
    let mut t = bctx.ctx.tensor(gtyp, shape);

    Ok(match (ltensor.typ, ltensor.data) {
        (RwkvGgmlType::Float32, RWKVLoadedTensorData::Float32(buf)) => {
            unsafe {
                t.with_data_mut(|d| {
                    d.as_mut()
                        .copy_from_slice(bytemuck::cast_slice(buf.as_slice()))
                });
            }
            t
        }
        (
            RwkvGgmlType::Q4_0 | RwkvGgmlType::Q4_1 | RwkvGgmlType::Q4_2 | RwkvGgmlType::Q4_3,
            RWKVLoadedTensorData::U8(buf),
        ) => {
            unsafe { t.with_data_mut(|d| d.copy_from_slice(buf.as_slice())) }
            t
        }
        _ => bail!(
            "Bad combination of loaded tensor type {:?} and data for {}",
            ltensor.typ,
            ltensor.name
        ),
    })
}

pub struct RWKVLoaderItemDefinition<T> {
    layer: Option<u32>,
    name: String,
    shape: [usize; 2],
    out_type: RwkvGgmlType,
    use_parallel: bool,
    data: T,
}

// FIXME: Should be in an impl probably.
pub fn load_rwkv(
    max_load_threads: usize,
    atype: RwkvGgmlType,
    wtype: RwkvGgmlType,
    tdm: TensorDataMap<'_>,
) -> Result<RWKVLoadMap> {
    use crate::loader::TensorType;

    let tensorcount = tdm.len();
    let mut plan = tdm
        .into_iter()
        .map(|(name, td)| {
            ensure!(
                td.dtype == TensorType::BFloat16,
                "Currently can only handle BF16 tensors!"
            );
            let shape = td
                .shape
                .iter()
                .copied()
                .filter(|i| *i != 1)
                .collect::<Vec<_>>();
            let shape = match shape.len() {
                1 => [shape[0], 1],
                2 => [shape[0], td.shape[1]],
                _ => bail!("Expected 1 or 2d tensor!"),
            };
            let (layer, name) = {
                if let Some(rest) = name.strip_prefix("blocks.") {
                    let result = rest
                        .split_once('.')
                        .ok_or_else(|| anyhow!("Bad tensor name format"))?;
                    let lnum: u32 = result.0.parse()?;
                    (Some(lnum), result.1.to_owned())
                } else {
                    (None, name.to_owned())
                }
            };
            let (use_parallel, out_type) = if RWKVLoader::WTENSORS.contains(&name.as_str()) {
                (true, wtype)
            } else {
                (name.as_str() == RWKVLoader::DECAY, atype)
            };
            Ok(RWKVLoaderItemDefinition {
                layer,
                name,
                shape,
                out_type,
                use_parallel,
                data: td,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    // The point here is to try to load the data from the mmapped model
    // sequentially to avoid giving spinning rust a hard time.
    // let mut tdi = tdm.into_iter().collect::<Vec<_>>();
    plan.sort_unstable_by_key(|pi| pi.data.data.as_ptr() as usize);
    let itemdefs = plan.into_iter().map(|pi| {
        let mut buf = Vec::new();
        bf16_tensor_to_f32_buf(&pi.data, &mut buf);
        RWKVLoaderItemDefinition {
            layer: pi.layer,
            name: pi.name,
            shape: pi.shape,
            out_type: pi.out_type,
            use_parallel: pi.use_parallel,
            data: buf,
        }
    });
    let mut loaded = HashMap::new();
    info!("RWKV: Starting load of {tensorcount} tensor(s).");
    RWKVLoader::<'static>::default().start_loading(&mut loaded, max_load_threads, itemdefs)?;
    Ok(loaded)
}

#[repr(transparent)]
pub struct Tents<const DIMS: usize>(Tensor<DIMS>);

impl From<Tensor<1>> for Tents<1> {
    fn from(value: Tensor<1>) -> Self {
        Self(value)
    }
}

impl From<(&Context, Array1<ATy>)> for Tents<1> {
    fn from((ctx, arr): (&Context, Array1<ATy>)) -> Self {
        let shp = arr.shape();
        let mut t = ctx.tensor(GT32, [shp[0]]);
        unsafe {
            t.with_data_mut(|d| {
                d.copy_from_slice(bytemuck::cast_slice(
                    arr.as_slice().expect("Impossible, can't get slice?"),
                ))
            });
        }
        Self(t)
    }
}

impl From<(&Context, Array2<ATy>)> for Tents<2> {
    fn from((ctx, arr): (&Context, Array2<ATy>)) -> Self {
        let shp = arr.shape();
        // ??? NOTE: The order for shapes is reversed in GGML.
        let mut t = ctx.tensor(GT32, [shp[0], shp[1]]);
        unsafe {
            t.with_data_mut(|d| {
                d.copy_from_slice(bytemuck::cast_slice(
                    arr.as_slice().expect("Impossible, can't get slice?"),
                ))
            });
        }
        Self(t)
    }
}

impl TryFrom<(usize, &mut BuildCtx<'_, '_>)> for LayerNorm {
    type Error = Error;

    #[instrument(skip_all, name = "convert_layer_norm", level = "DEBUG")]
    fn try_from((idx, bctx): (usize, &mut BuildCtx<'_, '_>)) -> Result<Self> {
        Ok(Self {
            weight: gk(bctx, &format!("ln{idx}.weight"))?,
            bias: gk(bctx, &format!("ln{idx}.bias"))?,
        })
    }
}

impl TryFrom<&mut BuildCtx<'_, '_>> for AttTime {
    type Error = Error;

    #[instrument(skip_all, err, name = "convert_attn_time_mix", level = "DEBUG")]
    fn try_from(bctx: &mut BuildCtx<'_, '_>) -> Result<Self> {
        Ok(Self {
            first: gk(bctx, "att.time_first")?,
            decay: gk(bctx, "att.time_decay")?,
            mix_k: Mix(gk(bctx, "att.time_mix_k")?),
            mix_v: Mix(gk(bctx, "att.time_mix_v")?),
            mix_r: Mix(gk(bctx, "att.time_mix_r")?),
        })
    }
}

impl TryFrom<&mut BuildCtx<'_, '_>> for Attention {
    type Error = Error;

    #[instrument(skip_all, name = "convert_att", level = "DEBUG")]
    fn try_from(bctx: &mut BuildCtx<'_, '_>) -> Result<Self> {
        Ok(Self {
            key_weight: gk(bctx, "att.key.weight")?,
            value_weight: gk(bctx, "att.value.weight")?,
            output_weight: gk(bctx, "att.output.weight")?,
            receptance_weight: gk(bctx, "att.receptance.weight")?,
            time: AttTime::try_from(bctx)?,
        })
    }
}

impl TryFrom<&mut BuildCtx<'_, '_>> for FFNTime {
    type Error = Error;

    #[instrument(skip_all, name = "convert_ffn_time_mix", level = "DEBUG")]
    fn try_from(bctx: &mut BuildCtx<'_, '_>) -> Result<Self> {
        Ok(Self {
            mix_k: Mix(gk(bctx, "ffn.time_mix_k")?),
            mix_r: Mix(gk(bctx, "ffn.time_mix_r")?),
        })
    }
}

impl TryFrom<&mut BuildCtx<'_, '_>> for FeedForwardNetwork {
    type Error = Error;

    #[instrument(skip_all, name = "convert_ffn", level = "DEBUG")]
    fn try_from(bctx: &mut BuildCtx<'_, '_>) -> Result<Self> {
        Ok(FeedForwardNetwork {
            key_weight: gk(bctx, "ffn.key.weight")?,
            value_weight: gk(bctx, "ffn.value.weight")?,
            receptance_weight: gk(bctx, "ffn.receptance.weight")?,
            time: FFNTime::try_from(bctx)?,
        })
    }
}

impl TryFrom<&mut BuildCtx<'_, '_>> for RWKVLayer {
    type Error = Error;

    #[instrument(skip_all, name = "convert_layer", level = "DEBUG")]
    fn try_from(bctx: &mut BuildCtx<'_, '_>) -> Result<Self> {
        Ok(Self {
            ln_tm: LayerNorm::try_from((1, &mut *bctx))?,
            ln_cm: LayerNorm::try_from((2, &mut *bctx))?,
            att: Attention::try_from(&mut *bctx)?,
            ffn: FeedForwardNetwork::try_from(bctx)?,
        })
    }
}

impl TryFrom<(RwkvGgmlType, RWKVLoadMap<'_>)> for RWKV {
    type Error = Error;

    #[instrument(skip_all, name = "load_model")]
    fn try_from((wtype, mut tensors): (RwkvGgmlType, RWKVLoadMap<'_>)) -> Result<Self> {
        info!("Discovering model structure.");
        let (n_layers, f32_size, quantized_size) =
            tensors
                .values()
                .fold((0, 0, 0), |(maxlayer, f32size, qsize), lt| {
                    let maxlayer = maxlayer.max(lt.layer.map(|i| i + 1).unwrap_or(0) as usize);
                    let (f32size, qsize) = match &lt.data {
                        RWKVLoadedTensorData::Float32(v) => (f32size + (v.len() * 4), qsize),
                        RWKVLoadedTensorData::U8(v) => (f32size, qsize + v.len()),
                    };
                    (maxlayer, f32size, qsize)
                });

        anyhow::ensure!(n_layers > 0, "Not even one measly layer?");
        info!("Model stats: Layer(s): {n_layers}, f32 tensor total size: {:.3}GiB, quantized tensor total size: {:.3}GiB",
        (f32_size as f64) / (1024.0 * 1024.0 * 1024.0), (quantized_size as f64) / (1024.0 * 1024.0 * 1024.0)
    );
        let (n_vocab, n_embed) = tensors
            .get(&(None, "emb.weight".to_string()))
            .ok_or_else(|| anyhow!("Missing emb.weight tensor"))
            .map(|x| {
                let shp = &x.shape;
                assert_eq!(shp.len(), 2, "Bad shape for emb.weight!");
                (shp[0], shp[1])
            })?;
        // FIXME; Better stuff here.
        let ctx_size = f32_size + quantized_size;
        let ctx_size = ctx_size + (ctx_size / 40);
        info!(
            "Guessed GGML context size: {:.3}GiB",
            ctx_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let ctx = GgmlContextBuilder::new().mem_size(ctx_size).build();

        // It's possible to just precompute the embeddings in advance.
        let emb = {
            let ln0weight = tensors
                .remove(&(Some(0), "ln0.weight".to_string()))
                .ok_or_else(|| anyhow!("ln0.weight tensor missing!"))?;
            let ln0bias = tensors
                .remove(&(Some(0), "ln0.bias".to_string()))
                .ok_or_else(|| anyhow!("ln0.bias tensor missing!"))?;
            let ln0 = match (ln0weight, ln0bias) {
                (
                    RWKVLoadedTensor {
                        typ: RwkvGgmlType::Float32,
                        data: RWKVLoadedTensorData::Float32(wdata),
                        ..
                    },
                    RWKVLoadedTensor {
                        typ: RwkvGgmlType::Float32,
                        data: RWKVLoadedTensorData::Float32(bdata),
                        ..
                    },
                ) => {
                    #[allow(clippy::unnecessary_to_owned)]
                    // Shush, Clippy. It actually is necessary here.
                    crate::simple::model::LayerNorm::<f32> {
                        weight: Array1::from_iter(wdata.into_owned().into_iter()),
                        bias: Array1::from_iter(bdata.into_owned().into_iter()),
                    }
                }
                _ => bail!("Unexpected format for ln0 tensors!"),
            };
            info!("Precomputing embedding... Embedding: {n_embed}, Vocab: {n_vocab}");
            let mut emba = match tensors
                .remove(&(None, "emb.weight".to_string()))
                .ok_or_else(|| anyhow!("Missing emb.weight tensor!"))?
            {
                RWKVLoadedTensor {
                    shape,
                    typ: RwkvGgmlType::Float32,
                    data: RWKVLoadedTensorData::Float32(wdata),
                    ..
                } if shape[1] != 1 => Array2::from_shape_vec(shape, wdata.into_owned())?,
                _ => bail!("Unexpected format for emb.weight tensor!"),
            };

            (0..n_vocab).for_each(|idx| {
                use crate::model_traits::RunLayerNorm;
                let idxemb = emba
                    .index_axis_mut(ndarray::Axis(0), idx)
                    .into_slice_memory_order()
                    .expect("Impossible: into_slice_memory_order failed!");
                idxemb.copy_from_slice(&ln0.norm(&idxemb).into_raw_vec());
            });
            let Tents(emb) = (&ctx, emba).into();
            emb
        };

        info!("Building {n_layers} layer(s):");
        let mut bctx = BuildCtx {
            lnum: None,
            ctx: &ctx,
            lm: tensors,
        };
        let layers = (0..n_layers)
            .map(|lnum| {
                bctx.lnum = Some(lnum);
                if wtype == RwkvGgmlType::Float32 {
                    print!(".");
                    stdout().flush().ok();
                }
                RWKVLayer::try_from(&mut bctx)
            })
            .collect::<Result<Vec<_>, _>>()?;

        if wtype == RwkvGgmlType::Float32 {
            println!();
        }
        bctx.lnum = None;
        info!("Building non-layer tensors.");
        let head_weight = gk(&mut bctx, "head.weight")?;
        let ln_out = LayerNorm {
            weight: gk(&mut bctx, "ln_out.weight")?,
            bias: gk(&mut bctx, "ln_out.bias")?,
        };
        info!(
            "GGML context size after load: {:.3}GiB",
            ctx.used_mem() as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let rwkv = RWKV {
            emb,
            head_weight,
            ln_out,
            layers,
            n_layers,
            n_embed,
            n_vocab,
            ctx,
        };

        Ok(rwkv)
    }
}
