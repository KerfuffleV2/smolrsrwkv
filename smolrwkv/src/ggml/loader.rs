use std::{
    collections::HashMap,
    io::{stdout, Write},
};

use anyhow::{anyhow, Error, Ok as AOk, Result};
use ndarray::{Array1, Array2};
use tracing::{info, instrument};

use ggml::{Context, Tensor, Type as GT};

use super::model::*;
use crate::{
    loader::{TensorData, TensorDataMap},
    util::{bf16_tensor_to_f32_buf, ConvertBF16Tensor},
};

type ATy = f32;
const GT32: ggml::Type = GT::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RwkvGgmlType {
    Float32,
    Q4_0,
    Q4_1,
}

#[allow(clippy::from_over_into)]
// Note: Only Into here because can't handle all GGML types.
impl Into<ggml::Type> for RwkvGgmlType {
    fn into(self) -> ggml::Type {
        match self {
            RwkvGgmlType::Float32 => GT::F32,
            RwkvGgmlType::Q4_0 => GT::Q4_0,
            RwkvGgmlType::Q4_1 => GT::Q4_1,
        }
    }
}

struct BuildCtx<'a, 'b> {
    n_layers: usize,
    lnum: usize,
    ctx: &'a Context,
    lm: HashMap<String, TensorData<'b>>,
    wtype: RwkvGgmlType,
    buf: Vec<ATy>,
    qbuf: Vec<u8>,
}

#[repr(transparent)]
struct Tents(Tensor);

impl From<Tensor> for Tents {
    fn from(value: Tensor) -> Self {
        Self(value)
    }
}

fn quantize(bctx: &mut BuildCtx<'_, '_>, td: &TensorData<'_>) -> Tensor {
    let wtype = bctx.wtype;

    assert_eq!(
        td.dtype,
        crate::loader::TensorType::BFloat16,
        "Bad input type, must be BF16!"
    );
    let nels = td.data.len() / 2;
    let shp = &td.shape;
    let in_size = nels * 4;
    // FIXME: Verify this is safe, but 32bit -> 4bit shouldn't take more than 8bits per element
    // plus maybe an extra block. Riiight?
    bctx.qbuf.clear();
    bctx.qbuf
        .reserve((in_size / 4) + ggml::blck_size(wtype.into()));

    let mut hist = [0i64; 16];
    // output.fill(0);
    let out_size = unsafe {
        match wtype {
            RwkvGgmlType::Q4_0 => ggml_sys::ggml_quantize_q4_0(
                bctx.buf.as_ptr(),
                bctx.qbuf.as_mut_ptr() as *mut std::ffi::c_void,
                nels as i32,
                shp[1] as i32,
                hist.as_mut_ptr(),
            ),
            RwkvGgmlType::Q4_1 => ggml_sys::ggml_quantize_q4_1(
                bctx.buf.as_ptr(),
                bctx.qbuf.as_mut_ptr() as *mut std::ffi::c_void,
                nels as i32,
                shp[1] as i32,
                hist.as_mut_ptr(),
            ),
            _ => panic!("Bad weight type!"),
        }
    };
    unsafe { bctx.qbuf.set_len(out_size) };
    info!(
        "--> QUANT: len {in_size} -> {out_size} ({})",
        in_size - out_size
    );
    let t = bctx.ctx.new_tensor_2d(wtype.into(), shp[1], shp[0]);
    unsafe { (t.data() as *mut u8).copy_from_nonoverlapping(bctx.qbuf.as_ptr(), out_size) }
    t
}

/// Helper function for extracting a 1d tensor from the HashMap by string key.
/// Takes a closure to convert from the TensorData struct to a usable format.
fn gk1(bctx: &mut BuildCtx<'_, '_>, key: &str) -> Result<Tensor> {
    let td = bctx.lm.get(key).ok_or_else(|| anyhow!("Bad format"))?;

    let shp = td
        .shape
        .iter()
        .copied()
        .filter(|i| *i != 1)
        .collect::<Vec<_>>();
    let t = bctx.ctx.new_tensor_1d(GT32, shp[0]);
    bf16_tensor_to_f32_buf(td, &mut bctx.buf);
    unsafe { (t.data() as *mut f32).copy_from_nonoverlapping(bctx.buf.as_ptr(), bctx.buf.len()) }
    Ok(t)
}

/// Helper function for extracting a 2d tensor from the HashMap by string key.
/// Takes a closure to convert from the TensorData struct to a usable format.
fn gk2(bctx: &mut BuildCtx<'_, '_>, key: &str) -> Result<Tensor> {
    let td = bctx.lm.get(key).ok_or_else(|| anyhow!("Bad format"))?;
    bf16_tensor_to_f32_buf(td, &mut bctx.buf);
    let shp = &td.shape;
    let t = bctx.ctx.new_tensor_2d(GT32, shp[1], shp[0]);
    unsafe { (t.data() as *mut f32).copy_from_nonoverlapping(bctx.buf.as_ptr(), bctx.buf.len()) }
    Ok(t)
}

fn qgk2(bctx: &mut BuildCtx<'_, '_>, key: &str) -> Result<Tensor> {
    if bctx.wtype == RwkvGgmlType::Float32 {
        return gk2(bctx, key);
    }
    let td = bctx
        .lm
        .get(key)
        .ok_or_else(|| anyhow!("Bad format"))?
        .clone();
    info!(
        "[{}/{}]: Quantizing {key}({:?})",
        bctx.lnum + 1,
        bctx.n_layers,
        td.shape
    );
    bf16_tensor_to_f32_buf(&td, &mut bctx.buf);
    Ok(quantize(bctx, &td))
}

impl From<(&Context, Array1<ATy>)> for Tents {
    fn from((ctx, arr): (&Context, Array1<ATy>)) -> Self {
        let shp = arr.shape();
        let t = ctx.new_tensor_1d(GT32, shp[0]);
        unsafe { (t.data() as *mut f32).copy_from_nonoverlapping(arr.as_ptr(), arr.len()) }
        Self(t)
    }
}

impl From<(&Context, Array2<ATy>)> for Tents {
    fn from((ctx, arr): (&Context, Array2<ATy>)) -> Self {
        let shp = arr.shape();
        // NOTE: The order for shapes is reversed in GGML.
        let t = ctx.new_tensor_2d(GT32, shp[1], shp[0]);
        unsafe { (t.data() as *mut f32).copy_from_nonoverlapping(arr.as_ptr(), arr.len()) }
        Self(t)
    }
}

impl TryFrom<(usize, &mut BuildCtx<'_, '_>)> for LayerNorm {
    type Error = Error;

    #[instrument(skip_all, name = "convert_layer_norm", level = "DEBUG")]
    fn try_from((idx, bctx): (usize, &mut BuildCtx<'_, '_>)) -> Result<Self> {
        Ok(Self {
            weight: gk1(bctx, &format!("ln{idx}.weight"))?,
            bias: gk1(bctx, &format!("ln{idx}.bias"))?,
        })
    }
}

impl TryFrom<&mut BuildCtx<'_, '_>> for AttTime {
    type Error = Error;

    #[instrument(skip_all, err, name = "convert_attn_time_mix", level = "DEBUG")]
    fn try_from(bctx: &mut BuildCtx<'_, '_>) -> Result<Self> {
        let (ctx, lm) = (bctx.ctx, &bctx.lm);
        let mut decay = <ATy as ConvertBF16Tensor<Array1<ATy>>>::convert_tensor(
            lm.get("att.time_decay")
                .ok_or_else(|| anyhow!("Bad format"))?,
        )?;
        decay.mapv_inplace(|el| -el.exp());
        let Tents(decay) = (ctx, decay).into();
        Ok(Self {
            first: gk1(bctx, "att.time_first")?,
            decay,
            mix_k: Mix(gk1(bctx, "att.time_mix_k")?),
            mix_v: Mix(gk1(bctx, "att.time_mix_v")?),
            mix_r: Mix(gk1(bctx, "att.time_mix_r")?),
        })
    }
}

impl TryFrom<&mut BuildCtx<'_, '_>> for Attention {
    type Error = Error;

    #[instrument(skip_all, name = "convert_att", level = "DEBUG")]
    fn try_from(bctx: &mut BuildCtx<'_, '_>) -> Result<Self> {
        Ok(Self {
            key_weight: qgk2(bctx, "att.key.weight")?,
            value_weight: qgk2(bctx, "att.value.weight")?,
            output_weight: qgk2(bctx, "att.output.weight")?,
            receptance_weight: qgk2(bctx, "att.receptance.weight")?,
            time: AttTime::try_from(bctx)?,
        })
    }
}

impl TryFrom<&mut BuildCtx<'_, '_>> for FFNTime {
    type Error = Error;

    #[instrument(skip_all, name = "convert_ffn_time_mix", level = "DEBUG")]
    fn try_from(bctx: &mut BuildCtx<'_, '_>) -> Result<Self> {
        Ok(Self {
            mix_k: Mix(gk1(bctx, "ffn.time_mix_k")?),
            mix_r: Mix(gk1(bctx, "ffn.time_mix_r")?),
        })
    }
}

impl TryFrom<&mut BuildCtx<'_, '_>> for FeedForwardNetwork {
    type Error = Error;

    #[instrument(skip_all, name = "convert_ffn", level = "DEBUG")]
    fn try_from(bctx: &mut BuildCtx<'_, '_>) -> Result<Self> {
        Ok(FeedForwardNetwork {
            key_weight: qgk2(bctx, "ffn.key.weight")?,
            value_weight: qgk2(bctx, "ffn.value.weight")?,
            receptance_weight: qgk2(bctx, "ffn.receptance.weight")?,
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

impl TryFrom<(RwkvGgmlType, TensorDataMap<'_>)> for RWKV {
    type Error = Error;

    #[instrument(skip_all, name = "load_model")]
    fn try_from((wtype, tensors): (RwkvGgmlType, TensorDataMap<'_>)) -> Result<Self> {
        info!("Discovering model structure.");
        let mut layers = Vec::with_capacity(32);
        let mut nlm = HashMap::default();
        tensors.iter().try_for_each(|(name, tensor)| {
            let mut name = name.to_owned();
            if let Some(rest) = name.strip_prefix("blocks.") {
                let result = rest.split_once('.').ok_or_else(|| anyhow!("Bad format"))?;
                let lnum: usize = result.0.parse()?;
                if lnum >= layers.len() {
                    layers.resize_with(lnum + 1, HashMap::default);
                }

                name = result.1.to_string();
                layers[lnum].insert(name, tensor.clone());
                AOk(())
            } else {
                nlm.insert(name.to_owned(), tensor.clone());
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

        let (n_vocab, n_embed) = nlm
            .get("emb.weight")
            .ok_or_else(|| anyhow!("Bad format"))
            .map(|x| {
                let shp = &x.shape;
                assert_eq!(shp.len(), 2, "Bad shape for emb.weight!");
                (shp[0], shp[1])
                //
            })?;
        // FIXME; Better stuff here.
        let ctx_size = (n_layers + (4.max(n_layers / 5))) * (n_embed + (n_embed / 20)) * n_vocab;
        info!(
            "Guessed GGML context size: {:.3}GiB",
            ctx_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let ctx = ggml::Context::init(ctx_size);

        // It's possible to just precompute the embeddings in advance.
        let emb = {
            let ln0 = crate::simple::model::LayerNorm::<f32>::try_from((0, &layers[0]))?;
            info!("Precomputing embedding... Embedding: {n_embed}, Vocab: {n_vocab}");
            let mut emba = <ATy as ConvertBF16Tensor<Array2<ATy>>>::convert_tensor(
                nlm.get("emb.weight").ok_or_else(|| anyhow!("Bad format"))?,
            )?;

            (0..n_vocab).for_each(|idx| {
                use crate::model_traits::RunLayerNorm;
                let idxemb = emba
                    .index_axis_mut(ndarray::Axis(0), idx)
                    .into_slice_memory_order()
                    .expect("Impossible: into_slice_memory_order failed!");
                idxemb.copy_from_slice(&ln0.norm(&idxemb).into_raw_vec());
            });
            drop(ln0);

            let Tents(emb) = (&ctx, emba).into();
            emb
        };

        info!("Loading {n_layers} layer(s):");

        let mut bctx = BuildCtx {
            n_layers,
            lnum: 0,
            ctx: &ctx,
            lm: Default::default(),
            wtype,
            buf: Vec::with_capacity(256 * 1024 * 1024),
            qbuf: Vec::default(),
        };
        let layers = layers
            .into_iter()
            .enumerate()
            .map(|(lnum, lm)| {
                bctx.lnum = lnum;
                bctx.lm = lm;
                if wtype == RwkvGgmlType::Float32 {
                    print!(".");
                    stdout().flush().ok();
                }
                RWKVLayer::try_from(&mut bctx)
            })
            .collect::<Result<Vec<_>, _>>()?;

        println!();

        info!("Loading non-layer tensors.");
        bctx.lm = nlm;
        let head_weight = gk2(&mut bctx, "head.weight")?;
        let ln_out = LayerNorm {
            weight: gk1(&mut bctx, "ln_out.weight")?,
            bias: gk1(&mut bctx, "ln_out.bias")?,
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
