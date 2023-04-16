use std::{
    borrow::Cow,
    collections::{HashMap, VecDeque},
    io::{stdout, Write},
    sync::{mpsc, Arc, Mutex},
    thread,
};

use anyhow::{anyhow, bail, Error, Ok as AOk, Result};
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
pub struct Tents(Tensor);

impl From<Tensor> for Tents {
    fn from(value: Tensor) -> Self {
        Self(value)
    }
}

pub struct RWKVLoader<'a> {
    pub(crate) atype: RwkvGgmlType,
    pub(crate) wtype: RwkvGgmlType,
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

    pub fn new() -> Self {
        Self {
            atype: RwkvGgmlType::Float32,
            wtype: RwkvGgmlType::Q4_1,
            _marker: std::marker::PhantomData,
        }
    }
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

pub trait Loader: Send + Sync + Sized {
    type Context: Send;
    type ItemDefinition: Send;
    type LoadedItem: Send;

    fn load(
        &self,
        context: &mut Self::Context,
        load_threads: usize,
        mut input_items: impl Iterator<Item = Self::ItemDefinition> + Send,
    ) -> Result<()> {
        thread::scope(move |sc| {
            // There's probably a better way but the idea behind this song and dance
            // is to avoid sending the items or definitions across a channel which can
            // be really complicated since it may have a non-'static lifetime. For example
            // it could be references into the mmaped model data.
            let (wo_tx, wo_rx) = mpsc::channel::<()>();
            let (wi_txs, wi_rxs) = {
                let mut wi_txs = Vec::with_capacity(load_threads);
                let mut wi_rxs = Vec::with_capacity(load_threads);
                (0..load_threads).for_each(|_| {
                    let (tx, rx) = mpsc::sync_channel::<()>(1);
                    wi_txs.push(tx);
                    wi_rxs.push(rx)
                });
                (wi_txs, wi_rxs)
            };
            let work_in: Arc<Mutex<VecDeque<Self::ItemDefinition>>> =
                Arc::new(Mutex::new(VecDeque::new()));
            let work_out: Arc<Mutex<VecDeque<Result<Self::LoadedItem>>>> =
                Arc::new(Mutex::new(VecDeque::new()));

            println!("Create loader");
            let loader_thread = sc.spawn({
                let (work_in, work_out) = (work_in.clone(), work_out.clone());
                move || {
                    let (mut pending, mut have_items) = (0, true);
                    let mut last_worker_idx = 0;

                    // While the iterator has item or there are pending expensive items being computed:
                    'handle_items: while have_items || pending > 0 {
                        println!("Loop: have_items={have_items}, pending={pending}");
                        if pending >= load_threads || !have_items {
                            // The iterator is either empty and there are items pending or
                            // the work queue is already full.

                            wo_rx.recv()?;
                            let litem = work_out
                                .lock()
                                .map_err(|e| anyhow!("Work out mutex failure: {e:?}"))?
                                .pop_front()
                                .ok_or_else(|| anyhow!("Unexpected empty output work queue!"))??;
                            pending -= 1;
                            self.loaded_item(context, litem)?;
                            continue;
                        }
                        let itemdef = if let Some(i) = input_items.next() {
                            i
                        } else {
                            have_items = false;
                            continue;
                        };
                        if load_threads < 2 || !self.is_expensive(&itemdef) {
                            // Cheap item, just process inline.
                            println!("Not expensive");
                            let litem = self.load_item(itemdef)?;
                            self.loaded_item(context, litem)?;
                            continue;
                        }
                        // At this point we have an expensive item we need to add to the work
                        // queue.
                        work_in
                            .lock()
                            .map_err(|e| anyhow!("Work in mutex failure: {e:?}"))?
                            .push_back(itemdef);

                        for worker_idx in
                            (last_worker_idx + 1..load_threads).chain(0..=last_worker_idx)
                        {
                            println!("Trying worker {worker_idx} (last {last_worker_idx}");
                            match wi_txs[worker_idx].try_send(()) {
                                Ok(_) => {
                                    last_worker_idx = worker_idx;
                                    pending += 1;
                                    continue 'handle_items;
                                }
                                Err(e) => match e {
                                    mpsc::TrySendError::Full(_) => continue,
                                    mpsc::TrySendError::Disconnected(_) => {
                                        bail!("Worker {worker_idx} unexpected disconnected!")
                                    }
                                },
                            }
                        }
                        bail!("Unexpectedly there are no ready workers!");
                    }
                    anyhow::Ok(())
                }
            });
            println!("Create workers");
            let mut worker_threads = Vec::with_capacity(load_threads);
            // This actually can't be something like .for_each because the
            // closure counts as a "scope" apparently, which means the first
            // thread will be created and then the scope will wait for it to exit.
            for (tnum, wi_rx) in (0..load_threads).zip(wi_rxs.into_iter()) {
                worker_threads.push({
                    let (work_in, work_out) = (work_in.clone(), work_out.clone());
                    let wo_tx = wo_tx.clone();

                    sc.spawn({
                        move || {
                            let result = (|| {
                                // println!(">> Thread {tnum} start.");
                                while wi_rx.recv().is_ok() {
                                    // println!("Thread {tnum} got token");
                                    let itemdef = work_in
                                        .lock()
                                        .map_err(|e| {
                                            anyhow!("Worker {tnum} work in mutex failure: {e:?}")
                                        })?
                                        .pop_front()
                                        .ok_or_else(|| {
                                            anyhow!("Unexpected empty input work queue!")
                                        })?;
                                    println!("\tWorker {tnum} working.");
                                    let maybelitem = self.load_item(itemdef);
                                    work_out
                                        .lock()
                                        .map_err(|e| {
                                            anyhow!("Worker {tnum} work out mutex failure: {e:?}")
                                        })?
                                        .push_back(maybelitem);
                                    wo_tx.send(())?;
                                }
                                println!("<< Thread {tnum} done.");
                                anyhow::Ok(())
                            })();
                            println!("\tThread {tnum} result: {result:?}");
                            result
                        }
                    })
                });
            }

            for wt in worker_threads.into_iter() {
                match wt.join() {
                    Ok(r) => r?,
                    Err(e) => std::panic::resume_unwind(e),
                }
            }
            match loader_thread.join() {
                Ok(r) => r?,
                Err(e) => std::panic::resume_unwind(e),
            }
            anyhow::Ok(())
        })?;

        Ok(())
    }
    fn is_expensive(&self, itemdef: &Self::ItemDefinition) -> bool;
    fn load_item(&self, itemdef: Self::ItemDefinition) -> Result<Self::LoadedItem>;
    fn loaded_item(&self, context: &mut Self::Context, item: Self::LoadedItem) -> Result<()>;
}

impl<'a> Loader for RWKVLoader<'a> {
    type Context = Vec<RWKVLoadedTensor<'a>>;
    type ItemDefinition = (String, TensorData<'a>, Vec<f32>);
    type LoadedItem = RWKVLoadedTensor<'a>;

    fn is_expensive(&self, itemdef: &Self::ItemDefinition) -> bool {
        true
    }

    fn load_item(&self, (name, td, buf): Self::ItemDefinition) -> Result<Self::LoadedItem> {
        use crate::loader::TensorType;
        println!("load_item: {name}");

        assert_eq!(
            td.dtype,
            TensorType::BFloat16,
            "Bad input type, must be BF16!"
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
        let out_type = if Self::WTENSORS.contains(&name.as_str()) {
            self.wtype
        } else {
            self.atype
        };

        match (&td.dtype, out_type) {
            (TensorType::BFloat16, RwkvGgmlType::Q4_0 | RwkvGgmlType::Q4_1) => {
                println!("QUANT: {name}");
                Ok(RWKVLoadedTensor {
                    layer,
                    name,
                    typ: out_type,
                    data: RWKVLoadedTensorData::U8(Cow::Owned(quantize_simple(&td, out_type, buf))),
                    shape,
                })
            }
            (TensorType::BFloat16, RwkvGgmlType::Float32) => {
                let data = RWKVLoadedTensorData::Float32(Cow::Owned(buf));
                Ok(RWKVLoadedTensor {
                    layer,
                    name,
                    data,
                    typ: out_type,
                    shape,
                })
            }
        }
    }

    fn loaded_item(&self, context: &mut Self::Context, item: Self::LoadedItem) -> Result<()> {
        println!("Loaded {}", item.name);
        context.push(item);
        Ok(())
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
    let t = bctx.ctx.new_tensor_2d(wtype.into(), shp[1], shp[0]);
    unsafe { (t.data() as *mut u8).copy_from_nonoverlapping(bctx.qbuf.as_ptr(), out_size) }
    t
}

fn quantize_simple(td: &TensorData<'_>, wtype: RwkvGgmlType, buf: Vec<f32>) -> Vec<u8> {
    assert_eq!(
        td.dtype,
        crate::loader::TensorType::BFloat16,
        "Bad input type, must be BF16!"
    );
    let nels = td.data.len() / 2;
    let shp = &td.shape;
    let in_size = nels * 4;

    // let mut buf: Vec<f32> = Vec::with_capacity(nels);
    // bf16_tensor_to_f32_buf(td, &mut buf);

    // FIXME: Verify this is safe, but 32bit -> 4bit shouldn't take more than 8bits per element
    // plus maybe an extra block. Riiight?
    let mut qbuf = Vec::with_capacity((in_size / 4) + ggml::blck_size(wtype.into()));

    let mut hist = [0i64; 16];
    let out_size = unsafe {
        match wtype {
            RwkvGgmlType::Q4_0 => ggml_sys::ggml_quantize_q4_0(
                buf.as_ptr(),
                qbuf.as_mut_ptr() as *mut std::ffi::c_void,
                nels as i32,
                shp[1] as i32,
                hist.as_mut_ptr(),
            ),
            RwkvGgmlType::Q4_1 => ggml_sys::ggml_quantize_q4_1(
                buf.as_ptr(),
                qbuf.as_mut_ptr() as *mut std::ffi::c_void,
                nels as i32,
                shp[1] as i32,
                hist.as_mut_ptr(),
            ),
            _ => panic!("Bad weight type!"),
        }
    };
    unsafe { qbuf.set_len(out_size) };
    qbuf
}

/// Helper function for extracting a 1d tensor from the HashMap by string key.
/// Takes a closure to convert from the TensorData struct to a usable format.
fn gk1(bctx: &mut BuildCtx<'_, '_>, key: &str) -> Result<Tensor> {
    let td = bctx
        .lm
        .remove(key)
        .map_or_else(|| Err(anyhow!("Missing tensor: {key}")), Ok)?;
    let shp = td
        .shape
        .iter()
        .copied()
        .filter(|i| *i != 1)
        .collect::<Vec<_>>();
    let t = bctx.ctx.new_tensor_1d(GT32, shp[0]);
    bf16_tensor_to_f32_buf(&td, &mut bctx.buf);
    unsafe { (t.data() as *mut f32).copy_from_nonoverlapping(bctx.buf.as_ptr(), bctx.buf.len()) }
    Ok(t)
}

/// Helper function for extracting a 2d tensor from the HashMap by string key.
/// Takes a closure to convert from the TensorData struct to a usable format.
fn gk2(bctx: &mut BuildCtx<'_, '_>, key: &str) -> Result<Tensor> {
    let td = bctx
        .lm
        .remove(key)
        .map_or_else(|| Err(anyhow!("Missing tensor: {key}")), Ok)?;
    bf16_tensor_to_f32_buf(&td, &mut bctx.buf);
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
        .remove(key)
        .map_or_else(|| Err(anyhow!("Missing tensor: {key}")), Ok)?;
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
        let key = "att.time_decay";
        let td = bctx
            .lm
            .remove(key)
            .map_or_else(|| Err(anyhow!("Missing tensor: {key}")), Ok)?;
        let mut decay = <ATy as ConvertBF16Tensor<Array1<ATy>>>::convert_tensor(&td)?;
        decay.mapv_inplace(|el| -el.exp());
        let Tents(decay) = (bctx.ctx, decay).into();
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
