use std::io::Write;

use anyhow::Result;
use ndarray::{Array2, ArrayView1};

use smolrwkv::{quantized::model::TensorQ2, simple::context::RWKVContext};

/// Used for non-quantized tensors and values.
pub type FloatType = f32;

pub enum Ctx {
    FloatCtx(RWKVContext<FloatType, Array2<FloatType>>),
    QuantCtx(RWKVContext<FloatType, TensorQ2>),
}

impl Ctx {
    pub fn params(&self) -> (usize, usize, usize) {
        match self {
            Ctx::FloatCtx(ctx) => (ctx.rwkv.n_layers, ctx.rwkv.n_embed, ctx.rwkv.n_vocab),
            Ctx::QuantCtx(ctx) => (ctx.rwkv.n_layers, ctx.rwkv.n_embed, ctx.rwkv.n_vocab),
        }
    }

    pub fn feed_prompt<S: AsRef<str>>(&mut self, s: S, f: Option<impl Fn(String)>) -> Result<()> {
        match self {
            Ctx::FloatCtx(ctx) => ctx.feed_prompt(s, f),
            Ctx::QuantCtx(ctx) => ctx.feed_prompt(s, f),
        }
    }

    pub fn infer_next_token(
        &mut self,
        samplefun: impl FnMut(&ArrayView1<FloatType>) -> Result<usize>,
    ) -> Result<Option<String>> {
        match self {
            Ctx::FloatCtx(ctx) => ctx.infer_next_token(samplefun),
            Ctx::QuantCtx(ctx) => ctx.infer_next_token(samplefun),
        }
    }
}

/// Helper to print out a string without a newline and then flush the console.
pub fn show_token(token: String) {
    print!("{token}");
    std::io::stdout().flush().ok();
}
