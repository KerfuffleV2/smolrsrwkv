#![allow(unused_imports, unused_variables, dead_code, unused_mut)]
use anyhow::{anyhow, Result};
use ndarray::ArrayView1;
use tokenizers::Tokenizer;

use ggml::{Tensor, Type as GT};

use super::{
    graph::*,
    model::{RWKVLayerState, RWKV},
};

/// Context that holds the state of the RWKV model.
pub struct RWKVContext {
    /// The RWKV model data — immutable.
    pub rwkv: RWKV,
    /// Model state.
    pub state: Vec<RWKVLayerState>,
    /// Probabilities from the last step (starts filled with zeros).
    // pub last_probs: Array1<T>,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
}

impl RWKVContext {
    pub fn new(rwkv: RWKV, tokenizer: Tokenizer) -> Self {
        let ctx = &rwkv.ctx;

        // FIXME: Handle token passing and initial probs.
        let token_idx = ctx.new_tensor_1d(GT::I32, 1);

        let x = ctx.op_get_rows(&rwkv.emb, &token_idx);

        let mut initial_state = (0..rwkv.n_layers)
            .map(|_| RWKVLayerState::new(ctx, rwkv.n_embed))
            .collect::<Vec<_>>();

        // let initial_probs = Array1::zeros(rwkv.n_vocab);
        let rwkv_graph = rwkv.evaluate_ops(ctx, &mut initial_state, 666);

        Self {
            rwkv,
            state: initial_state,
            // last_probs: initial_probs,
            tokenizer,
        }
    }

    /// Feeds some text to the model. A closure can be specified here to allow
    /// showing progress since it can take a while for large prompts/models.
    ///
    /// Evaluating the model generates probabilities, but they're not used here.
    pub fn feed_prompt<S: AsRef<str>>(&mut self, s: S, f: Option<impl Fn(String)>) -> Result<()> {
        // let toks = self
        //     .tokenizer
        //     .encode(s.as_ref(), false)
        //     .map_err(|e| anyhow!(e))?;

        // for tid in toks.get_ids().iter() {
        //     self.last_probs = self.rwkv.evaluate(*tid as usize, &mut self.state);
        //     if let Some(f) = &f {
        //         self.tokenizer
        //             .decode(vec![*tid], false)
        //             .map(f)
        //             .map_err(|e| anyhow!(e))?;
        //     }
        // }
        Ok(())
    }

    /// Infers the next token. Takes a closure that looks at the probabilities
    /// vector and figures out what token to pick.
    pub fn infer_next_token(
        &mut self,
        mut samplefun: impl FnMut(&ArrayView1<f32>) -> Result<usize>,
    ) -> Result<Option<String>> {
        // let tokid = samplefun(&self.last_probs.view())?;
        // if tokid == 0 {
        //     return Ok(None);
        // }
        // let output = self
        //     .tokenizer
        //     .decode(vec![tokid as u32], false)
        //     .map_err(|e| anyhow!(e))?;
        // self.last_probs = self.rwkv.evaluate(tokid, &mut self.state);
        // Ok(Some(output))
        todo!()
    }
}
