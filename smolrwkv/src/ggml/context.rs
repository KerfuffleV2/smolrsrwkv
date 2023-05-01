use anyhow::{anyhow, ensure, Result};
use ndarray::{Array1, ArrayView1};
use tokenizers::Tokenizer;

use rusty_ggml::prelude::*;

use super::model::{RWKVLayerState, RWKV};

/// Context that holds the state of the RWKV model.
pub struct RWKVContext {
    /// The RWKV model data — immutable.
    pub rwkv: RWKV,
    /// Model state.
    pub state: Vec<RWKVLayerState>,
    /// Probabilities from the last step (starts filled with zeros).
    pub last_probs: Array1<f32>,
    /// It's a 1d tensor with length 1 that contains the token ID.
    pub token_tensor: GTensor1,
    /// This is the base of the graph and also where the probs appear.
    pub result_tensor: GTensor1,
    /// The GGML computation graph.
    pub ggml_graph: GGraph,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
}

impl RWKVContext {
    pub fn new(rwkv: RWKV, tokenizer: Tokenizer, eval_threads: usize) -> Result<Self> {
        let ctx = &rwkv.ctx;

        let token_tensor = ctx.tensor(GType::I32, [1])?;
        let mut initial_state = (0..rwkv.n_layers)
            .map(|_| RWKVLayerState::new(ctx, rwkv.n_embed))
            .collect::<Vec<_>>();

        let initial_probs = Array1::zeros(rwkv.n_vocab);
        let rwkv_ops_graph = rwkv.evaluate_ops(&mut initial_state, token_tensor.clone());

        let mut ggml_graph = GGraph::new(eval_threads);
        ggml_graph.build_forward_expand(&rwkv_ops_graph)?;
        initial_state.iter().try_for_each(|s| {
            ggml_graph.build_forward_expand(&s.tm_last_x)?;
            ggml_graph.build_forward_expand(&s.cm_last_x)?;
            ggml_graph.build_forward_expand(&s.tm_aa)?;
            ggml_graph.build_forward_expand(&s.tm_bb)?;
            ggml_graph.build_forward_expand(&s.tm_pp)?;
            anyhow::Ok(())
        })?;

        Ok(Self {
            rwkv,
            state: initial_state,
            last_probs: initial_probs,
            token_tensor,
            result_tensor: rwkv_ops_graph,
            ggml_graph,
            tokenizer,
        })
    }

    /// Feeds some text to the model. A closure can be specified here to allow
    /// showing progress since it can take a while for large prompts/models.
    ///
    /// Evaluating the model generates probabilities, but they're not used here.
    pub fn feed_prompt<S: AsRef<str>>(&mut self, s: S, f: Option<impl Fn(String)>) -> Result<()> {
        let ctx = &self.rwkv.ctx;
        let toks = self
            .tokenizer
            .encode(s.as_ref(), false)
            .map_err(|e| anyhow!(e))?;

        for tid in toks.get_ids().iter() {
            self.token_tensor.set_i32_1d(0, *tid as i32);
            ctx.compute(&mut self.ggml_graph)?;
            if let Some(f) = &f {
                self.tokenizer
                    .decode(vec![*tid], false)
                    .map(f)
                    .map_err(|e| anyhow!(e))?;
            }
        }
        ensure!(
            self.result_tensor.shape()[0] == self.last_probs.len()
                && self.result_tensor.elements() == self.last_probs.len(),
            "Unexpected shape for result tensor"
        );
        self.result_tensor.copy_to_slice_f32(
            self.last_probs
                .as_slice_mut()
                .expect("Could get slice from last_probs?"),
        );
        Ok(())
    }

    /// Infers the next token. Takes a closure that looks at the probabilities
    /// vector and figures out what token to pick.
    pub fn infer_next_token(
        &mut self,
        mut samplefun: impl FnMut(&ArrayView1<f32>) -> Result<usize>,
    ) -> Result<Option<String>> {
        let ctx = &self.rwkv.ctx;

        let tokid = samplefun(&self.last_probs.view())?;
        if tokid == 0 {
            return Ok(None);
        }
        let output = self
            .tokenizer
            .decode(vec![tokid as u32], false)
            .map_err(|e| anyhow!(e))?;

        self.token_tensor.set_i32_1d(0, tokid as i32);

        ctx.compute(&mut self.ggml_graph)?;
        ensure!(
            self.result_tensor.shape()[0] == self.last_probs.len()
                && self.result_tensor.elements() == self.last_probs.len(),
            "Unexpected shape for result tensor"
        );
        self.result_tensor.copy_to_slice_f32(
            self.last_probs
                .as_slice_mut()
                .expect("Could get slice from last_probs?"),
        );
        Ok(Some(output))
    }
}
