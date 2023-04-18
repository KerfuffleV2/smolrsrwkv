use anyhow::{anyhow, Result};
use ndarray::{Array1, ArrayView1};
use tokenizers::Tokenizer;

use ggml::{ComputationGraph, Tensor, Type as GT};

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
    pub token_tensor: Tensor,
    /// This is the base of the graph and also where the probs appear.
    pub result_tensor: Tensor,
    /// The GGML computation graph.
    pub ggml_graph: ComputationGraph,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
}

impl RWKVContext {
    pub fn new(rwkv: RWKV, tokenizer: Tokenizer, eval_threads: usize) -> Self {
        let ctx = &rwkv.ctx;

        let token_tensor = ctx.new_tensor_1d(GT::I32, 1);
        let mut initial_state = (0..rwkv.n_layers)
            .map(|_| RWKVLayerState::new(ctx, rwkv.n_embed))
            .collect::<Vec<_>>();

        let initial_probs = Array1::zeros(rwkv.n_vocab);
        let rwkv_ops_graph = rwkv.evaluate_ops(ctx, &mut initial_state, token_tensor.share());

        let mut ggml_graph = ComputationGraph::new(eval_threads);
        ggml_graph.build_forward_expand(&rwkv_ops_graph);
        initial_state.iter().for_each(|s| {
            ggml_graph.build_forward_expand(&s.tm_last_x);
            ggml_graph.build_forward_expand(&s.cm_last_x);
            ggml_graph.build_forward_expand(&s.tm_aa);
            ggml_graph.build_forward_expand(&s.tm_bb);
            ggml_graph.build_forward_expand(&s.tm_pp);
        });

        Self {
            rwkv,
            state: initial_state,
            last_probs: initial_probs,
            token_tensor,
            result_tensor: rwkv_ops_graph,
            ggml_graph,
            tokenizer,
        }
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
            unsafe {
                self.token_tensor
                    .write_data(bytemuck::bytes_of(&(*tid as i32)));
            }
            ctx.graph_compute(&mut self.ggml_graph);
            if let Some(f) = &f {
                self.tokenizer
                    .decode(vec![*tid], false)
                    .map(f)
                    .map_err(|e| anyhow!(e))?;
            }
        }
        assert_eq!(
            self.result_tensor.get_ne()[0] as usize,
            self.last_probs.len()
        );
        // FIXME: Use ggml tensor manipulation methods?
        unsafe {
            (self.result_tensor.data() as *const f32)
                .copy_to_nonoverlapping(self.last_probs.as_mut_ptr(), self.last_probs.len());
        }
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

        unsafe {
            self.token_tensor
                .write_data(bytemuck::bytes_of(&(tokid as i32)));
        }

        ctx.graph_compute(&mut self.ggml_graph);
        assert_eq!(
            self.result_tensor.get_ne()[0] as usize,
            self.last_probs.len()
        );
        // FIXME: Use ggml tensor manipulation methods?
        unsafe {
            (self.result_tensor.data() as *const f32)
                .copy_to_nonoverlapping(self.last_probs.as_mut_ptr(), self.last_probs.len());
        }
        Ok(Some(output))
    }
}
