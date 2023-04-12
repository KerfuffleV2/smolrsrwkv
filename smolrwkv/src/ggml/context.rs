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
    pub token_tensor: Tensor,
    pub result_tensor: Tensor,
    pub ggml_graph: ComputationGraph,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
}

impl RWKVContext {
    pub fn new(rwkv: RWKV, tokenizer: Tokenizer) -> Self {
        let ctx = &rwkv.ctx;

        let token_tensor = ctx.new_tensor_1d(GT::I32, 1);
        let mut initial_state = (0..rwkv.n_layers)
            .map(|_| RWKVLayerState::new(ctx, rwkv.n_embed))
            .collect::<Vec<_>>();

        let initial_probs = Array1::zeros(rwkv.n_vocab);
        let rwkv_ops_graph = rwkv.evaluate_ops(ctx, &mut initial_state, token_tensor.share());

        let mut ggml_graph = ComputationGraph::new(4);
        ggml_graph.build_forward_expand(&rwkv_ops_graph);

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

        println!(" <<{:?}>> ", self.last_probs[0]);
        for tid in toks.get_ids().iter() {
            let tid = *tid as i32;
            self.token_tensor.set_i32_1d(0, tid);
            ctx.graph_compute(&mut self.ggml_graph);

            // print!(" <<{tid}:{:?}>> ", self.last_probs[0]);
            if let Some(f) = &f {
                self.tokenizer
                    .decode(vec![tid as u32], false)
                    .map(f)
                    .map_err(|e| anyhow!(e))?;
            }
        }
        assert_eq!(
            self.result_tensor.get_ne()[0] as usize,
            self.last_probs.len()
        );
        unsafe {
            (self.result_tensor.data() as *const f32)
                .copy_to_nonoverlapping(self.last_probs.as_mut_ptr(), self.last_probs.len());
        }
        // self.ggml_graph.dump_graph();
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

        let tid = tokid as i32;
        self.token_tensor.set_i32_1d(0, tid);

        ctx.graph_compute(&mut self.ggml_graph);
        assert_eq!(
            self.result_tensor.get_ne()[0] as usize,
            self.last_probs.len()
        );
        unsafe {
            (self.result_tensor.data() as *const f32)
                .copy_to_nonoverlapping(self.last_probs.as_mut_ptr(), self.last_probs.len());
        }
        // print!(" <<{tid}:{:?}>> ", self.last_probs[0]);
        Ok(Some(output))
    }
}
