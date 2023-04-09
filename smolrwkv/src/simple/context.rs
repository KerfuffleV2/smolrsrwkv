#![allow(clippy::upper_case_acronyms)]

use anyhow::{anyhow, Result};
use ndarray::{Array1, ArrayView1};
use tokenizers::Tokenizer;

use crate::{
    model_traits::RunRWKV,
    simple::model::{RWKVLayerState, RWKV},
    util::{ParDot, ReqOps},
};

/// Context that holds the state of the RWKV model.
pub struct RWKVContext<T, WT> {
    /// The RWKV model data — immutable.
    pub rwkv: RWKV<T, WT>,
    /// Model state.
    pub state: Vec<RWKVLayerState<T>>,
    /// Probabilities from the last step (starts filled with zeros).
    pub last_probs: Array1<T>,
    /// The tokenizer.
    pub tokenizer: Tokenizer,
}

impl<T: ReqOps, WT: ParDot<Output = Array1<T>>> RWKVContext<T, WT> {
    pub fn new(rwkv: RWKV<T, WT>, tokenizer: Tokenizer) -> Self {
        let initial_state = std::iter::repeat(RWKVLayerState::new(rwkv.n_embed))
            .take(rwkv.n_layers)
            .collect::<Vec<_>>();
        let initial_probs = Array1::zeros(rwkv.n_vocab);

        Self {
            rwkv,
            state: initial_state,
            last_probs: initial_probs,
            tokenizer,
        }
    }

    /// Feeds some text to the model. A closure can be specified here to allow
    /// showing progress since it can take a while for large prompts/models.
    ///
    /// Evaluating the model generates probabilities, but they're not used here.
    pub fn feed_prompt<S: AsRef<str>>(&mut self, s: S, f: Option<impl Fn(String)>) -> Result<()> {
        let toks = self
            .tokenizer
            .encode(s.as_ref(), false)
            .map_err(|e| anyhow!(e))?;

        for tid in toks.get_ids().iter() {
            self.last_probs = self.rwkv.evaluate(*tid as usize, &mut self.state);
            if let Some(f) = &f {
                self.tokenizer
                    .decode(vec![*tid], false)
                    .map(f)
                    .map_err(|e| anyhow!(e))?;
            }
        }
        Ok(())
    }

    /// Infers the next token. Takes a closure that looks at the probabilities
    /// vector and figures out what token to pick.
    pub fn infer_next_token(
        &mut self,
        mut samplefun: impl FnMut(&ArrayView1<T>) -> Result<usize>,
    ) -> Result<Option<String>> {
        let tokid = samplefun(&self.last_probs.view())?;
        if tokid == 0 {
            return Ok(None);
        }
        let output = self
            .tokenizer
            .decode(vec![tokid as u32], false)
            .map_err(|e| anyhow!(e))?;
        self.last_probs = self.rwkv.evaluate(tokid, &mut self.state);
        Ok(Some(output))
    }
}
