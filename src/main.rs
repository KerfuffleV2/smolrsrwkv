use std::io::Write;

use anyhow::{anyhow, Result};
use ndarray::ArrayView1;
use tokenizers::Tokenizer;

pub mod context;
pub mod loader;
pub mod model;
pub mod util;

use crate::{
    context::RWKVContext,
    model::RWKV,
    util::{mmap_file, sample_probs},
};

const TESTSTR: &str = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";
const MODEL: &str = "./RWKV-4-Pile-430M-20220808-8066.safetensors";
const TOKENIZER: &str = "./20B_tokenizer.json";
type ModelType = f32;

fn main() -> Result<()> {
    let mut rng = rand::thread_rng();
    let tokenizer = Tokenizer::from_file(TOKENIZER).map_err(|e| anyhow!(e))?;
    let rwkv: RWKV<ModelType> = mmap_file(MODEL)?.try_into()?;
    let mut context = RWKVContext::new(rwkv, tokenizer);

    let show_token = |token| {
        print!("{token}");
        std::io::stdout().flush().ok();
    };
    let mut do_sample =
        |probs: &ArrayView1<ModelType>| Ok(sample_probs(&mut rng, probs, 1.0, 0.85));

    println!(
        "** Loaded: layers={}, embed={:?}",
        context.n_layers, context.n_embed
    );

    context.feed_prompt(TESTSTR, Some(show_token))?;

    while let Some(token) = context.infer_next_token(&mut do_sample)? {
        show_token(token);
    }

    println!(" [end of text]");
    Ok(())
}
