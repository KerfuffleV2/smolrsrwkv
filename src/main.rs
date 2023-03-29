use std::io::Write;

use anyhow::{anyhow, Result};
use ndarray::ArrayView1;
use tokenizers::Tokenizer;

pub mod context;
pub mod loader;
pub mod model;
pub mod util;

use context::RWKVContext;
use model::*;
use util::*;

const TESTSTR: &str = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";
const MODEL: &str = "./RWKV-4-Pile-430M-20220808-8066.safetensors";
const TOKENIZER: &str = "./20B_tokenizer.json";

fn main() -> Result<()> {
    let mut rng = rand::thread_rng();
    let tokenizer = Tokenizer::from_file(TOKENIZER).map_err(|e| anyhow!(e))?;
    let rwkv: RWKV<Ty> = mmap_file(MODEL)?.try_into()?;
    let mut context = RWKVContext::new(rwkv, tokenizer);
    let mut samplefun = |probs: &ArrayView1<Ty>| Ok(sample_probs(&mut rng, probs, 1.0, 0.85));

    println!(
        "** Loaded: layers={}, embed={:?}",
        context.n_layers, context.n_embed
    );

    context.feed_prompt(TESTSTR)?;

    while let Some(token) = context.infer_next_token(&mut samplefun)? {
        print!("{token}");
        std::io::stdout().flush().ok();
    }

    println!(" [end of text]");
    Ok(())
}
