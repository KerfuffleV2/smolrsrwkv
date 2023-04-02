use std::io::Write;

use anyhow::{anyhow, Result};
use ndarray::ArrayView1;
use tokenizers::Tokenizer;

pub mod quantized;
pub mod simple;

// FIXME: Make these comments not garbage.
/// Context related functions. Holds the model state and and last probabilities vector.
// pub mod context;
/// Functions related to loading the model from disk.
// pub mod loader;
/// The actual model and code related to evaluating it.
// pub mod model;
// pub mod model_impls;
//
/// Traits representing the components involved in evaluating RWKV.
pub mod model_traits;
pub mod rwkvops;
/// Utility functions.
pub mod util;
// pub mod modelna;

#[allow(unused_imports)]
use crate::{
    quantized as Q, simple as S,
    util::{mmap_file, run_threadlimited, sample_probs},
};

/// Used as the prompt.
const PROMPT: &str = "\nIn a shocking finding, scientists discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";

/// Example of a small model to try.
// const MODEL: &str = "./RWKV-4-Pile-430M-20220808-8066.safetensors";
const MODEL: &str = "../../models/RWKV-4-Pile-3B-20221110-ctx4096.safetensors";

/// Tokenizer definition file. See README.
const TOKENIZER: &str = "./20B_tokenizer.json";
/// When enabled, we'll try to generate tokens forever by setting the probability of the
/// EndOfText token to 0.0.
const FOREVER: bool = true;
/// The higher the temperature, the more random the results.
const TEMPERATURE: f32 = 1.0;
/// Here's a better explanation of top_p than I could ever write: https://huggingface.co/blog/how-to-generate
const TOP_P: f32 = 0.85;

/// Number of threads to use when loading the model.
const MAX_LOAD_THREADS: usize = 4;
/// Number of threads to use when evaluating the model. 0 means one per logical core.
const MAX_EVAL_THREADS: usize = 6;

/// Currently can only be f32. It would be pretty easy to add f64 if you wanted to waste
/// massive amounts of memory for no reason. Unfortunately, using 16, 8 or 4 bit types
/// here would be quite difficult since the ndarray crate only supports calculation with
/// `f32` and `f64`
type ModelType = f32;

fn load_simple(
    mm: mmap_rs::Mmap,
    tokenizer: Tokenizer,
) -> Result<S::context::RWKVContext<ModelType>> {
    Ok(S::context::RWKVContext::new(mm.try_into()?, tokenizer))
}

fn load_quantized(mm: mmap_rs::Mmap, tokenizer: Tokenizer) -> Result<Q::context::RWKVContext> {
    Ok(Q::context::RWKVContext::new(mm.try_into()?, tokenizer))
}

fn main() -> Result<()> {
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(124);
    println!("* Loading tokenizer from: {TOKENIZER}");
    let tokenizer = Tokenizer::from_file(TOKENIZER).map_err(|e| anyhow!(e))?;
    println!("* Loading model from: {MODEL}");
    let mm = mmap_file(MODEL)?;
    let mut context = run_threadlimited(MAX_LOAD_THREADS, || {
        // load_simple(mm, tokenizer)
        load_quantized(mm, tokenizer)
    })?;

    // Helper to print out a string without a newline and then flush the console.
    let show_token = |token| {
        print!("{token}");
        std::io::stdout().flush().ok();
    };
    let mut do_sample = |probs: &ArrayView1<ModelType>| {
        Ok(sample_probs(&mut rng, probs, FOREVER, TEMPERATURE, TOP_P))
    };

    println!(
        "* Loaded: layers={}, embed={}, vocab={}",
        context.rwkv.n_layers, context.rwkv.n_embed, context.rwkv.n_vocab
    );

    run_threadlimited(MAX_EVAL_THREADS, || {
        context.feed_prompt(PROMPT, Some(show_token))?;

        while let Some(token) = context.infer_next_token(&mut do_sample)? {
            show_token(token);
        }
        Result::<_, anyhow::Error>::Ok(())
    })?;

    println!(" [end of text]");
    Ok(())
}
