#![allow(clippy::upper_case_acronyms)]
use std::io::Write;

use anyhow::{anyhow, Result};
use ndarray::Array1;

pub mod loader;
pub mod model;
pub mod util;

use model::*;
use util::*;

const TESTSTR: &str = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";
const MODEL: &str = "./RWKV-4-Pile-430M-20220808-8066.safetensors";
const TOKENIZER: &str = "./20B_tokenizer.json";

fn main() -> Result<()> {
    let mut rng = rand::thread_rng();
    let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER).map_err(|e| anyhow!(e))?;
    let rwkv: RWKV<Ty> = mmap_file(MODEL)?.try_into()?;
    let n_embed = rwkv.emb.shape()[1];
    let n_layers = rwkv.layers.len();
    println!(
        "** Loaded: layers={}, embed={:?}",
        n_layers,
        rwkv.emb.shape()
    );
    let mut state = std::iter::repeat(RWKVLayerState::new(n_embed))
        .take(n_layers)
        .collect::<Vec<_>>();
    let toks = tokenizer.encode(TESTSTR, false).map_err(|e| anyhow!(e))?;
    let mut probs = Array1::<f32>::zeros(rwkv.emb.shape()[0]);

    toks.get_ids().iter().for_each(|tid| {
        probs = rwkv.evaluate(*tid as usize, &mut state);
        let tokstr = tokenizer.decode(vec![*tid], false).unwrap();
        print!("{}", tokstr);
        std::io::stdout().flush().ok();
    });
    loop {
        let tokid = sample_probs(&mut rng, &probs.view(), 1.0, 0.85);
        if tokid == 0 {
            println!(" [end of text]");
            break;
        }
        let tokstr = tokenizer.decode(vec![tokid as u32], false).unwrap();
        print!("{}", tokstr);
        std::io::stdout().flush().ok();
        probs = rwkv.evaluate(tokid, &mut state);
    }
    println!("Hokay.");
    Ok(())
}
