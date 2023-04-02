use anyhow::{anyhow, Ok, Result};
use clap::Parser;
use ndarray::ArrayView1;
use rand::{rngs::StdRng, SeedableRng};
use tokenizers::Tokenizer;

use smolrwkv::{
    quantized as Q, simple as S,
    util::{mmap_file, run_threadlimited, sample_probs},
};

mod args;
mod util;

use args::Args;
use util::{show_token, Ctx, FloatType};

fn main() -> Result<()> {
    let args = Args::parse();
    let tokenizerfn = &args.tokenizer;
    let modelfn = &args.model;
    println!("> Using configuration: {args:?}\n");

    let mut rng: rand::rngs::StdRng = if let Some(seed) = &args.seed {
        StdRng::seed_from_u64(*seed)
    } else {
        StdRng::from_entropy()
    };

    println!("* Loading tokenizer from: {tokenizerfn}");
    let tokenizer = Tokenizer::from_file(tokenizerfn).map_err(|e| anyhow!(e))?;
    println!("* Loading model from: {modelfn}");
    let mm = mmap_file(modelfn)?;
    let mut context = run_threadlimited(args.max_load_threads, move || {
        anyhow::Ok(if args.no_quantized {
            Ctx::FloatCtx(S::context::RWKVContext::<FloatType>::new(
                mm.try_into()?,
                tokenizer,
            ))
        } else {
            Ctx::QuantCtx(Q::context::RWKVContext::new(mm.try_into()?, tokenizer))
        })
    })?;

    let mut do_sample = |probs: &ArrayView1<FloatType>| {
        Ok(sample_probs(
            &mut rng,
            probs,
            args.forever,
            args.temperature,
            args.top_p,
        ))
    };

    let (n_layers, n_embed, n_vocab) = context.params();
    println!("* Loaded: layers={n_layers}, embed={n_embed}, vocab={n_vocab}",);

    run_threadlimited(args.max_eval_threads, || {
        context.feed_prompt(&args.prompt, Some(show_token))?;

        while let Some(token) = context.infer_next_token(&mut do_sample)? {
            show_token(token);
        }
        Ok(())
    })?;

    println!(" [end of text]");
    Ok(())
}
