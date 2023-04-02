use std::io::Write;

use anyhow::{anyhow, Result};
use clap::Parser;
use ndarray::ArrayView1;
use tokenizers::Tokenizer;

use smolrwkv::{
    quantized as Q, simple as S,
    util::{mmap_file, run_threadlimited, sample_probs},
};

mod args;
use args::Args;

enum Ctx {
    FloatCtx(S::context::RWKVContext<f32>),
    QuantCtx(Q::context::RWKVContext),
}

impl Ctx {
    fn params(&self) -> (usize, usize, usize) {
        match self {
            Ctx::FloatCtx(ctx) => (ctx.rwkv.n_layers, ctx.rwkv.n_embed, ctx.rwkv.n_vocab),
            Ctx::QuantCtx(ctx) => (ctx.rwkv.n_layers, ctx.rwkv.n_embed, ctx.rwkv.n_vocab),
        }
    }

    pub fn feed_prompt<S: AsRef<str>>(&mut self, s: S, f: Option<impl Fn(String)>) -> Result<()> {
        match self {
            Ctx::FloatCtx(ctx) => ctx.feed_prompt(s, f),
            Ctx::QuantCtx(ctx) => ctx.feed_prompt(s, f),
        }
    }

    pub fn infer_next_token(
        &mut self,
        samplefun: impl FnMut(&ArrayView1<f32>) -> Result<usize>,
    ) -> Result<Option<String>> {
        match self {
            Ctx::FloatCtx(ctx) => ctx.infer_next_token(samplefun),
            Ctx::QuantCtx(ctx) => ctx.infer_next_token(samplefun),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tokenizerfn = &args.tokenizer;
    let modelfn = &args.model;
    println!("> Using configuration: {args:?}\n");
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(124);
    println!("* Loading tokenizer from: {tokenizerfn}");
    let tokenizer = Tokenizer::from_file(tokenizerfn).map_err(|e| anyhow!(e))?;
    println!("* Loading model from: {modelfn}");
    let mm = mmap_file(modelfn)?;
    let mut context = run_threadlimited(args.max_load_threads, move || {
        anyhow::Ok(if args.no_quantized {
            Ctx::FloatCtx(S::context::RWKVContext::<f32>::new(
                mm.try_into()?,
                tokenizer,
            ))
        } else {
            Ctx::QuantCtx(Q::context::RWKVContext::new(mm.try_into()?, tokenizer))
        })
    })?;

    // Helper to print out a string without a newline and then flush the console.
    let show_token = |token| {
        print!("{token}");
        std::io::stdout().flush().ok();
    };
    let mut do_sample = |probs: &ArrayView1<f32>| {
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
        Result::<_, anyhow::Error>::Ok(())
    })?;

    println!(" [end of text]");
    Ok(())
}
