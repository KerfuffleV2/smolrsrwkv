use anyhow::{anyhow, Ok, Result};
use clap::Parser;
use ndarray::{Array2, ArrayView1};
use rand::{rngs::StdRng, SeedableRng};
use tokenizers::Tokenizer;
use tracing::info;

use smolrwkv::{
    loader::TensorDataMap,
    quantized::model::TensorQ2,
    simple as S,
    util::{mmap_file, run_threadlimited, sample_probs},
};

mod args;
mod util;

use args::Args;
use tracing_subscriber::fmt::format::FmtSpan;
use util::{show_token, Ctx, FloatType};

pub fn setup_logging() {
    use tracing::metadata::LevelFilter;
    use tracing_subscriber::{fmt, layer::SubscriberExt, Layer};

    let fmt_layer = fmt::layer()
        .compact()
        // .with_span_events(FmtSpan::FULL)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .with_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        );

    let sub = tracing_subscriber::registry().with(fmt_layer);

    tracing::subscriber::set_global_default(sub).expect("Could tracing subscriber");
}

fn go() -> Result<()> {
    let args = Args::parse();
    let tokenizerfn = &args.tokenizer;
    let modelfn = &args.model;
    info!("Using configuration: {args:?}\n");

    let mut rng: rand::rngs::StdRng = if let Some(seed) = &args.seed {
        StdRng::seed_from_u64(*seed)
    } else {
        StdRng::from_entropy()
    };

    info!("Loading tokenizer from: {tokenizerfn}");
    let tokenizer = Tokenizer::from_file(tokenizerfn).map_err(|e| anyhow!(e))?;
    info!("Loading model from: {modelfn}");
    let mm = mmap_file(modelfn)?;
    let tdm: TensorDataMap<'_> = (modelfn.clone(), mm.as_slice()).try_into()?;

    let mut context = {
        use smolrwkv::ggml::{context::*, loader::*, model::*};

        let rwkv: RWKV = tdm.try_into()?;
        let rwkvctx = Ctx::GgmlCtx(RWKVContext::new(rwkv, tokenizer));
        rwkvctx
    };

    // let mut context = run_threadlimited(args.max_load_threads, move || {
    //     anyhow::Ok(if args.no_quantized {
    //         info!("Model type: non-quantized (full 32bit).");
    //         Ctx::FloatCtx(
    //             S::context::RWKVContext::<FloatType, Array2<FloatType>>::new(
    //                 tdm.try_into()?,
    //                 tokenizer,
    //             ),
    //         )
    //     } else {
    //         info!("Model type: 8 bit-quantized weights.");
    //         Ctx::QuantCtx(S::context::RWKVContext::<FloatType, TensorQ2>::new(
    //             tdm.try_into()?,
    //             tokenizer,
    //         ))
    //     })
    // })?;

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
    info!("Loaded: layers={n_layers}, embed={n_embed}, vocab={n_vocab}",);

    let max_tokens = args.max_tokens.unwrap_or(usize::MAX);
    let (tcount, elapsed) = {
        use std::time::Instant;

        context.feed_prompt(&args.prompt, Some(show_token))?;

        let mut tcount = 0;
        let stime = Instant::now();
        while let Some(token) = context.infer_next_token(&mut do_sample)? {
            show_token(token);
            tcount += 1;
            if tcount > max_tokens {
                break;
            }
        }
        let etime = Instant::now();
        Ok((tcount, etime - stime))
    }?;
    // let (tcount, elapsed) = run_threadlimited(args.max_eval_threads, || {
    //     use std::time::Instant;

    //     context.feed_prompt(&args.prompt, Some(show_token))?;

    //     let mut tcount = 0;
    //     let stime = Instant::now();
    //     while let Some(token) = context.infer_next_token(&mut do_sample)? {
    //         show_token(token);
    //         tcount += 1;
    //         if tcount > max_tokens {
    //             break;
    //         }
    //     }
    //     let etime = Instant::now();
    //     Ok((tcount, etime - stime))
    // })?;

    println!(" [end of text]\n");
    let tps = tcount as f64 / (elapsed.as_millis() as f64 / 1000.0);
    info!(
        "Completion. Token(s) generated: {tcount}, elapsed time: {:?}, TPS: {tps}",
        elapsed
    );
    Ok(())
}

pub fn main() -> Result<()> {
    setup_logging();
    go()
}
