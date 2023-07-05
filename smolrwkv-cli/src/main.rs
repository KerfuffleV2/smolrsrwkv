use std::time::Instant;

use anyhow::{anyhow, Ok, Result};
use clap::Parser;
use ndarray::{Array2, ArrayView1};
use rand::{rngs::StdRng, SeedableRng};
use tokenizers::Tokenizer;
use tracing::info;

use llm_samplers::prelude::*;

use smolrwkv::{
    loader::TensorDataMap,
    quantized::model::TensorQ2,
    simple as S,
    util::{mmap_file, run_threadlimited},
};

mod args;
mod util;

use args::Args;
use util::{show_token, Ctx, FloatType};

pub fn setup_logging() {
    use tracing::metadata::LevelFilter;
    use tracing_subscriber::{fmt, fmt::time::FormatTime, layer::SubscriberExt, Layer};

    #[derive(Clone, Debug, Copy, PartialEq, Eq)]
    struct Elapsed(Instant);
    impl FormatTime for Elapsed {
        fn format_time(&self, w: &mut fmt::format::Writer<'_>) -> std::fmt::Result {
            let e = self.0.elapsed();
            write!(w, "{:4}.{:02}s", e.as_secs(), e.subsec_millis() / 10)
        }
    }

    let fmt_layer = fmt::layer()
        .compact()
        // .with_span_events(FmtSpan::FULL)
        .with_timer(Elapsed(Instant::now()))
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

    info!("Loading tokenizer from: {tokenizerfn}");
    let tokenizer = Tokenizer::from_file(tokenizerfn).map_err(|e| anyhow!(e))?;
    info!("Loading model from: {modelfn}");
    let mm = mmap_file(modelfn)?;
    #[cfg(unix)]
    mm.advise(memmap2::Advice::Random)?;
    let tdm: TensorDataMap<'_> = (modelfn.clone(), &mm).try_into()?;
    #[cfg(unix)]
    {
        mm.advise(memmap2::Advice::WillNeed)?;
        mm.advise(memmap2::Advice::Sequential)?;
    }

    let context = match &args.eval_mode {
        args::EvalType::NDf32 => {
            Ctx::NdFloat32(run_threadlimited(args.max_load_threads, move || {
                anyhow::Ok({
                    info!("Backend type: NDArray non-quantized (full 32bit).");
                    S::context::RWKVContext::<FloatType, Array2<FloatType>>::new(
                        tdm.try_into()?,
                        tokenizer,
                    )
                })
            })?)
        }

        args::EvalType::NDu8 => {
            Ctx::NdQuant8(run_threadlimited(args.max_load_threads, move || {
                anyhow::Ok({
                    info!("Backend type: NDArray 8 bit-quantized weights.");
                    S::context::RWKVContext::<FloatType, TensorQ2>::new(tdm.try_into()?, tokenizer)
                })
            })?)
        }
        #[cfg(feature = "ggml")]
        args::EvalType::GGMLf32
        | args::EvalType::GGMLQ8_0
        | args::EvalType::GGMLQ4_0
        | args::EvalType::GGMLQ4_1
        | args::EvalType::GGMLQ5_0
        | args::EvalType::GGMLQ2_K
        | args::EvalType::GGMLQ3_K
        | args::EvalType::GGMLQ4_K
        | args::EvalType::GGMLQ5_K
        | args::EvalType::GGMLQ5_1
        | args::EvalType::GGMLQ6_K => {
            use smolrwkv::ggml::{
                context::RWKVContext,
                loader::{load_rwkv, ElementType},
            };

            let wtype = match args.eval_mode {
                args::EvalType::GGMLf32 => ElementType::F32,
                args::EvalType::GGMLQ8_0 => ElementType::Q8_0,
                args::EvalType::GGMLQ4_0 => ElementType::Q4_0,
                args::EvalType::GGMLQ4_1 => ElementType::Q4_1,
                args::EvalType::GGMLQ5_0 => ElementType::Q5_0,
                args::EvalType::GGMLQ5_1 => ElementType::Q5_1,
                args::EvalType::GGMLQ2_K => ElementType::Q2_K,
                args::EvalType::GGMLQ3_K => ElementType::Q3_K,
                args::EvalType::GGMLQ4_K => ElementType::Q4_K,
                args::EvalType::GGMLQ5_K => ElementType::Q5_K,
                args::EvalType::GGMLQ6_K => ElementType::Q6_K,
                _ => panic!("Impossible: Bad eval mode!"),
            };
            info!("Backend type: GGML {wtype:?}");
            let ltensors = load_rwkv(args.max_load_threads, ElementType::F32, wtype, tdm)?;
            Ctx::Ggml(RWKVContext::new(
                (wtype, ltensors).try_into()?,
                tokenizer,
                args.max_eval_threads,
            )?)
        }
    };

    let (n_layers, n_embed, n_vocab) = context.params();
    info!("Loaded: layers={n_layers}, embed={n_embed}, vocab={n_vocab}",);

    let max_tokens = args.max_tokens.unwrap_or(usize::MAX);

    let mut sres = SimpleSamplerResources::new(
        Some(Box::new(if let Some(seed) = &args.seed {
            StdRng::seed_from_u64(*seed)
        } else {
            StdRng::from_entropy()
        })),
        Some(Vec::with_capacity(max_tokens.min(8192))),
    );
    let mut samplers = SamplerChain::new();
    if args.forever {
        samplers += SampleFlatBias::new(&[(0u32, f32::NEG_INFINITY)]);
    }
    samplers = samplers
        + SampleRepetition::default().penalty(1.15)
        + SampleFreqPresence::default().presence(0.02).frequency(0.05)
        + SampleTemperature::new(args.temperature)
        + SampleMirostat1::new(n_vocab, 5.0, 0.1);

    let mut do_sample = |probs: &ArrayView1<FloatType>| -> Result<usize> {
        let mut logits = Logits::try_from_iter(probs.iter().copied())?;
        let tid = logits
            .sample_token(&mut sres, &mut samplers)?
            .expect("No token sampled!?");
        sres.with_last_tokens_mut(&mut |lt| lt.push(tid))?;
        Ok(tid as usize)
    };

    // FIXME: Duplicated code.
    // The solution isn't as simple as it may appear because the GGML types aren't Sync
    // which means thread limiting requires special handling.
    let (tcount, elapsed) = match context {
        Ctx::NdFloat32(mut context) => run_threadlimited(args.max_eval_threads, || {
            context.feed_prompt(args.prompt, Some(show_token))?;

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
        }),
        Ctx::NdQuant8(mut context) => run_threadlimited(args.max_eval_threads, || {
            context.feed_prompt(args.prompt, Some(show_token))?;

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
        }),
        #[cfg(feature = "ggml")]
        Ctx::Ggml(mut context) => {
            context.feed_prompt(args.prompt, Some(show_token))?;

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
            let used_mem = context.rwkv.ctx.used_mem()?;
            println!();
            info!(
                "GGML memory used: {:.3}GiB",
                (used_mem as f64) / (1024.0f64 * 1024.0 * 1024.0)
            );
            Ok((tcount, etime - stime))
        }
    }?;

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
