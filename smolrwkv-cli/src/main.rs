use std::time::Instant;

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
        args::EvalType::GGMLf32 | args::EvalType::GGMLQ4_0 | args::EvalType::GGMLQ4_1 => {
            use smolrwkv::ggml::{context::RWKVContext, loader::RwkvGgmlType};
            let wtype = match args.eval_mode {
                args::EvalType::GGMLf32 => RwkvGgmlType::Float32,
                args::EvalType::GGMLQ4_0 => RwkvGgmlType::Q4_0,
                args::EvalType::GGMLQ4_1 => RwkvGgmlType::Q4_1,
                _ => panic!("Impossible: Bad eval mode!"),
            };
            info!("Backend type: GGML {wtype:?}");
            Ctx::Ggml(RWKVContext::new(
                (wtype, tdm).try_into()?,
                tokenizer,
                args.max_eval_threads,
            ))
        }
    };

    let (n_layers, n_embed, n_vocab) = context.params();
    info!("Loaded: layers={n_layers}, embed={n_embed}, vocab={n_vocab}",);

    let max_tokens = args.max_tokens.unwrap_or(usize::MAX);

    let mut do_sample = |probs: &ArrayView1<FloatType>| {
        Ok(sample_probs(
            &mut rng,
            probs,
            args.forever,
            args.temperature,
            args.top_p,
        ))
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
            let used_mem = context.rwkv.ctx.used_mem();
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
