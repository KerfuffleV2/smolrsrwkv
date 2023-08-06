use std::{error::Error, str::FromStr};

use clap::{Parser, ValueEnum};

use llm_samplers::{configure::*, prelude::*};

/// Used as the prompt.
const DEFAULT_PROMPT: &str = "\nIn a shocking finding, scientists discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";

/// Example of a small model to try.
const DEFAULT_MODEL: &str = "./RWKV-4-Pile-430M-20220808-8066.safetensors";

/// Tokenizer definition file. See README.
const DEFAULT_TOKENIZER: &str = "./20B_tokenizer.json";

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum EvalType {
    #[value(name = "ndf32")]
    /// ndarray-backed 32 bit floats. Uses a lot of memory.
    NDf32,

    #[value(name = "ndq8")]
    /// ndarray-backed 8 bit quantized. Better memory usage but quite slow.
    NDu8,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlf32")]
    /// GGML-backed 32 bit. As above, uses a lot of memory.
    GGMLf32,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq8_0")]
    /// GGML-backed 8 bit quantized, method 1.
    GGMLQ8_0,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq4_0")]
    /// GGML-backed 4 bit quantized, method 1. Poor quality.
    GGMLQ4_0,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq4_1")]
    /// GGML-backed 4 bit quantized, method 2. Decent quality,
    /// but slower (to load?)
    GGMLQ4_1,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq5_0")]
    /// GGML-backed 5 bit quantized, method 1.
    GGMLQ5_0,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq5_1")]
    /// GGML-backed 5 bit quantized, method 2.
    GGMLQ5_1,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq2_k")]
    /// GGML-backed k_quants 2 bit.
    GGMLQ2_K,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq3_k")]
    /// GGML-backed k_quants 3 bit.
    GGMLQ3_K,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq4_k")]
    /// GGML-backed k_quants 4 bit.
    GGMLQ4_K,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq5_k")]
    /// GGML-backed k_quants 5 bit.
    GGMLQ5_K,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq6_k")]
    /// GGML-backed k_quants 6 bit.
    GGMLQ6_K,
}

#[derive(Clone, Debug, Parser)]
/// Simple commandline interface to RWKV
pub struct Args {
    /// Model filename. Should end in ".st", ".safetensors", ".pt" or ".pth". For the last two,
    /// the torch feature will need to be enabled.
    #[arg(short = 'm', long, default_value = DEFAULT_MODEL)]
    pub model: String,

    /// Tokenizer filename
    #[arg(short = 't', long, default_value = DEFAULT_TOKENIZER)]
    pub tokenizer: String,

    /// Evaluation mode
    #[arg(short = 'e',  long, value_enum, default_value_t = EvalType::NDu8)]
    pub eval_mode: EvalType,

    /// Configure sampler settings using a string in the format: sampler_name:key1=value1:key2=value2
    /// To configure multiple samplers at once, separate the sampler configuration strings with space or '/' (forward slash).
    /// NOTE: Mirostat samplers are incompatible with top-p, top-k, locally typical and tail free samplers.
    /// TIPS:
    ///   1. Sampler options aren't required. For example "mirostat1" will enable Mirostat 1 with its default options.
    ///   2. It's possible to specify partial option names, as long as they are unambiguous.
    ///   3. Underscore and dash are ignored in sampler names, so "top-p" is the same as "topp" or "top_p".
    ///
    /// Configurable samplers (defaults shown in parenthesis):
    ///
    /// freq_presence (default: disabled) - Allows penalizing tokens for presence and frequency. May be specified more than once.
    ///   frequency_penalty(0.0): Penalty to apply to tokens based on frequency. For example, if a token has appeared 3 times within the last_n range then it will have its probability decreased by 3 * frequency_penalty.
    ///   presence_penalty(0.0): Penalty to apply to tokens that are already present within the last_n tokens.
    ///   last_n(64): Number of previous tokens to consider.
    ///
    /// locally_typical (default: disabled) - An approach to sampling that attempts to maximize natural and human-like output. See: https://arxiv.org/abs/2202.00666
    ///   p(1.0): Referred to as τ in the paper. It suggests using 0.2 as a value for story generation and `0.95` for "abstractive summarization" (presumably this means more factual output). 1.0 appears to be the same as disabled which is similar to top-p sampling.
    ///   min_keep(1): Minimum tokens to keep. Setting this to 0 is not recommended.
    ///
    /// mirostat1 (default: disabled) - See: https://arxiv.org/abs/2007.14966
    ///   eta(0.1): Learning rate
    ///   tau(5.0): Target entropy
    ///   mu(tau * 2): Initial learning state value. Setting this is generally not recommended.
    ///
    /// mirostat2 (default: disabled) - See: https://arxiv.org/abs/2007.14966
    ///   eta(0.1): Learning rate
    ///   tau(5.0): Target entropy
    ///   mu(tau * 2): Initial learning state value. Setting this is generally not recommended.
    ///
    /// repetition - Allows setting a repetition penalty. May be specified more than once.
    ///   penalty(1.30): The penalty for repeating tokens. Higher values make the generation less likely to get into a loop, but may harm results when repetitive outputs are desired.
    ///   last_n(64): Number of previous tokens to consider.
    ///
    /// sequence_repetition (default: disabled) - **WARNING: Experimental!** This sampler penalizes repeating sequences of tokens that have already been seen within the last_n window. In other words, the tokens that would result in continuing a sequence that already exists will have the penalty applied. May be specified more than once.
    ///   last_n(64): Number of last tokens to consider.
    ///   min_length(0): The minimum length for a sequence to match. This should generally be set to at least 3.
    ///   flat_penalty(0.0): Flat penalty to apply to the token that would continue the matched sequence.
    ///   stacking_penalty(0.0): Stacking penalty to the token that would continue the matched sequence, it is multiplied by the sequence length.
    ///   tolerance(0): Tolerance basically acts like a wildcard to allow fuzzy sequence matching. For example, if tolerance is set to 1 then 1, 6, 3 could match with 1, 2, 3.
    ///   max_merge(1): Controls the number of consecutive non-matching tokens that the tolerance wildcard can match. Setting this to 0 or 1 deactivates it. Setting it to 2 would allow `1, 6, 6, 3` to match with 1, 2, 3.
    ///
    /// tail_free (default: disabled) - An approach to sampling that attempts to outperform existing nucleus (top-p and top-k) methods. See: https://trentbrick.github.io/Tail-Free-Sampling/
    ///   z(1.0): It is not entirely clear what a reasonable value here is but 1.0 appears to be the same as disabled which is similar to top-p sampling.
    ///   min_keep(1): Minimum tokens to keep. Setting this to 0 is not recommended.
    ///
    /// temperature - Temperature used for sampling.
    ///   temperature(0.8): Temperature (randomness) used for sampling. A higher number is more random.
    ///
    /// top_k - The top k (or min_keep if it is greater) tokens by score are kept during sampling.
    ///   k(40): Number of tokens to keep.
    ///   min_keep(1): Minimum tokens to keep. Setting this to 0 is not recommended.
    ///
    /// top_p - The probability for the top tokens are added until the result is greater or equal to P and at least min_keep tokens have been seen.
    ///   p(0.95): The cumulative probability after which no more tokens are kept for sampling.
    ///   min_keep(1): Minimum tokens to keep. Setting this to 0 is not recommended.
    #[arg(long = "sampler", short = 's', verbatim_doc_comment)]
    pub samplers: Vec<String>,

    /// Generate tokens forever (by setting the probability of the EndOfText token to 0)
    #[arg(long, default_value_t = false)]
    pub forever: bool,

    /// Maximum tokens to generate.
    #[arg(short = 'n', long)]
    pub max_tokens: Option<usize>,

    /// Prompt
    #[arg(short = 'p', long, default_value = DEFAULT_PROMPT)]
    pub prompt: String,

    /// Number of threads to use when loading the model.
    /// Note that this will probably use substantially more memory. 0 means one per logical core.
    #[arg(long, default_value_t = 4)]
    pub max_load_threads: usize,

    /// Number of threads to use when evaluating the model. 0 means one per logical core.
    #[arg(long, default_value_t = 0)]
    pub max_eval_threads: usize,

    /// Seed for random numbers. If unset will generate different results each time.
    #[arg(long, default_value = None)]
    pub seed: Option<u64>,
}

#[derive(Debug)]
pub struct ConfiguredSamplers {
    pub(crate) builder: SamplerChainBuilder,
    pub(crate) mirostat1: bool,
    pub(crate) mirostat2: bool,
    pub(crate) incompat_mirostat: bool,
}

impl Default for ConfiguredSamplers {
    fn default() -> Self {
        Self {
            builder: SamplerChainBuilder::from([
                (
                    "repetition",
                    SamplerSlot::new_chain(
                        || Box::new(SampleRepetition::default().penalty(1.30).last_n(64)),
                        [],
                    ),
                ),
                (
                    "freqpresence",
                    SamplerSlot::new_chain(
                        || Box::new(SampleFreqPresence::default().last_n(64)),
                        [],
                    ),
                ),
                (
                    "seqrepetition",
                    SamplerSlot::new_chain(|| Box::<SampleSeqRepetition>::default(), []),
                ),
                (
                    "topk",
                    SamplerSlot::new_single(
                        || Box::new(SampleTopK::default().k(40)),
                        Option::<SampleTopK>::None,
                    ),
                ),
                (
                    "tailfree",
                    SamplerSlot::new_single(
                        || Box::<SampleTailFree>::default(),
                        Option::<SampleTailFree>::None,
                    ),
                ),
                (
                    "locallytypical",
                    SamplerSlot::new_single(
                        || Box::<SampleLocallyTypical>::default(),
                        Option::<SampleLocallyTypical>::None,
                    ),
                ),
                (
                    "topp",
                    SamplerSlot::new_single(
                        || Box::new(SampleTopP::default().p(0.95)),
                        Option::<SampleTopP>::None,
                    ),
                ),
                (
                    "temperature",
                    SamplerSlot::new_single(
                        || Box::new(SampleTemperature::default().temperature(0.8)),
                        Option::<SampleTemperature>::None,
                    ),
                ),
                (
                    "mirostat1",
                    SamplerSlot::new_single(
                        || Box::<SampleMirostat1>::default(),
                        Option::<SampleMirostat1>::None,
                    ),
                ),
                (
                    "mirostat2",
                    SamplerSlot::new_single(
                        || Box::<SampleMirostat2>::default(),
                        Option::<SampleMirostat2>::None,
                    ),
                ),
            ]),
            mirostat1: false,
            mirostat2: false,
            incompat_mirostat: false,
        }
    }
}

impl ConfiguredSamplers {
    pub fn ensure_default_slots(&mut self) {
        self.builder.iter_mut().for_each(|(name, slot)| {
            let mirostat = self.mirostat1 || self.mirostat2;
            match name as &str {
                "temperature" | "repetition" => slot.ensure_present(),
                "topp" | "topk" if !mirostat => slot.ensure_present(),
                _ => (),
            }
        });

        if !(self.mirostat1 || self.mirostat2) {
            self.builder += (
                "randdistrib".to_string(),
                SamplerSlot::new_static(|| Box::<SampleRandDistrib>::default()),
            )
        }
    }

    pub fn ensure_valid(&self) -> Result<(), Box<dyn Error + Send + Sync>> {
        if self.mirostat1 && self.mirostat2 {
            Err(Box::<dyn Error + Send + Sync>::from(
                "Cannot enable both Mirostat 1 and Mirostat 2 samplers",
            ))?
        } else if (self.mirostat1 || self.mirostat2) && self.incompat_mirostat {
            Err(Box::<dyn Error + Send + Sync>::from(
                "Cannot enable top-p, top-k, locally typical or tail free samplers with Mirostat 1 or 2",
            ))?
        }
        Ok(())
    }
}

impl FromStr for ConfiguredSamplers {
    type Err = Box<dyn Error + Send + Sync + 'static>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut result = Self::default();

        let s = s.trim().to_lowercase();
        let opts = s
            .split(|c: char| c == '/' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .map(|s| {
                if let Some((name, opts)) = s.split_once(':') {
                    (
                        name.trim()
                            .chars()
                            .filter(|c| *c != '_' && *c != '-')
                            .collect(),
                        opts.trim(),
                    )
                } else {
                    (s.trim().to_string(), "")
                }
            })
            .inspect(|(name, _slot)| match name.as_str() {
                "mirostat1" => result.mirostat1 = true,
                "mirostat2" => result.mirostat2 = true,
                "topp" | "topk" | "locallytypical" | "tailfree" => result.incompat_mirostat = true,
                _ => (),
            })
            .collect::<Vec<_>>();

        opts.into_iter()
            .try_for_each(|(name, args)| result.builder.configure(name, args))?;

        result.ensure_default_slots();
        result.ensure_valid()?;

        Ok(result)
    }
}
