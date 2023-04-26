use clap::{Parser, ValueEnum};

/// Used as the prompt.
const DEFAULT_PROMPT: &str = "\nIn a shocking finding, scientists discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.";

/// Example of a small model to try.
const DEFAULT_MODEL: &str = "./RWKV-4-Pile-430M-20220808-8066.safetensors";

/// Tokenizer definition file. See README.
const DEFAULT_TOKENIZER: &str = "./20B_tokenizer.json";

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
    #[value(name = "ggmlq4_2")]
    /// GGML-backed 4 bit quantized, method 3.
    GGMLQ4_2,

    #[cfg(feature = "ggml")]
    #[value(name = "ggmlq4_3")]
    /// GGML-backed 4 bit quantized, method 4.
    GGMLQ4_3,
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

    /// The higher the temperature, the more random the results.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f32,

    /// a better explanation of top_p than I could ever write: https://huggingface.co/blog/how-to-generate
    #[arg(long, default_value_t = 0.85)]
    pub top_p: f32,

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
