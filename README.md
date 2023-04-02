# Smol Rust RWKV

## What is it?

A simple example of the RWKV approach to language models written in Rust by someone that
knows basically nothing about math or neural networks. Very, very heavily based on the
amazing information and Python example here: https://johanwind.github.io/2023/03/23/rwkv_details.html

Also see the RWKV creator's repository: https://github.com/BlinkDL/ChatRWKV/

## Features

1. Written in Rust. Static typing can really help when trying to understand something, since it's clear what type of thing every object is.
2. Relatively clear/simple code.
3. Doesn't depend on massive frameworks like Torch or Cuda.
4. Can use all threads/cores for inference.
5. Supports float32 and 8bit inference.

Currently, the primary goal here isn't to create an application or library suitable for end users but instead just to provide a
clear example for other people who are aiming to implement RWKV.

## Shortcomings

1. Not optimized for performance.
2. Can only use 32bit or 8bit mode for models. (Models are always stored as full 32bit).
3. Can only run inference on CPU.

If loading in 32bit mode it uses a _lot_ of memory. The 3B model uses around 11GB RAM and the 7B one might _just_ fit on a 32GB machine
you're willing to close other applications or deal with some swapping. Even loading in 8bit mode uses a fair amount of memory, but
it will drop down once loading has completed.

## How can I use it?

You'll need Rust set up. You'll probably want a Python environment activated with PyTorch and safetensors packages available.

You will need to download this file (about 820MB): https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth

Also the tokenizer here: https://github.com/BlinkDL/ChatRWKV/blob/main/20B_tokenizer.json

The first step is to convert the `.pth` to SafeTensors format. Look at `pth_to_safetensors.py` for an example.

After that, you should just be able to `cargo run --release`. You can try compiling without `--release` but
it's likely everything will be insanely slow. Also try `cargo run --release -- --help` to see commandline options.

**Note**: The default is to use all logical cores, see the commandline options.

## How it works

Here is a (possibly wrong) high level description of the steps involved in evaluating the model.
You will need to refer to the source in `smolrwkv/src/simple/model.rs` for this to make sense.

Also, strongly consider reading these first:

1. https://johanwind.github.io/2023/03/23/rwkv_overview.html — High level explanation.
2. https://johanwind.github.io/2023/03/23/rwkv_details.html — More detailed explanation with a Python example.

By the way, fun fact: "Tensor" sounds real fancy but it's basically just an array. A one dimensional
tensor is just a one dimensional array, a two dimensional dimensional tensor is a two dimensional
array. They can have special properties (like being immutable) but that doesn't matter for understanding
the concept in general. If you know arrays, you have the general idea of tensors already.

To evaluate a token:

1. Calculate an initial value for `x` from `ln0`.
2. Feed this `x` to each layer sequentially, using the `x` the layer generated for the next one.
    1. Take `x` that got fed in.
    2. Apply `ln1` to `x` and feed it to time mixing. This uses tensor from the FFN part of the model.
       1. Take `tm_state` from the layer state and call it `last_x`. (Why? Who knows!)
       2. Take `tm_num` and `tm_den` as `last_num`, `last_den`.
       3. Do a bunch of fancy math stuff I'm not qualified to explain.
       4. The above calculated new values for `tm_[state,num,den]` so update your layer state with these.
       5. Also return `x` that resulted from the calculations.
    3. Add the `x` from time mixing to `x` (`x += time_mixing_x`).
    4. Apply `ln2` to `x` and feed it to channel mixing. This uses tensors from the feed forward network part of the model.
       1. Take `cm_state` from the layer state and call it `last_x`.
       2. More fancy math stuff (less involved than time mixing though).
       3. As with time mixing, this will calculate a new `cm_state` so update the layer state.
       4. Return `x` that resulted from the channel mixing calculation.
    5. Add the `x` from channel mixing to `x`.
3. Do fancy math stuff to the `x` that was the result after evaluating the last layer.
4. Return it as the list of probabilities for each token.

The model has a list of tokens it "knows". Sometimes a token is equal to a word, sometimes it's
just part of a word. There are usually a large number of tokens, in the range of 30,000-60,000.
I believe the current RWKV models have 50,277 tokens. Anyway, you'll get a list of 50,277 floating
point numbers back after running the model.

The highest value from that list is the token the model predicts is the most likely continuation and so on.
If you generated a sorted list of the top 10-40 or so token probabilities and select one randomly, you'll
get fairly reasonable output, relatively speaking. Fair to say a tiny 430M model doesn't produce the most
reasonable output in general.

Good explanation of how to handle the next step once you have the list of probabilities:
https://huggingface.co/blog/how-to-generate

## Trivia

There's various complicated math stuff involved in evaluating the model, but the only thing that really
matters is the matrix multiplication (`pardot` in the source). In the case of RWKV it's matrix-vector
multiplication (a 2D array multiplied with a 1D array). >90% of the time spent evaluating the model
is in those matrix multiplication calls.

The math/array handling here uses the `ndarray` crate. It provides a `.dot` function, however this
will _never_ actually calculate a matrix-vector multiplication in parallel even though the crate
claims threading support. Because this calculation is so critical for performance, I ended up writing
my own function to split the calculation into chunks and run it in parallel. See the functions in the
`dumdot` module in `smolrwkv/src/util.rs`.

The fact that you get a list of probabilities back and and no definite "answer" from the model
seems like a decent counterargument to the idea that LLMs are or could be conscious in some way.
When you look at output from an LLM, a lot of the time you aren't even going to be seeing the
most likely token. Also, fun fact: When you feed a prompt to a model, it comes up with a list
of probabilities just like when you're asking it for a response. However, those probabilities
are just thrown away except for the result after processing the very last prompt token.

## Example Output

Prompt in bold. So, are the dragons tree snakes or dogs? The world may never know!

***

```plaintext
* Loading tokenizer from: ./20B_tokenizer.json
* Loading model from: ./RWKV-4-Pile-430M-20220808-8066.safetensors
* Discovering model structure.
-   Loading layer 1/24
[...]
-   Loading layer 24/24
* Loading non-layer tensors.
* Loaded: layers=24, embed=1024, vocab=50277
```

**In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.**

These dragons all spoke different dialects and these dialects didn’t match the dogs' native language.

In an attempt to decipher what these dragons spoke, they called the dragons and found that their language was different from human.

"The Dragons understood human words and more precisely human languages. The dragons spoke the human language. They also understood the rules for Chinese,” the research team told Mongabay.

By conducting the research, they are hoping to shed light on the mysterious history of the dragons in the remote, remote regions of the world, especially in Tibet.

The research project, published in the journal Open Science, also shows that dragons are, in fact, reptiles, or a.k.a. tree snakes.

Dragon, not snake

According to the research team, the dragons found in Tibet are a race of dogs, not a reptile.

While the research team was still unable to come up with any explanation as to why these dragons live in Tibet, it was previously believed that they were most likely present on land near the Tibetan plateau.

"The dragons live there as part of the great Qinghai-Tibet Plateau that is almost completely undisturbed and the entire Qinghai-Tibet plateau was gradually converted to an agricultural state. Therefore, they have a distinctive pattern of chewing on the trees, and probably the animals are not too big to be kept in nature," the researchers explained.

