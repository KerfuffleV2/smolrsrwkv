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

Currently, the primary goal here isn't to create an application or library suitable for end users but instead just to provide a
clear example for other people who are aiming to implement RWKV.

## Shortcomings

1. Not optimized for performance.
2. Only can load models in 32bit mode — expect to require roughly twice as much memory as the model file. Models are stored as `bf16` and converted to `f32` when loaded.
3. Can (currently) only be configured by the `const` definitions in `src/main.rs` (prompt, model file name, etc).
4. Can only run inference on CPU.

Because of the second one, it uses a _lot_ of memory. The 3B model uses around 11GB RAM and the 7B one might _just_ fit on a 32GB machine
you're willing to close other applications or deal with some swapping.

## How can I use it?

You'll need Rust set up. You'll probably want a Python environment activated with PyTorch and safetensors packages available.

You will need to download this file (about 820MB): https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth

Also the tokenizer here: https://github.com/BlinkDL/ChatRWKV/blob/main/20B_tokenizer.json

The first step is to convert the `.pth` to SafeTensors format. Look at `pth_to_safetensors.py` for an example.

After that, you should just be able to `cargo run --release`.

**Note**: The default is to use all logical cores. If you don't want that, set the `RAYON_NUM_THREADS` environment variable.

