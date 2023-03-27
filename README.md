# Smol Rust RWKV

## What is it?

A very basic example of the RWKV approach to language models written in Rust by someone that
knows basically nothing about math or neural networks. Very, very heavily based on the
amazing information and Python example here: https://johanwind.github.io/2023/03/23/rwkv_details.html

Also see RWKV creator's repository: https://github.com/BlinkDL/ChatRWKV/

## How can I use it?

You'll need Rust set up. You'll probably want a Python environment activated with PyTorch and safetensors packages available.

You will need to download this file (about 820MB): https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/RWKV-4-Pile-430M-20220808-8066.pth

Also the tokenizer here: https://github.com/BlinkDL/ChatRWKV/blob/main/20B_tokenizer.json

The first step is to convert the `.pth` to SafeTensors format. Look at `pth_to_safetensors.py` for an example.

After that, you should just be able to `cargo run --release`.

