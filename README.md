# gguf-tensor-overrider

## Install

To install, run the following command:

```bash
curl https://raw.githubusercontent.com/k-koehler/gguf-tensor-overrider/refs/heads/main/install.sh | sudo /bin/bash
````

## Example Command

```bash
gguf-tensor-overrider -g https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF/resolve/main/UD-Q4_K_XL/Qwen3-235B-A22B-UD-Q4_K_XL-00001-of-00003.gguf -c 32000 --no-check --verbose
```

## Purpose

Tired of fucking around with `--tensor-override` regexes in llama.cpp? This tool aims to automatically allocate tensors optimally across your GPUs and CPUs.

## How it works

`gguf-tensor-overrider` does the following:

1. Downloads and extracts metadata for a model from Hugging Face, including the complete list of tensors
2. Iterates over each tensor and assigns it to GPU (if available) and then to CPU
3. Uses multiple passes trying to assign the most critical tensors to GPU before less critical ones. For example, in a MoE model, it will assign expert tensors last
4. Generates an output like `-ot "tensor_name_1=<device>" -ot "tensor_name_2=<device>"`

`gguf-tensor-overrider` uses the following priority in tensor allocation:

1. Attention tensors. It attempts to estimate KV cache size to ensure the device allocates these tensor bytes accurately
2. FFN tensors
3. Gate tensors
4. Norm tensors
5. Expert tensors and other tensors

## Options

1. `-g`, `--gguf-url`: The Hugging Face URL for the GGUF. In the case of multipart GGUFs, `gguf-tensor-overrider` automatically parses them if you provide the first file.
2. `-c`, `--context-length`: The context length you're passing to llama.cpp. Used to estimate the KV cache of the model to safely allocate attention tensors
3. `--context-quantization-size`: The quantization type of the KV cache. Currently assumes both K and V are quantized to the same type
4. `--check`, `--no-check`: Check if your system can handle the allocation without using swap
5. `--gpu-percentage` (default 0.9): How much of the GPU(s) to use for allocation. Useful if the script didn't allocate the cache accurately
6. `--granular-gpu-percentage`: Percentage for each GPU in your system. Useful if you don't want to use a certain GPU or llama.cpp compute buffer is making you sad
7. `--verbose`: Logs detailed information. Useful to see where things are being allocated

## How to pipe this into my llama command?

Here's an example of how you can pipe the arguments into your llama command:

```bash
#!/bin/bash

# Generate tensor overrides
TENSOR_OVERRIDES=$(gguf-tensor-overrider -g https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q8_0.gguf -c 32000)

# Build command with tensor overrides
CMD="/home/user/llama.cpp/build/bin/llama-cli \
  -m qwen3_32/qwen3_32b.gguf \
  -c 32000 \
  -fa \
  -sm row \
  $TENSOR_OVERRIDES"

# Execute command directly
eval "$CMD"
```

## Gotchas (for now)

- Only supports NVIDIA
- Only supports llama.cpp
- Only supports GGUF files from Hugging Face
- Only supports Qwen, Llama, Mistral, and DeepSeek architectures

## Can I use this code for xyz?

Go wild. The code in this repository is free, open source, modifiable, distributable, whatever-the-fuck-you-wantable.
