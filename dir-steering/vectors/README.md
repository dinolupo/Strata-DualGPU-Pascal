# Steering Vectors Registry

This directory contains pre-computed steering matrices for supported models.

## Pre-computed Vectors

### 1. `Qwen3.8-Flash-Next-steering.f32` (DS4 Raw Format)
- **Model**: Qwen3.8-Flash-Next (and fine-tuned variants: Swift 1.5, Unsloth, Coder).
- **Shape**: `[48, 2560]` (48 transformer layers $\times$ 2560 hidden dimension).
- **Format**: Flat binary 32-bit floating point (`float32`).
- **File size**: $48 \times 2560 \times 4 = 491.520\text{ bytes}$.
- **Sidecar metadata**: `Qwen3.8-Flash-Next-steering.json`.
- **Purpose**: Direct answer / latency reduction / CoT suppression. When applied with positive scale ($F = 1.0$) and transient steering ($N = 5$ tokens), it accelerates output and suppresses redundant `<think>` preambles.

### 2. `Qwen3.8-Flash-Next-experimental-speed-projection.gguf` (GGUF Format)
- **Model**: Qwen3.8-Flash-Next.
- **Architecture**: `general.architecture = 'controlvector'`.
- **Tensors**: `direction.0` through `direction.47` (each of shape `[2560]`).
- **File size**: $483.520\text{ bytes}$.
- **Sidecar metadata**: `Qwen3.8-Flash-Next-experimental-speed-projection.json`.
- **Equivalence**: 100% bit-for-bit identical to the `.f32` vector via `tools/ds4_to_cvec.py`.

## Usage Quick Reference

```bash
# Load directly as .f32 in Strata CLI:
./engine/strata --pack <pack_path> --native <gguf_path> \
    --dir-steering-file dir-steering/vectors/Qwen3.8-Flash-Next-steering.f32 \
    --dir-steering-tokens 5 \
    --dir-steering-ffn 1.0

# Or load as .gguf in Strata Server:
# (config.json: "args": ["--cvec", "dir-steering/vectors/Qwen3.8-Flash-Next-experimental-speed-projection.gguf", "--cvec-tokens", "5"])
```
