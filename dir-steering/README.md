# Directional Steering in Strata

Directional steering is a runtime activation edit for Strata, adapting and extending the methodology created by Salvatore Sanfilippo ([`antirez/ds4`](https://github.com/antirez/ds4/tree/main/dir-steering)) to multi-GPU pipelined architectures and quantized Mixture-of-Experts models.

A steering file is either a flat binary `f32` matrix (DS4 native format) or a standard GGUF control vector container. During inference, Strata applies the projection edit to the residual stream:

```text
y = y - scale * direction[layer] * dot(direction[layer], y)
```

- **Positive scale**: Suppresses the represented direction.
- **Negative scale**: Amplifies the represented direction.
- **Zero scale or no file**: Follows the normal, unsteered inference path.

Unlike single-threaded synchronous engines, Strata runs across an asynchronous pipelined multi-GPU setup (e.g. Dual NVIDIA Pascal: GTX 1080 Ti + GTX 1070) with speculative decoding (MTP) and DP4A INT8 quantization. To support dynamic vector disarming without CUDA driver lock contention or pipeline stalls, Strata introduces **Transient Steering** via zero-copy pinned host memory.

---

## Model Shapes

A direction file contains one normalized hidden-width direction per normal transformer layer:

- **Qwen3.8 Flash Next**: `48 x 2560` (48 layers, 2560 hidden dimension, 4 hyper-connection residual streams).
- **Qwen3.8 Flash Next (Headless)**: `47 x 2560` (omitting the final layer).
- **DeepSeek V4 Flash**: `43 x 4096`.
- **GLM 5.3 Flash**: `45 x 4096`.

In Strata, raw `.f32` files, `.bin` files, and `.gguf` control vectors are all loaded transparently: the engine detects the file type automatically from the first bytes.

---

## Runtime Options

Strata supports both native DS4 naming and historical Strata/llama.cpp flags:

```text
--dir-steering-file FILE       Load steering vectors from .f32, .bin or .gguf
--dir-steering-ffn F           Scale multiplier (default: 1.0; >0 suppresses, <0 amplifies)
--dir-steering-tokens N        Apply steering for prompt + first N generated tokens only (Transient Steering)

# Strata native control-vector flags (the real engine interface):
--control-vector FILE          GGUF control vector, scale 1 (llama.cpp's format)
--control-vector-scaled FILE:SCALE  control vector with an explicit scale
--cvec-mode add|project        h += s v (default) or h -= s (h.v) v with v unit
--cvec-tokens N                Alias for --dir-steering-tokens (Transient Steering)

# Not implemented in Strata (DS4-only, for reference):
# --dir-steering-attn F        Strata applies the projection on the layer's folded
#                              residual write only (the FFN side); there is no
#                              attention-output steering hook in the CUDA graph.
```

### Transient Steering (`--dir-steering-tokens N`)

Continuous steering across long outputs can cause stylistic over-saturation, repetitive tokens, or grammatical drift after 20–40 tokens. 

**Transient steering** solves this by applying the directional vector only to the prompt and the first $N$ generated tokens (e.g. $N = 5$ or $12$):
1. The model is guided to establish the desired initial trajectory (for example, immediately emitting `</think>` to suppress verbose chain-of-thought, or choosing concise phrasing).
2. Once token $N$ is produced, the steering vector is deactivated on the fly.
3. The autoregressive loop completes the remaining answer naturally with full unconstrained linguistic capability.

---

## Multi-GPU Zero-Copy Architecture (Dual Pascal)

On a multi-GPU pipelined setup (e.g. GTX 1080 Ti + GTX 1070 with `--pipeline-windows 2`), disarming the steering vector mid-flight using traditional `cudaMemcpy` + `cudaDeviceSynchronize()` causes a fatal deadlock: the CPU host thread blocks waiting for the GPU, while the GPU pipeline is waiting for the next window's tokens from the host.

Strata eliminates this through **zero-copy mapped host memory**:

```cpp
// Host pinned allocation with portable PCIe mapping:
cudaHostAlloc((void**) &on_host, sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable);
cudaHostGetDevicePointer((void**) &on_dev, on_host, 0);
```

1. **Kernel Execution**: CUDA kernels read `*on_dev` directly over the PCIe bus without caching conflicts.
2. **Instant Deactivation**: When token count reaches `--dir-steering-tokens`, the host executes an atomic store:
   ```cpp
   *on_host = 0;  // Immediate CPU store (< 10 ns), zero driver locks, zero pipeline stalls
   ```

---

## Building Steering Vectors

Vectors are built from paired contrast prompts (e.g. concise vs verbose, direct vs thinking) using `tools/build_steering_vector.py`:

```sh
python3 tools/build_steering_vector.py \
  --profile qwen3.8-flash-next \
  --good-file dir-steering/examples/succinct.txt \
  --bad-file dir-steering/examples/verbose.txt \
  --out-f32 dir-steering/vectors/verbosity.f32 \
  --out-gguf dir-steering/vectors/verbosity.gguf \
  --method gram-schmidt
```

### Mathematical Methodology
1. **Contrast Extraction**: Difference of means between target and control activation states at the prompt boundary:
   $$\vec{d}_l = \frac{1}{N_{\text{pos}}} \sum_i \vec{h}_{l, \text{pos}}^{(i)} - \frac{1}{N_{\text{neg}}} \sum_j \vec{h}_{l, \text{neg}}^{(j)}$$
2. **Gram-Schmidt Orthogonalization**: The neutral centroid $\vec{\mu}_l$ is projected out to isolate the stylistic direction without shifting baseline token probabilities:
   $$\vec{v}_l = \vec{d}_l - \frac{\vec{d}_l \cdot \vec{\mu}_l}{\|\vec{\mu}_l\|^2} \vec{\mu}_l, \quad \hat{v}_l = \frac{\vec{v}_l}{\|\vec{v}_l\|}$$

---

## Format Conversion Tool (`tools/ds4_to_cvec.py`)

To convert between DS4 `.f32` flat matrices and GGUF `controlvector` files:

```sh
# Convert DS4 .f32 to GGUF:
python3 tools/ds4_to_cvec.py dir-steering/vectors/my-vector.f32 --to-gguf dir-steering/vectors/my-vector.gguf

# Convert GGUF back to DS4 .f32:
python3 tools/ds4_to_cvec.py dir-steering/vectors/my-vector.gguf --to-ds4 dir-steering/vectors/my-vector.f32

# Verify exact bit-for-bit parity:
python3 tools/ds4_to_cvec.py dir-steering/vectors/my-vector.f32 --to-gguf /tmp/check.gguf --check-roundtrip
```

---

## HTTP API Usage (OpenAI Compatible)

When running the Strata web/API server (`serve/server.py`), steering options can be passed per-request in `/v1/chat/completions`:

```json
{
  "model": "swift-1.5-iq2_xs",
  "messages": [
    {"role": "user", "content": "Write a one-sentence motto for deep space exploration."}
  ],
  "dir_steering_tokens": 5,
  "cvec": 1,
  "temperature": 0.0
}
```

- `"cvec": 0` disables steering for that specific request.
- `"dir_steering_tokens": N` limits steering to the first $N$ generated tokens.

---

## Measured Behavior (Dual Pascal GTX 1080 Ti + 1070)

Model: `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` (35B MoE, 32K context).  
Prompt: `"Write a one-sentence motto for deep space exploration."` (Greedy, T=0).

| Mode | Parameters | Generated Output | Result |
| :--- | :--- | :--- | :--- |
| **Baseline** | `cvec=0` | `<think>\nThe user wants a short motto...` | Standard lengthy CoT reasoning block. |
| **Full Steering** | `cvec=1, cvec_tokens=0` | `"Beyond the stars, we find ourselves."` | Immediate direct output; `<think>` suppressed. |
| **Transient (5 tokens)** | `cvec=1, cvec_tokens=5` | `<think>\n\n</think>\n\n"Beyond the horizon...` | Steered for 5 tokens to close `<think>`, then releases model into natural generation. |
| **Transient (12 tokens)** | `cvec=1, cvec_tokens=12` | `<think>\n\n</think>\n\n"Beyond the horizon, into the unknown."` | Stable trajectory maintained after disarm. |
| **Reset** | `cvec=0` | `<think>\nThe user wants a short motto...` | Bit-exact identity with baseline; zero residue in memory. |
