# Two GPUs and 32 GB of RAM

This is what the fork changes, why, and what each change measured. Everything was measured on one PC:

- RTX 5080 16 GB (drives two 4K monitors) and RTX 4060 Ti 16 GB (on a PCIe 4.0 x4 chipset slot)
- i9-14900KF (8 P-cores, 16 E-cores), 32 GB of RAM, Windows 11
- Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, 256K context, Q4_0 KV cache, `--spec 4`

The benchmark is the same every time: a fixed set of code and prose prompts, greedy, 500 tokens, plus a 32K-token
prompt. "Code" and "prose" below are decode speeds in tokens per second.

## The starting point

On a 32 GB PC the model's experts do not fit in RAM next to everything else, so Strata uses its low-RAM mode: the
GPU cache holds the most-used experts and the rest are copied into RAM once (`--resident-experts`). Upstream only
does that on one card. With a layer split it falls back to reading the experts through the OS file cache, which
on 32 GB means reading them from the SSD while it answers. So setup recommends one GPU, and the second card sits idle.

Two 16 GB cards hold about twice the experts one card holds. Getting them to work with the RAM copy was the first
step; the rest of the work was about the fact that, on a split, the cards take turns.

## 1. The resident RAM mode on a layer split

`--resident-experts` now works with `--layer-split`. Each stage's GPU cache is left out of the RAM copy, and what
remains is copied hottest first until the RAM budget runs out (upstream's #467 path, which already handles a copy
that does not fit). With 2 x 16 GB the caches hold about 13,000 experts and the RAM copy the other ~11,000
(15.5 GiB here), so nothing is read from the SSD while it answers.

One bug had to be fixed on the way: an adaptive swap copies the evicted expert from VRAM back into RAM, and with a
split it read the slot number of a later stage's cache out of CUDA0's cache. The RAM copy then held another
expert's weights and the output turned to garbage after the first few swaps. A short benchmark does not catch
this; a 30-request stress run does.

`--vram-reserve-later-mib` gives the later cards their own VRAM reserve. The card that drives the monitors needs
more headroom than the other one.

## 2. Card order: the faster card last

The last stage also runs the output head, the draft layer and (with pipelining, below) the draft chain that the
next window waits for. Putting the 5080 last and the 4060 Ti first was worth about 12% on its own (code 85-90 to
96-101 tok/s at the time). Upstream's docs recommend the opposite order.

The server now orders the cards by speed (SMs x clock, read from `strata --list-gpus`), fastest last, when the
config's `layer_split` is `auto`. `"gpu_order": "as_given"` keeps your order. An explicit split such as `"20"` also
keeps it, because the number was picked for that order.

## 3. The adaptive expert tier no longer stalls a window

Upstream's adaptive tier swaps experts between VRAM and RAM in rounds: pick the swaps, copy the evicted experts back
with a stream sync, copy the new ones in from pageable RAM. With ~90 swaps a round took ~24 ms, every 4th window.

`--adapt-async 1` (the default) splits a round in four steps that run on a job thread and advance between windows,
so no window waits for one. A slot is overwritten only after its old expert stopped being planned for the GPU, and
a RAM place only after its expert is resident. The same change made the CPU side of a window cheaper (an AVX2
activation quantizer, no host zeroing of rows the GPU fills anyway, no PCIe path recorded in a stage that never uses
one).

## 4. Fewer host waits per window

On a split the host used to synchronize stage 0 before launching stage 1, and paid for the commit and the draft
round one after the other. Now:

- the stages hand off through a flag in mapped pinned memory that the next stage's graph waits on, so both graphs
  are launched at once (a cross-GPU round trip through mapped memory measured 7-10 us here)
- commits are queued on each stage's stream and waited for once per request
- the MTP draft round is a single CUDA graph, with each chain step behind a conditional node that the GPU evaluates
  with the same min-p rule the host used
- the draft round is launched before the window's tokens are emitted

On the 0.1.30 base, sections 3 and 4 together with 15 pool workers (section 7) and a draft min-p of 0.65 took code
from 89.6 to 117.5 tok/s and prose from 70.3 to 86.3. Measured alone, the host-wait changes were worth code 95 to
108 and prose 75 to 85.

## 5. Dense kernels

A decode window is a long chain of small kernels, and on the 5080 many of them finish before the next one has its
weights. Programmatic dependent launch (sm_90 and newer) lets each kernel start fetching its weights into L2 while the
previous one finishes; the query side of attention runs on a side stream beside the K/V side; a Q4_0 KV cache, the
PLE post-ops and the split's hand-off copies are batched over the window. Every change keeps the arithmetic of the
path it replaces, checked with per-layer hashes over whole requests. Code 117.5 to 131.8 tok/s, prose 86.3 to 97.1
(with the hyper-connection read of the next paragraph included).

Upstream later shipped its own staged hyper-connection read (#315) and decode-once IQ kernels (#242), which did what
two parts of this work did, so the fork uses upstream's.

## 6. Pipelined windows: both cards at once

Even with all of the above, a profile of one window showed each card idle for more than half of it: the 4060 Ti ran
its 20 layers (~14 ms), then waited while the 5080 ran its 28 layers, the head and the draft (~12 ms), and the
other way round.

`--pipeline-windows 2` runs window K+1 on stage 0 while stage 1 still verifies window K. The tokens for K+1 come from
a teacher-forced MTP chain that starts as soon as K's first outputs land. If K accepts what the chain assumed, K+1
is already half done. If it does not, stage 0 restores its DeltaNet state from a snapshot taken on a side stream and
K+1 is rebuilt from the real tokens. A gate skips the speculative launch when the chain's own confidence says it is
unlikely to be kept.

Same server, serial vs pipelined:

| | code | prose | thinking (T=1.0) | copy-heavy edit |
|---|---:|---:|---:|---:|
| serial | 116.0 | 87.8 | 84.4 | 136.1 |
| pipelined | 136.1 | 96.0 | 89.2 | 177.3 |
| | x1.17 | x1.09 | x1.06 | x1.30 |

Edits gain the most because a prompt-lookup window (text copied from the context) chains straight into the next
speculative window.

It is on by default when the split runs on exactly two GPUs. `--pipeline-windows 0` turns it off. Requests with
repetition penalties or coupled drafting use the serial loop.

The automatic split (`--layer-split auto`) had to learn about it. Upstream's search adds the stages' times up, which
is right when they take turns: on this PC it put 2 of the 48 layers on the 4060 Ti and decoded code at 29 tok/s. With
pipelining a window costs about the slower stage, and a stage's CPU misses wait on that stage, so the search now
prices each stage by its layers, its misses and (on the last card) the head and the draft chain, and takes the
slower one. It picks K=19 here: 158 tok/s on code against 165 for the hand-tuned K=20.

## 7. CPU and OS defaults

- On a hybrid Intel CPU the expert pool now uses the P-cores (minus the one the host loop spins on) plus half of
  the E-cores. An E-core runs the expert kernels about 2.2x slower than a P-core and a layer waits for its slowest
  part. On the 14900KF that is 15 workers instead of 23, and with pipelining it made the biggest difference of any
  default: code 106 to 165 tok/s, prose 84 to 116. `--pool-workers N` overrides it.
- On Windows the engine opts out of power throttling (EcoQoS), which can run a background process's threads at
  efficiency clocks. `STRATA_NO_ECOQOS=0` leaves it to Windows.

## 8. VRAM reserves

When the config gives no reserve, the server reserves 1800 MiB on a card that drives a display and 512 MiB on one
that drives none. 1800 MiB is what two 4K monitors, a browser and the compositor needed here without the driver
running out of VRAM. With one 1080p monitor you can go lower with `--vram-reserve-mib` (first card) and
`--vram-reserve-later-mib` (the others). `"vram_reserve": "as_given"` in the config turns the automatic choice off.

## 9. Each card keeps only its own layers' weights

A split used to load the dense weights of all 48 layers on every card, although each card runs only its own layers.
Now each card keeps the weights of its layers, and the VRAM this frees goes to its expert cache. Upstream has the same
idea in two open pull requests, #559 (blange48) and #639 (JeanP00l), for explicit split points. Here it also works
with `--layer-split auto`: every card loads everything first, the search counts what trimming will free on each card
when it places the boundary, and then each card reloads only its own layers before the sessions and caches are made.

On this PC that frees 1.9 GiB on the 4060 Ti and 1.3 GiB on the 5080. Same engine, `--no-trim-stage-weights`
against the default, K=20:

| | code | prose | code, 32K context | prose, 32K context | 32K prompt |
|---|---:|---:|---:|---:|---:|
| IQ2_XS, every card holds every layer | 162 | 109 | 149 | 100 | 1,899 |
| IQ2_XS, own layers only | 172 | 111 | 157 | 104 | 2,201 |
| IQ3_XXS (160K context), every layer | 141 | 95 | 98 | 71 | 1,006 |
| IQ3_XXS, own layers only | 144 | 97 | 131 | 92 | 1,978 |

IQ3_XXS gains the most. Its experts are bigger, and before this change the ones no card held did not all fit in RAM,
so about 2,400 of them were read from the SSD in one benchmark run. With 2,174 more of them in VRAM the rest fit, and
none are read from the SSD. The automatic split still picks K=20 for it.

More experts in VRAM means more of them computed by the GPU, which rounds differently from the CPU (see
[Is the output still the same model?](#is-the-output-still-the-same-model)). On the 5,335 tokens of that comparison,
IQ3_XXS with and without trimming picked the same top token at 93.5% of the positions, with a perplexity of 6.95 and
7.03.

## 10. The CPU's experts

The experts no GPU cache holds are computed by the CPU pool. With IQ3_XXS that was the 5080 stage's largest wait,
about 5 ms of a 21 ms window. Three changes to the AVX-2 kernels:

- the IQ3_S, IQ3_XXS and IQ2_S grid entries are read with one gather on the P-cores (the E-cores gather slower than
  they insert, so they keep the old decode)
- on 12th-gen Intel and later, AVX-VNNI instructions do two multiply steps in one
- IQ3_S takes the multi-token kernel for a single token too: ggml's single-token dot was almost twice as slow on a
  P-core, and most experts the CPU computes in decode serve one token of the window

The first two give the same bits as before. One IQ3_S expert on a P-core went from 0.52 to 0.36 ms at three tokens
and from 0.39 to 0.25 ms at one. The engine gains less, mostly because the E-cores gain little: same engine, all three
against none, the 5080 stage's CPU time went from 5.3 to 4.8 ms per window, code from 135 to 139 tok/s and prose
from 96 to 98 (IQ3_XXS). IQ2_XS stays within the noise.

## Where it ended up

The stock draft head throughout, decode in tokens per second:

| | code | prose | 32K prompt |
|---|---:|---:|---:|
| 0.1.30 with sections 1 and 2 | 89.6 | 70.3 | |
| + sections 3, 4 and 7 | 117.5 | 86.3 | 1634 |
| + section 5 | 131.8 | 97.1 | 1618 |
| + section 6 | 145.3 | 101.6 | |
| all of it rebased on 0.1.38 (this repo) | 143.4 | 102.3 | 1940 |

## Is the output still the same model?

Not bit for bit, and it never was across settings: a token's CPU expert rows round differently when they are
computed alone or in a group, so the text depends on how the drafts cut it into windows, and an expert computed on
the CPU rounds differently from the same expert on the GPU. Anything that changes which experts sit in VRAM changes
the last bits, and at a near-tie that picks the other token. Upstream documents the same thing.

To check that nothing worse happened, I compared the next-token distributions of upstream 0.1.38 and this fork on
the same 5,333 tokens (a Python file, an English doc and an Italian text, teacher-forced through the verify windows
with `STRATA_LOGPOS`; upstream ran the same split, from the file cache). The end-of-turn token a chat model
expects inside a user message is taken out of each distribution, as in upstream's own comparison with llama.cpp.

| | upstream 0.1.38 | this fork | this fork, 1 GB smaller cache on the 5080 |
|---|---:|---:|---:|
| perplexity | 7.28 | 7.24 | 7.26 |
| top-1 = the real next token | 64.1% | 64.0% | 63.9% |

Upstream and the fork pick the same top token at 93.9% of the positions; the fork against itself with a smaller
cache, at 96.3%. Most of the disagreements are at near-ties.

With `STRATA_IQ_MT_MIN=1 --adapt-every 0` the output no longer depends on the window cuts. On one server (so the
same experts in VRAM) the pipelined loop then wrote the same text as the serial one in 45 of 45 requests: greedy,
seeded sampling, and runs with forced rollbacks.

## Settings and switches

| | |
|---|---|
| `--resident-experts` with `--layer-split` | the RAM copy across a split (section 1) |
| `--vram-reserve-later-mib N` | VRAM left free on the later cards of a split |
| `--pipeline-windows 0/1/2` | pipelined windows; 2 by default on a two-GPU split |
| `--layer-split auto` | balances the two stages when pipelined (section 6) |
| `--no-trim-stage-weights` | every card keeps the dense weights of all the layers (section 9) |
| `--adapt-async 0` | the blocking adaptive tier |
| `--list-gpus` | prints the visible cards and exits |
| `"gpu_order": "as_given"` | the server keeps the config's card order |
| `"vram_reserve": "as_given"` | the server adds no reserves |
| `STRATA_XSTAGE=0` | the host synchronizes each stage instead of the device flags |
| `STRATA_COMMIT_ASYNC=0` | wait for each commit |
| `STRATA_MTP_CHAIN=0` | one graph launch per draft step |
| `STRATA_DF_PDL=0` | no programmatic dependent launch |
| `STRATA_DF_BRANCH=1` | opt in to side streams inside the verify graph (off by default) |
| `STRATA_NO_ECOQOS=0` | leave Windows power throttling on |
| `STRATA_HOST_CPU=n` | pin the host loop to logical CPU n |
| `STRATA_IQ_GATHER=0` | no gathered i-quant decodes (section 10; `=1` forces them on every core) |
| `STRATA_NO_AVXVNNI=1` | the CPU kernels without AVX-VNNI |
| `STRATA_DECODE_TIMING=1` | per-window host and per-stage GPU timings in the engine log |
| `STRATA_PIPELINE_TRACE=<file>` | a timeline of the pipelined loop |

## Selective upstream reliability backports

The integration branch targets Linux/CUDA 12.9 on GTX 1080 Ti + GTX 1070 (sm_61).
It retains the fork's pipelined max-stage predictor, resident layer split, stage-weight
trimming on every GPU and CUDA 12.9 conditional-graph compatibility. It is **not**
a wholesale upgrade to upstream 0.1.40.3, and the measurements above are the original
fork author's hardware results, not Pascal validation results.

Upstream reference: `d5ea7133741e67743c0e886bb426c0ce8d69cf6c`.

| Reference | Integration |
|---|---|
| `5738d38` | Linux pinning guard: release source file cache when needed, touch the complement in RAM-budgeted steps, register an expert-aligned prefix, and distinguish registered memory from `cudaHostAlloc` ownership |
| `3fd0460`, `6c0fca3` | MTP catches up only through row `a`; native router requires its supported expert count and top-k |
| `bc089d5`, `2a8cded` | Sleeping stager waits on Linux, optional yield mode; Windows keeps yield mode |
| `7b80403` | Linux hybrid CPU detection from PMU lists, with the capacity-threshold fallback |
| `4dc544e` | BPE scan caches the read-only merge lookup locally; synthetic shared-tokenizer threading parity test |
| `73d3983` | A prefill layer that needs a full MMQ layout cannot run in undersized fused MoE buffers: report an error naming the layer instead of risking an illegal access or hang |
| `09e5ab4` | Automatic split candidates get a prompt-start budget gate, adapted to this fork's ring, loan and own-buffer fallback; a warning precedes ungated fallback |
| `e6ca0a0` | The fork already prices each trimmed range from its loaded arena and native matrices. The upstream late-loading/header estimator and RTX allocation-granule assumptions are not transplanted |
| `02be84a`, `2b87574`, `ec7f6a0`, `423f589` | Bounded engine/vision startup and image reads, restart/close lifecycle, and subsecond cancellation polling without increasing heartbeat frequency |

Adaptive pipeline fence errors and drain failures now propagate instead of being
treated as completed work. Failed or incomplete GPU probes keep the configured GPU
order and VRAM reserves, with a warning.

Additional upstream changes requiring absent paths are not imported: the VMM loader
is not present here; streamed K8V4 is explicitly unsupported in this fork, so the
new upstream batched K8V4 host-copy fix is not reachable. Intel/SYCL, AMD-specific
and experimental newer-architecture optimizations are outside this backport.

| Override | Meaning |
|---|---|
| `STRATA_PIN_GUARD=0/1/2` | Original bulk allocation / automatic Linux guard (default) / force paced registration |
| `STRATA_PIN_RESERVE_GIB=N` | Positive finite RAM reserve while touching pages; default is at least 2 GiB and the resident headroom |
| `STRATA_MTP_CATCHUP_ALL=1` | Restore whole-window MTP catch-up for A/B validation |
| `STRATA_STAGER_SLEEP=0` | Restore yield-based stager waits |
| `STRATA_ENGINE_READY_S=N` | Engine READY timeout, default 900 seconds |
| `STRATA_VISION_READY_S=N` | Vision READY timeout, default 300 seconds |
| `STRATA_VISION_ENCODE_S=N` | Image encoding read timeout, default 300 seconds |

Timeout overrides accept finite nonnegative seconds; zero waits indefinitely.
Invalid timeout and pinning overrides report an error rather than silently changing
the memory policy. Tests and build artifacts must be kept separate from the stable
engine and configuration; steering is not part of this integration.

### Build identity and unattended validation

`STRATA_BUILD_LABEL` is a CMake cache string for the engine's reported identity;
it defaults to the project's version. The candidate uses
`0.1.38-pascal-backport.d5ea713`, identifying the original base and the upstream
reference, not claiming full upstream 0.1.40.3 compatibility.

The Linux file-cache change `805088d` is not transplanted independently: this
fork's direct/unbuffered expert file tier is Windows-only. Its Linux path already
reads mappings through the page cache. The new upstream Linux direct-I/O path
would need a separate implementation and validation.

`reasoning_budget_tokens` already closes thinking with a wrap-up and continues
generation. The optional `reasoning_min_answer_tokens` (config or request,
nonnegative integer, default zero) rejects requests whose output cap cannot hold
the positive reasoning budget, wrap-up tokens and reserved answer tokens.
A request overrides the config. Zero preserves the existing behavior. The reserve
does not guarantee the model will complete an arbitrary answer; it checks available
space, and validation must still reject missing or truncated final answers.
It does not change Anthropic `thinking.budget_tokens`, which selects effort rather
than this hard budget.

The isolated model runner is `python -m tools.test_upstream_pascal`. It can test
token-exact synthetic prompts (`--prompt-tokens`), a known-answer gate
(`--known-answer`), reasoning (`--thinking --reasoning-budget N --min-answer N`),
conversation growth (`--conversation`), and a fresh-prefill comparison
(`--fresh-parity`). It reports numeric metadata and hashes, not prompt/answer text.
`--benchmark` is performance-only: an output cap there is not a passing
final-answer correctness test.

The sequential supervisor is `python -m tools.night_pascal --baseline PATH
--output-dir PATH`. Without `--execute` it only reports the matrix size and starts
no model. Execution requires a baseline containing an authorized absolute
`deadline_unix`. The supervisor bounds each worker, monitors memory and GPU
temperature, and cleans up only its own process group. Results and resource
samples are saved per case; existing cases are skipped on resume. Do not resume
an old result directory with a different binary/configuration and treat it as
one comparable run. Runtime failures, answer mismatches and incomplete cases
must remain visible even when another configuration succeeds.

Generate the evidence report with `python -m tools.night_report_pascal
--baseline PATH --output-dir PATH`. It reconciles the completed summary with
saved supervisor records: a started case without a completed summary is
`INCOMPLETE_EVIDENCE`, never a PASS. Monitor errors remain visible separately
from model failures. The report also checks protected-file hashes and requires
long-context, thinking, HTTP, fresh-prefill parity and 100-request soak gates
before writing a recommended configuration.

The 2026-10-08 bounded validation run completed 28 cases: 19 PASS, four FAIL,
four ERROR and one GPU-occupied case. The 16K case was interrupted with partial
evidence; 32 further cases were not run. The largest fully passing synthetic
prompt was 24,576 tokens with thinking. Soak, conversation parity, extreme
contexts and matched performance comparisons were not completed, so **no
configuration is qualified** and no production configuration was installed.
Subsequent boot-log inspection confirmed an unclean host restart at 04:02 CEST,
after the last saved test sample at 04:01:27. This interrupted the run, rather
than the seven-hour deadline. The underlying reset cause is not established;
no MCE/EDAC, kernel panic, OOM or NVIDIA Xid was found in the available journal.
Recorded GPU temperatures peaked at 70/59 C; CPU temperatures were not recorded.
Both GPU orders and pipeline 0 passed limited screening, not full reliability
qualification. In particular, serial mode is not yet a certified fallback.

For the user's 128K manual trial, `strata-swift-upstream-pascal.json` selects pipeline
0, physical GPU order `[1,0]`, split 22, resident budget 16 GiB, and reserves
512/1800 MiB, retaining the screened candidate's other settings. Thinking has
no forced budget (`reasoning_budget_tokens: 0`). The existing launcher uses
this config; no model is started by changing it.

In Chat settings, leave **Max tokens** empty to use the remaining context,
and keep thinking enabled. A browser-saved positive value still limits the
request. During the earlier 32K screening, the output allowance was
`32768 - prompt_tokens - 8`, including both thinking and the final answer.
Leaving Max tokens empty removes an optional output cap, not the
context limit, and does not guarantee an untruncated answer or host stability.
The current manual config increases `--max-context` to 131072 and adds
`--kv-resident 32768` to keep a bounded KV tier in VRAM and stream the remainder
from host memory. These settings are experimental on the dual-Pascal candidate:
the large-context matrix did not complete. Extra host memory, prefill latency
and streaming traffic must be expected; startup may fail if memory is insufficient.
The current output allowance is `131072 - prompt_tokens - 8`, not the 32K formula
used for the earlier screening.

An isolated Pi 1.0.4 profile lives at `../pi-strata-agent/`. It does not change
the user's existing Pi/llama.cpp setup. After manually starting Strata, launch:

```sh
PI_CODING_AGENT_DIR=/home/dino/proj/genAI/strata/pi-strata-agent pi
```

The profile connects via OpenAI-compatible `/v1/chat/completions` on port 8080,
not via a llama.cpp INI or router. Its model context metadata matches 131072.
Sampling parameters override Pi's output cap with Strata's supported
`max_completion_tokens: -1` (remaining context), and set no hard thinking budget.
The dummy API key is for the unauthenticated local endpoint only.
Automatic compaction reserves 32768 tokens and retains 20000 recent tokens,
giving a nominal trigger above 98304 tokens of context. Compaction is a
summarization request between turns, not an unlimited context or mid-generation
rescue. Endpoint/tool/reasoning interoperability has not been tested with this
profile; no inference was run while preparing it. The isolated profile has
separate sessions and does not load global Pi packages or credentials.

The same `strata-local` provider is also added to the user's standard Pi
`~/.pi/agent/models.json`, without changing its settings, packages, llama.cpp
INI, or default model. From any chosen working directory, select it with:

```sh
pi --provider strata-local --model qwen3.8-flash-next --thinking high
```

Both provider configurations omit the explicit `maxTokens` metadata field.
Pi supplies its internal default when that metadata is absent, but the model's
`samplingParams.max_completion_tokens: -1` overrides the outgoing request cap
in Pi 1.0.4. Strata therefore uses the remaining context; this does not remove
the 131072-token context boundary. The standard profile retains its existing
compaction settings/defaults, not the isolated profile's custom settings.
Pi still loads context instructions from its working directory and ancestors:
using the standard profile does not suppress this project's AGENTS.md if Pi
is launched inside the project tree.

### Headless dual-Pascal performance trial (2026-10-08)

The separate [performance config](../strata-swift-pascal-performance.json) and
[launcher](../run-swift-iq2_xs-performance.sh) leave the existing 128K config,
launcher and binary unchanged. After moving the desktop to the integrated GPU,
`nvidia-smi` reported 13 MiB used on the 1080 Ti and 3 MiB on the 1070,
with 11143/8105 MiB free respectively; host RAM available was 27472 MiB.
These are idle readings, not post-load memory guarantees or performance results.

The experimental profile retains physical GPU order `[1,0]`, 131072 context,
INT8 KV with 32768 resident cells, MTP spec 4, four expert-pool workers and
asynchronous adaptation. It changes the split to `auto` so the engine prices
both stages with current memory and pipelining enabled. Pipeline windows are 2;
the base VRAM reserves are 512/512 MiB. The pipeline allocation allowance stays
enabled (`STRATA_PIPELINE_RESERVE=1`), adding 224/160 MiB: the reserves used for
cache sizing are therefore 736/672 MiB, not 512/512 MiB. These are sizing
allowances, not guaranteed free memory after every runtime allocation.

The expert RAM budget is at most 16 GiB. The engine clamps it
to available RAM after its allocations and preserves its default 4 GiB host
headroom. Paced page-locking (`STRATA_PIN_GUARD=2`) stays enabled. Do not disable
these protections to chase cache capacity. Thinking remains unbudgeted and
there is no new output cap; positive client-side limits still apply.

The initial 20 GiB budget and 256 MiB first-stage reserve were too aggressive
in the user's manual trial: system swap usage rose from about 1.6 to 4.7 GiB,
sampled swap reads were 0.7-11.7 MiB/s, and system I/O wait was 54-62%.
Applications including the terminal and desktop had swapped pages; the 1070
had only 35 MiB free at a later reading. The user stopped that instance.
The revised 16 GiB budget and 512 MiB first-stage reserve restore headroom;
their effect on runtime memory, responsiveness and speed remains unmeasured.
After stopping, no GPU compute process remained and host available RAM was
29182 MiB. Swap may remain occupied by inactive application pages after exit;
occupied swap alone does not demonstrate ongoing swap traffic.

Stop the other Strata instance yourself before launching this profile:

```sh
./run-swift-iq2_xs-performance.sh
```

It uses the same local port 8080 and model identity, so the existing Pi provider
does not need changes. A running Pi session should be restarted after switching
the server. To revert, stop the performance instance and use the original launcher.
No setup or rebuild is required.

Only static config/argument checks, shell syntax and CPU-only split-plan tests
were run when preparing this profile. No model startup, benchmark or inference
was run. It is an aggressive candidate, not a measured fastest configuration or
a reliability-qualified one. Previous pipeline stalls and the unexplained host
reset remain relevant. The source still contains a warning about split KV
streaming support; the 128K streaming/pipeline combination is not qualified.
An answer at short context does not establish correctness or stability at 128K.
Do not run an unattended stress test with this profile.
