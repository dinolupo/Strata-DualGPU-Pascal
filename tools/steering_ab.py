#!/usr/bin/env python3
"""Behavioral A/B + long-context qualification for vector steering.

Checks, without asserting any particular style:
1. Baseline (cvec=0) vs steered (.f32, ffn=-1) vs steered (ffn=+2): outputs differ.
2. Long context (128K): steering on, transient deactivation after N tokens:
   generation completes, deactivation event in the engine log, wall-clock in budget.
3. Clean reset: after deactivation, a cvec=0 request reproduces the baseline bitwise.

Outputs are logged to logs/steering-ab-outputs.log; stdout prints only lengths/verdicts.
"""
from __future__ import annotations

import pathlib
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from test_steering_e2e import start_engine, get_tokenizer, log_request, steering_layers, steering_seed, DEFAULT_LAYERS  # noqa: E402

F32 = ROOT / "data" / "experimental-speed-projection" / "Qwen3.8-Flash-Next-steering.f32"
OUT = ROOT / "logs" / "steering-ab-outputs.log"

PROMPTS = [
    "Explain why databases use indexes.",
    "What does DNS do?",
    "Give three tips for writing unit tests.",
]

IM_START = "<|" + "im_start" + "|>"
IM_END = "<|" + "im_end" + "|>"

def chat_prompt(text):
    return (IM_START + "system\nYou are a helpful assistant." + IM_END + "\n"
            + IM_START + "user\n" + text + IM_END + "\n"
            + IM_START + "assistant\n")

def run(engine, ids, max_new, sampling, log, extra, label):
    sampling = {**sampling, "seed": steering_seed()}
    log_request(log, label, extra, sampling)
    cancel = threading.Event()
    toks = [t for t in engine.generate(ids, max_new, sampling, cancel) if t is not None]
    last = engine.last
    if not toks or last.get("finish") not in ("stop", "length") or last.get("generated") != len(toks):
        raise RuntimeError(f"incomplete generation: {last}")
    return toks

BASELINE = {"experimental_speed_projection": False}  # -> cvec=0 on the GEN line

def main():
    tok = get_tokenizer()
    a, b = steering_layers()
    print(f"layer range: {a} {b} (STRATA_STEERING_LAYERS, default {DEFAULT_LAYERS}); seed: {steering_seed()}")
    ids = [tok.encode(chat_prompt(p), parse_special=True) for p in PROMPTS]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("")
    fails = []

    def log(label, toks):
        OUT.write_text(OUT.read_text() + f"\n===== {label} ({len(toks)} tok)\n"
                       + tok.decode(toks) + "\n")

    # ---- 1. short-context A/B: scale -1 engine, baseline via per-request cvec=0 ----
    extra1 = ["--dir-steering-file", str(F32), "--dir-steering-ffn", "-1"]
    e1, log1 = start_engine(extra1)
    base = []
    steered_m1 = []
    try:
        for i, p_ids in enumerate(ids):
            b = run(e1, p_ids, 60, BASELINE, log1, extra1, f"baseline prompt {i+1}")
            s = run(e1, p_ids, 60, {}, log1, extra1, f"steered ffn=-1 prompt {i+1}")
            base.append(b); steered_m1.append(s)
            log(f"baseline prompt {i+1}", b)
            log(f"steered ffn=-1 prompt {i+1}", s)
            if b == s:
                fails.append(f"A/B prompt {i+1}: steered(-1) identical to baseline")
    finally:
        e1.close()

    # ---- 2. scale +2 engine: outputs must differ from the same baseline ----
    extra2 = ["--dir-steering-file", str(F32), "--dir-steering-ffn", "2"]
    e2, log2 = start_engine(extra2)
    try:
        for i, p_ids in enumerate(ids):
            s = run(e2, p_ids, 60, {}, log2, extra2, f"steered ffn=+2 prompt {i+1}")
            log(f"steered ffn=+2 prompt {i+1}", s)
            if base[i] == s:
                fails.append(f"A/B prompt {i+1}: steered(+2) identical to baseline")
            if steered_m1[i] == s:
                fails.append(f"A/B prompt {i+1}: steered(-1) == steered(+2) (scale ignored?)")
    finally:
        e2.close()

    # ---- 3. long context 128K, transient deactivation after 20 tokens ----
    extra3 = ["--dir-steering-file", str(F32), "--dir-steering-ffn", "-1"]
    e3, log3 = start_engine(extra3, context=131072)
    try:
        t0 = time.time()
        s = run(e3, ids[0], 200, {"cvec_tokens": 20}, log3, extra3, "long-context transient-20")
        dt = time.time() - t0
        log(f"long-context 128K transient-20 prompt 1 ({dt:.1f}s)", s)
        eng_log = log3.read_text(encoding="utf-8", errors="replace")
        if "deactivated after 20 tokens" not in eng_log:
            fails.append("long-context: transient deactivation event missing from engine log")
        if dt > 600:
            fails.append(f"long-context: {dt:.1f}s > 600s budget (possible pipeline stall)")
        # clean reset: same prompt, cvec=0, must match the short-context baseline bitwise
        b2 = run(e3, ids[0], 60, BASELINE, log3, extra3, "clean-reset baseline after transient")
        if b2 != base[0]:
            fails.append("clean reset: post-deactivation baseline differs from first baseline")
    finally:
        e3.close()

    print("=" * 60)
    for f in fails:
        print("FAIL:", f)
    if not fails:
        print("ALL CHECKS PASSED: A/B differs, transient deactivation, clean reset, no stall")
    print(f"outputs: {OUT}")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
