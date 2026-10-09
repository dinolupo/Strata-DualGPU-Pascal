#!/usr/bin/env python3
"""End-to-end testing of directional steering and transient steering in Strata.

Tests:
1. Baseline generation (no vector).
2. Directional steering via GGUF vector file (--dir-steering-file file.gguf --dir-steering-ffn 1.0).
3. Directional steering via DS4 .f32 binary file (--dir-steering-file file.f32 --dir-steering-ffn 1.0).
   -> Verifies exact output parity between GGUF and DS4 formats.
4. Directional steering with negative scale (--dir-steering-ffn -1.0).
   -> Verifies signed scale effect (amplification vs suppression).
5. Transient steering (--dir-steering-tokens 5).
   -> Verifies deactivation at token 5 and seamless continuation.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import strata_tokenizer as ST
from serve.server import StrataEngine, child_env, engine_args

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENGINE = ROOT / "build-vector-steering" / "strata"
PACK = pathlib.Path("/home/dino/proj/Strata-data/packs/swift-iq2_xs")
NATIVE_SHARD = pathlib.Path(
    "/home/dino/proj/genAI/models/Qwen3.8-Flash-Next-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf"
)
GGUF_VECTOR = ROOT / "data" / "experimental-speed-projection" / "Qwen3.8-Flash-Next-experimental-speed-projection.gguf"
F32_VECTOR = ROOT / "data" / "experimental-speed-projection" / "Qwen3.8-Flash-Next-steering.f32"


def runtime_config(context: int = 2048):
    path = pathlib.Path(os.environ.get("STRATA_STEERING_CONFIG", ROOT / "strata-swift-upstream-pascal.json"))
    cfg = json.loads(path.read_text(encoding="utf-8"))
    args = engine_args(cfg)
    for flag in ("--resident-budget-gib", "--vram-reserve-mib", "--vram-reserve-later-mib",
                 "--expert-profile", "--mtp", "--prefill", "--spec", "--pipeline-windows"):
        if args.count(flag) != 1:
            raise ValueError(f"Steering config must give exactly one {flag}: {path}")
    budget = float(args[args.index("--resident-budget-gib") + 1])
    if not 0 < budget <= 16:
        raise ValueError("Steering test RAM budget must be positive and at most 16 GiB")
    if "--resident-experts" not in args or cfg.get("env", {}).get("STRATA_PIN_GUARD") != "2":
        raise ValueError("Steering tests require resident experts and STRATA_PIN_GUARD=2")
    args[args.index("--max-context") + 1] = str(context)
    cfg["args"] = args
    print(f"[test] runtime config: {path}; context override: {context}; binary: {ENGINE}", flush=True)
    return cfg


def start_engine(extra_args: list[str], context: int = 2048):
    if not ENGINE.is_file():
        raise FileNotFoundError(f"Steering build missing; refusing fallback to a different binary: {ENGINE}")
    cfg = runtime_config(context)
    log_dir = ROOT / "logs" / "steering-tests"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"engine-{uuid.uuid4().hex}.log"
    args = engine_args(cfg) + extra_args
    print(f"[test] engine log: {log}", flush=True)
    print(f"[test] GPUs: {cfg['gpu']}; args: {args}", flush=True)
    engine = StrataEngine(str(ENGINE), args, cwd=str(ROOT), env=child_env(cfg), log=str(log))
    return engine, log


def get_tokenizer():
    tpath = PACK / "tokenizer"
    vocab = json.loads((tpath / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def run_engine(extra_args: list[str], prompt_tokens: list[int], max_new: int = 15) -> tuple[str, list[int], str]:
    engine, log = start_engine(extra_args)
    try:
        out_tokens = [token for token in engine.generate(prompt_tokens, max_new, {}, threading.Event())
                      if token is not None]
        if (not out_tokens or engine.last.get("finish") not in ("stop", "length") or
                engine.last.get("generated") != len(out_tokens) or len(out_tokens) > max_new):
            raise RuntimeError(f"Incomplete or invalid engine generation: {engine.last}")
        result = json.dumps(engine.last)
    finally:
        engine.close()
    return result, out_tokens, log.read_text(encoding="utf-8")


def main():
    print("=" * 70)
    print("STRATA DIRECTIONAL & TRANSIENT STEERING VERIFICATION SUITE")
    print("=" * 70)

    tok = get_tokenizer()
    prompt_text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWrite a short motto for space explorers.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    prompt_ids = tok.encode(prompt_text, parse_special=True)
    print(f"Prompt ({len(prompt_ids)} tokens):\n{prompt_text.strip()}\n")

    max_new = 15

    # Test 1: Baseline
    print("\n--- Test 1: Baseline (No Steering) ---")
    _, tokens_base, stderr_base = run_engine([], prompt_ids, max_new)
    text_base = tok.decode(tokens_base)
    print(f"Tokens ({len(tokens_base)}): {tokens_base}")
    print(f"Decoded: {text_base!r}")

    # Test 2: GGUF Vector (scale = 1.0)
    print("\n--- Test 2: Directional Steering via GGUF (scale = 1.0) ---")
    _, tokens_gguf, stderr_gguf = run_engine(
        ["--dir-steering-file", str(GGUF_VECTOR), "--dir-steering-ffn", "1.0"],
        prompt_ids,
        max_new,
    )
    text_gguf = tok.decode(tokens_gguf)
    print(f"Tokens ({len(tokens_gguf)}): {tokens_gguf}")
    print(f"Decoded: {text_gguf!r}")

    # Test 3: DS4 .f32 Vector (scale = 1.0)
    print("\n--- Test 3: Directional Steering via DS4 .f32 (scale = 1.0) ---")
    _, tokens_f32, stderr_f32 = run_engine(
        ["--dir-steering-file", str(F32_VECTOR), "--dir-steering-ffn", "1.0"],
        prompt_ids,
        max_new,
    )
    text_f32 = tok.decode(tokens_f32)
    print(f"Tokens ({len(tokens_f32)}): {tokens_f32}")
    print(f"Decoded: {text_f32!r}")

    # PARITY CHECK: GGUF vs DS4 .f32
    print("\nChecking Parity between GGUF and DS4 .f32:")
    if tokens_gguf == tokens_f32:
        print(">>> SUCCESS: 100% IDENTICAL TOKENS between GGUF and DS4 .f32! <<<")
    else:
        raise AssertionError(f"GGUF/F32 token mismatch: GGUF={tokens_gguf} vs F32={tokens_f32}")

    # Test 4: Directional Steering with Negative Scale (scale = -1.0)
    print("\n--- Test 4: Directional Steering with Negative Scale (scale = -1.0) ---")
    _, tokens_neg, stderr_neg = run_engine(
        ["--dir-steering-file", str(F32_VECTOR), "--dir-steering-ffn", "-1.0"],
        prompt_ids,
        max_new,
    )
    text_neg = tok.decode(tokens_neg)
    print(f"Tokens ({len(tokens_neg)}): {tokens_neg}")
    print(f"Decoded: {text_neg!r}")
    if tokens_neg != tokens_f32:
        print(">>> SUCCESS: Negative scale correctly alters output trajectory vs positive scale! <<<")
    else:
        raise AssertionError("Signed-scale output effect was not demonstrated on this prompt")

    # Test 5: Transient Steering (tokens = 5)
    print("\n--- Test 5: Transient Steering (--dir-steering-tokens 5) ---")
    _, tokens_trans, stderr_trans = run_engine(
        [
            "--dir-steering-file", str(F32_VECTOR),
            "--dir-steering-ffn", "1.0",
            "--dir-steering-tokens", "5",
        ],
        prompt_ids,
        max_new,
    )
    text_trans = tok.decode(tokens_trans)
    print(f"Tokens ({len(tokens_trans)}): {tokens_trans}")
    print(f"Decoded: {text_trans!r}")

    deactivated = "directional steering deactivated after" in stderr_trans
    print(f"Transient deactivation reported in stderr: {deactivated}")
    for l in stderr_trans.splitlines():
        if "directional steering" in l:
            print(f"Stderr notice: {l}")

    if deactivated:
        print(">>> SUCCESS: Transient steering turned off mid-stream as expected! <<<")
    else:
        raise AssertionError("Transient steering deactivation was not reported")
    if len(tokens_trans) <= 5:
        raise AssertionError("No generated tokens after the transient steering cutoff")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
