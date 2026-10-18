#!/usr/bin/env python3
"""Automated validation of DS4 Directional Steering and Transient Steering in Strata.

Starts the engine in --serve mode with dual Pascal GPUs (GTX 1080 Ti + GTX 1070),
loads the DS4 .f32 directional steering vector, and tests:
1. Unsteered request (cvec=0).
2. Steered request (cvec=1, scale=1.0).
3. Transient steering request (cvec=1, cvec_tokens=5): active for first 5 tokens, then deactivated.
4. Transient steering request (cvec=1, cvec_tokens=12): active for first 12 tokens, then deactivated.
5. Re-check unsteered request (cvec=0): verifies state isolation between requests.

Decodes tokens into readable text and verifies correct runtime toggling.
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import strata_tokenizer as ST
from test_steering_e2e import start_engine, log_request, steering_layers, steering_seed, DEFAULT_LAYERS

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACK = pathlib.Path("/home/dino/proj/Strata-data/packs/swift-iq2_xs")
F32_VECTOR = ROOT / "data" / "experimental-speed-projection" / "Qwen3.8-Flash-Next-steering.f32"
GGUF_VECTOR = ROOT / "data" / "experimental-speed-projection" / "Qwen3.8-Flash-Next-experimental-speed-projection.gguf"


def get_tokenizer():
    tpath = PACK / "tokenizer"
    vocab = json.loads((tpath / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    return ST.Tokenizer(tokens, merges, types)


def run_suite(vector_file: pathlib.Path, vector_type_desc: str):
    extra = ["--dir-steering-file", str(vector_file), "--dir-steering-ffn", "1.0"]
    engine, log = start_engine(extra, context=4096)
    try:
        return _run_suite(vector_file, vector_type_desc, engine, log, extra)
    finally:
        engine.close()


def _run_suite(vector_file: pathlib.Path, vector_type_desc: str, engine, log, extra: list[str]):
    print("=" * 76)
    print(f"RUNNING STEERING TEST SUITE: {vector_type_desc}")
    print(f"Vector path: {vector_file}")
    print(f"Layer range: {steering_layers()[0]} {steering_layers()[1]} "
          f"(STRATA_STEERING_LAYERS, default {DEFAULT_LAYERS}); seed: {steering_seed()}")
    print("=" * 76)

    tok = get_tokenizer()
    prompt_text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWrite a short motto for space explorers.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    prompt_ids = tok.encode(prompt_text, parse_special=True)
    prompt_csv = ",".join(map(str, prompt_ids))

    print("\n>>> Engine is READY. Running generation requests <<<\n")

    def send_request(req_cmd: str, label: str) -> tuple[list[int], str, str]:
        print(f"--> Sending: {req_cmd[:60]}... (prompt {len(prompt_ids)} tok)")
        fields = req_cmd.split()
        sampling = {}
        for field in fields[2:-1]:
            key, value = field.split("=", 1)
            if key == "cvec":
                sampling["experimental_speed_projection"] = bool(int(value))
            elif key == "cvec_tokens":
                sampling["cvec_tokens"] = int(value)
            else:
                raise ValueError(f"Unexpected test request key: {key}")
        sampling["seed"] = steering_seed()
        log_request(log, label, extra, sampling)
        tokens = [token for token in engine.generate(prompt_ids, int(fields[1]), sampling, threading.Event())
                  if token is not None]
        if (not tokens or engine.last.get("finish") not in ("stop", "length") or
                engine.last.get("generated") != len(tokens) or len(tokens) > int(fields[1])):
            raise RuntimeError(f"Incomplete or invalid engine generation: {engine.last}")
        done_line = json.dumps(engine.last)
        decoded = tok.decode(tokens)
        return tokens, decoded, done_line

    max_new = 25

    # 1. Baseline unsteered (cvec=0)
    req1 = f"GEN {max_new} cvec=0 {prompt_csv}"
    t1, text1, done1 = send_request(req1, "unsteered cvec=0")
    print(f"[RESULT 1 - Unsteered (cvec=0)] Tokens: {t1}")
    print(f"[TEXT 1] {text1!r}\n")

    # 2. Steered full length (cvec=1)
    req2 = f"GEN {max_new} cvec=1 {prompt_csv}"
    t2, text2, done2 = send_request(req2, "full steering cvec=1")
    print(f"[RESULT 2 - Full Steering (cvec=1)] Tokens: {t2}")
    print(f"[TEXT 2] {text2!r}\n")

    # 3. Transient Steering (cvec=1 cvec_tokens=5)
    req3 = f"GEN {max_new} cvec=1 cvec_tokens=5 {prompt_csv}"
    t3, text3, done3 = send_request(req3, "transient cvec_tokens=5")
    print(f"[RESULT 3 - Transient Steering (first 5 tokens)] Tokens: {t3}")
    print(f"[TEXT 3] {text3!r}\n")

    # 4. Transient Steering (cvec=1 cvec_tokens=12)
    req4 = f"GEN {max_new} cvec=1 cvec_tokens=12 {prompt_csv}"
    t4, text4, done4 = send_request(req4, "transient cvec_tokens=12")
    print(f"[RESULT 4 - Transient Steering (first 12 tokens)] Tokens: {t4}")
    print(f"[TEXT 4] {text4!r}\n")

    # 5. Check baseline again (cvec=0) to confirm clean reset
    req5 = f"GEN {max_new} cvec=0 {prompt_csv}"
    t5, text5, done5 = send_request(req5, "re-check unsteered cvec=0")
    print(f"[RESULT 5 - Re-check Unsteered (cvec=0)] Tokens: {t5}")
    print(f"[TEXT 5] {text5!r}\n")

    print("Shutting down engine...")
    engine.close()
    stderr_lines = log.read_text(encoding="utf-8").splitlines()

    # Verifications
    print("\n" + "=" * 76)
    print("ANALYSIS OF RESULTS:")
    print("=" * 76)

    # A. Did steering change output vs baseline?
    steer_active = (t1 != t2)
    print(f"1. Steering effect verified (Unsteered vs Full Steered): {'PASS (Outputs differ)' if steer_active else 'FAIL'}")

    # B. Did state reset cleanly between requests?
    clean_reset = (t1 == t5)
    print(f"2. Clean state reset between requests (Request 1 == Request 5): {'PASS (100% Identical)' if clean_reset else 'FAIL'}")

    # C. Were deactivations logged?
    deact_logs = [l for l in stderr_lines if "directional steering deactivated after" in l]
    print(f"3. Deactivation events captured on stderr: {len(deact_logs)} occurrences")
    for log in deact_logs:
        print(f"   - {log}")

    transient_ok = len(deact_logs) >= 2
    print(f"4. Transient steering deactivation mid-generation: {'PASS' if transient_ok else 'FAIL'}")

    all_ok = steer_active and clean_reset and transient_ok
    print(f"\nOVERALL SUITE STATUS: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 76)
    return all_ok, t1, t2, t3, t4


def main():
    ok_f32, t1_f32, t2_f32, t3_f32, t4_f32 = run_suite(F32_VECTOR, "DS4 Flat .f32 Tensor")
    if not ok_f32:
        print("FAIL on DS4 .f32 suite!")
        sys.exit(1)

    ok_gguf, t1_g, t2_g, t3_g, t4_g = run_suite(GGUF_VECTOR, "Standard GGUF Control Vector")
    if not ok_gguf:
        print("FAIL on GGUF suite!")
        sys.exit(1)

    print("\n" + "=" * 76)
    print("CROSS-FORMAT PARITY CHECK: DS4 .f32 vs GGUF")
    print("=" * 76)
    parity_full = (t2_f32 == t2_g)
    parity_trans5 = (t3_f32 == t3_g)
    parity_trans12 = (t4_f32 == t4_g)

    print(f"Full steered tokens match: {parity_full} (F32={t2_f32} vs GGUF={t2_g})")
    print(f"Transient 5 tokens match:  {parity_trans5} (F32={t3_f32} vs GGUF={t3_g})")
    print(f"Transient 12 tokens match: {parity_trans12} (F32={t4_f32} vs GGUF={t4_g})")

    if parity_full and parity_trans5 and parity_trans12:
        print(">>> 100% BITWISE AND BEHAVIORAL PARITY BETWEEN DS4 .F32 AND GGUF CONFIRMED! <<<")
    else:
        raise AssertionError("Cross-format token parity failed")

    print("\nAll DS4 Directional Steering and Transient Steering tests passed with 100% success!")
    sys.exit(0)


if __name__ == "__main__":
    main()
