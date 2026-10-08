"""Bounded local-model reliability checks; never print prompts or generated text."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

from serve import server
from serve.frontend import ChatTemplate
import strata_tokenizer


class MeasuredEngine(server.StrataEngine):
    calls = 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls += 1
        yield from super().generate(ids, max_new, sampling, cancel, embeddings=embeddings)


def replace_arg(args: list[str], flag: str, value: str) -> None:
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend((flag, value))


def tokenizer(path: Path):
    vocab = json.loads((path / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    return strata_tokenizer.Tokenizer(tokens, (path / "merges.txt").read_text().split("\n"),
                                      json.loads((path / "token_type.json").read_text()))


def sized_prompt(tok, template, target: int, expected: str, position: int, thinking: bool, *, message_only=False):
    marker = "STRATA_FILLER_BOUNDARY"
    rendered = template.render([{"role": "user", "content":
        "Read the archive notes below. Only the blue archive identifier matters.\n" + marker +
        "\nReturn only the identifier of the blue archive, not other numbers."}], enable_thinking=thinking)
    before, after = rendered.split(marker)
    prefix = tok.encode(before, parse_special=True)
    suffix = tok.encode(after, parse_special=True)
    fact = tok.encode(f"\nThe identifier of the blue archive is {expected}.\n")
    room = target - len(prefix) - len(suffix) - len(fact)
    if room < 0:
        raise ValueError("target prompt is smaller than its template and known fact")
    notes = tok.encode("The green folder contains cedar records. The red folder contains stone records. "
                       "The gray folder contains river records. Review the labels carefully.\n")
    filler = (notes * (room // len(notes) + 1))[:room]
    offset = (0, room // 2, room)[position]
    prompt = prefix + filler[:offset] + fact + filler[offset:] + suffix
    assert len(prompt) == target
    if message_only:
        return ("Read the archive notes below. Only the blue archive identifier matters.\n" +
                tok.decode(filler[:offset] + fact + filler[offset:]) +
                "\nReturn only the identifier of the blue archive, not other numbers.")
    return prompt


def check_final(done, text, expected, max_new):
    if done is None or done["finish"] != "stop" or not text.strip():
        raise AssertionError("missing complete final answer")
    if expected is not None and re.findall(r"\d+", text) != [expected]:
        raise AssertionError("known answer mismatch")
    if done["completion_tokens"] > max_new:
        raise AssertionError("completion limit exceeded across reasoning passes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--pipeline", type=int, choices=(0, 1, 2), default=2)
    ap.add_argument("--adapt", choices=("off", "blocking", "async"), default="off")
    ap.add_argument("--requests", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--context", type=int)
    ap.add_argument("--split", type=int)
    ap.add_argument("--gpu-order", choices=("0,1", "1,0"))
    ap.add_argument("--resident-budget-gib", type=float)
    ap.add_argument("--vram-reserve-mib", type=int)
    ap.add_argument("--vram-reserve-later-mib", type=int)
    ap.add_argument("--catchup-all", action="store_true")
    ap.add_argument("--paced-pin", action="store_true")
    ap.add_argument("--cancel", action="store_true")
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--long-prompt", action="store_true")
    ap.add_argument("--vary", action="store_true")
    ap.add_argument("--branches", choices=("default", "off", "on"), default="off")
    ap.add_argument("--save-profile", action="store_true")
    ap.add_argument("--prompt-tokens", type=int)
    ap.add_argument("--known-answer", action="store_true")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--reasoning-budget", type=int, default=1024)
    ap.add_argument("--min-answer", type=int, default=256)
    ap.add_argument("--kv-resident", type=int)
    ap.add_argument("--spec", type=int)
    ap.add_argument("--stager-sleep", choices=("0", "1"))
    ap.add_argument("--prefill", type=int)
    ap.add_argument("--repeat-long", action="store_true")
    ap.add_argument("--conversation", action="store_true")
    ap.add_argument("--fresh-parity", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--reasoning-problem", action="store_true")
    a = ap.parse_args()
    if a.requests < 1 or a.max_new < 1 or (a.context is not None and a.context < 512):
        ap.error("requests/max-new must be positive and context must be at least 512")
    if a.prompt_tokens is not None and a.prompt_tokens < 128:
        ap.error("prompt-tokens must be at least 128")
    if a.reasoning_budget < 1 or a.min_answer < 1:
        ap.error("reasoning-budget and min-answer must be positive")
    if a.resident_budget_gib is not None and (not math.isfinite(a.resident_budget_gib) or a.resident_budget_gib <= 0):
        ap.error("resident budget must be positive and finite")
    if a.split is not None and a.split < 2:
        ap.error("split must leave at least two layers on the first stage")
    if (a.vram_reserve_mib is None) != (a.vram_reserve_later_mib is None):
        ap.error("both VRAM reserves must be supplied together")
    if a.vram_reserve_mib is not None and (a.vram_reserve_mib < 0 or a.vram_reserve_later_mib < 0):
        ap.error("VRAM reserves must be nonnegative")
    cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
    cfg.pop("expert_profile_save", None)
    args = list(cfg["args"])
    for flag in ("--expert-profile-save", "--expert-profile-save-every"):
        while flag in args:
            i = args.index(flag)
            del args[i:i + 2]
    cfg["args"] = args
    cfg["exe"] = str(a.exe.resolve())
    cfg["cwd"] = str(server.ROOT)
    cfg["log"] = str(a.output.with_suffix(".engine.log").resolve())
    if a.gpu_order is not None:
        cfg["gpu"] = [int(index) for index in a.gpu_order.split(",")]
        cfg["gpu_order"] = "as_given"
    cfg["env"] = dict(cfg.get("env", {}))
    if a.branches == "default":
        cfg["env"].pop("STRATA_DF_BRANCH", None)
    else:
        cfg["env"]["STRATA_DF_BRANCH"] = "1" if a.branches == "on" else "0"
    cfg["env"]["STRATA_IQ_MT_MIN"] = "1"
    cfg["env"]["STRATA_MTP_CATCHUP_ALL"] = "1" if a.catchup_all else "0"
    if a.paced_pin:
        cfg["env"]["STRATA_PIN_GUARD"] = "2"
    if a.stager_sleep is not None:
        cfg["env"]["STRATA_STAGER_SLEEP"] = a.stager_sleep
    replace_arg(args, "--pipeline-windows", str(a.pipeline))
    replace_arg(args, "--adapt-every", "0" if a.adapt == "off" else "4")
    replace_arg(args, "--adapt-async", "1" if a.adapt == "async" else "0")
    if a.context is not None:
        replace_arg(args, "--max-context", str(a.context))
    if a.split is not None:
        cfg["layer_split"] = str(a.split)
        replace_arg(args, "--layer-split", str(a.split))
    if a.resident_budget_gib is not None:
        replace_arg(args, "--resident-budget-gib", str(a.resident_budget_gib))
    for flag, value in (("--kv-resident", a.kv_resident), ("--spec", a.spec), ("--prefill", a.prefill)):
        if value is not None:
            replace_arg(args, flag, str(value))
    if a.vram_reserve_mib is not None:
        cfg["vram_reserve"] = "as_given"
        replace_arg(args, "--vram-reserve-mib", str(a.vram_reserve_mib))
        replace_arg(args, "--vram-reserve-later-mib", str(a.vram_reserve_later_mib))
    if a.save_profile:
        cfg["expert_profile_save"] = str(a.output.with_suffix(".profile.bin").resolve())
        cfg["expert_profile_save_every"] = 0
    cfg = server.plan_split(cfg, cfg["exe"])
    args = server.engine_args(cfg)
    env = server.child_env(cfg)
    if a.branches == "default":
        env.pop("STRATA_DF_BRANCH", None)
    tok = tokenizer(Path(cfg["tokenizer"]))
    template_path = Path(cfg["tokenizer"]) / "chat_template.jinja"
    template = ChatTemplate(template_path if template_path.exists() else server.ROOT / "serve/chat_template.jinja")
    started = time.monotonic()
    engine = MeasuredEngine(cfg["exe"], args, cwd=cfg["cwd"], log=cfg["log"], env=env)
    engine.silence_s = 120
    httpd = None
    results = []
    report = {"pipeline": a.pipeline, "adapt": a.adapt, "startup_s": time.monotonic() - started,
              "max_context": engine.max_context, "requests": results, "status": "RUNNING"}
    report["engine_info"] = dict(engine.info)
    report["thinking"] = a.thinking
    report["reasoning_budget"] = a.reasoning_budget if a.thinking else 0
    report["answer_reserve"] = a.min_answer if a.thinking else 0
    report["requested_requests"] = a.requests
    report["performance_only"] = a.benchmark
    for key, path in (("runner_sha256", Path(__file__)), ("server_sha256", Path(server.__file__)),
                      ("binary_sha256", a.exe)):
        with path.open("rb") as source:
            report[key] = hashlib.file_digest(source, "sha256").hexdigest()

    def persist():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    persist()
    safe_cfg = {key: cfg[key] for key in ("exe", "cwd", "tokenizer", "args", "gpus", "gpu", "layer_split", "log")
                if key in cfg}
    safe_cfg["args"] = args
    safe_cfg["env"] = {key: value for key, value in cfg["env"].items()
                       if key in ("STRATA_DF_BRANCH", "STRATA_IQ_MT_MIN", "STRATA_MTP_CATCHUP_ALL",
                                  "STRATA_PIN_GUARD", "STRATA_STAGER_SLEEP")}
    a.output.with_suffix(".config.json").write_text(json.dumps(safe_cfg, indent=2) + "\n")
    try:
        service = server.Service(engine, tok, template, model_name="pascal-local-test")
        if a.thinking:
            service.reasoning_budget_tokens = a.reasoning_budget
            service.reasoning_min_answer_tokens = a.min_answer
        if a.http:
            httpd = server.serve(service, host="127.0.0.1", port=0)
            base = f"http://127.0.0.1:{httpd.server_address[1]}"
            with urllib.request.urlopen(base + "/health", timeout=10) as response:
                assert response.status == 200
        history = []
        for index in range(a.requests):
            message = "In one short sentence, explain why unit tests are useful."
            if a.vary:
                topics = ("unit tests", "bounded memory", "clear error messages", "repeatable measurements",
                          "small functions", "version control", "local regression tests")
                message = f"In one short sentence, explain why {topics[index % len(topics)]} are useful."
            if a.benchmark:
                message = ("Describe how to design reliable local software tests. Give a numbered list of "
                           "ten concrete practices, each with an explanation and a short example.")
            if a.long_prompt and index == 1:
                message = ("The observation is that reliable systems need bounded resources. " * 800 +
                           "Summarize that observation in one short sentence.")
            expected = str(4187 + index % 7)
            if a.known_answer:
                message = (f"The identifier of the blue archive is {expected}. "
                           "Return only that identifier, not any other number.")
            if a.reasoning_problem:
                numbers = [37 + (17 * j + index * 3) % 93 for j in range(20)]
                expected = str(sum(numbers))
                message = ("Calculate the sum of these integers. Check the arithmetic carefully in your reasoning; "
                           "in your final answer return only the sum, no other number: " +
                           ", ".join(map(str, numbers)))
            messages = [{"role": "user", "content": message}]
            prompt = tok.encode(template.render(messages, enable_thinking=a.thinking), parse_special=True)
            if a.prompt_tokens is not None and (index == 1 or a.repeat_long or a.requests == 1):
                if a.conversation:
                    messages[0]["content"] = sized_prompt(tok, template, a.prompt_tokens, expected, index % 3,
                                                         a.thinking, message_only=True)
                else:
                    prompt = sized_prompt(tok, template, a.prompt_tokens, expected, index % 3, a.thinking)
            if a.conversation:
                messages = history + messages
                prompt = tok.encode(template.render(messages, enable_thinking=a.thinking), parse_special=True)
            assert len(prompt) + a.max_new + 8 <= engine.max_context, "prompt exceeds test context"
            cancel = threading.Event()
            timer = None
            if a.cancel and index % 10 == 5:
                timer = threading.Timer(0.2, cancel.set)
                timer.start()
            begin = time.monotonic()
            first = None
            ids = []
            content, reasoning, done = [], [], None
            calls_before = engine.calls
            report["current_request"] = {"index": index, "prompt_tokens": len(prompt),
                                         "input_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest()}
            persist()
            try:
                if a.known_answer or a.thinking:
                    for kind, event in service.run(prompt, a.thinking, [], a.max_new, {"temperature": 0}, cancel):
                        if kind == "event":
                            if first is None:
                                first = time.monotonic()
                            if event.kind == "content":
                                content.append(event.text)
                            elif event.kind == "reasoning":
                                reasoning.append(event.text)
                        elif kind == "done":
                            done = event
                    text = "".join(content).strip()
                    ids = tok.encode(text)
                    if not cancel.is_set():
                        check_final(done, text, expected if a.known_answer else None, a.max_new)
                        if a.conversation:
                            history = messages + [{"role": "assistant", "content": text}]
                else:
                    for token in engine.generate(prompt, a.max_new, {"temperature": 0}, cancel):
                        if token is None:
                            print(json.dumps({"request": index + 1, "heartbeat": 1,
                                              "elapsed_s": round(time.monotonic() - begin, 1)}), flush=True)
                            continue
                        if first is None:
                            first = time.monotonic()
                        assert isinstance(token, int) and 0 <= token < len(tok.tokens), "invalid token ID"
                        ids.append(token)
            finally:
                if timer is not None:
                    timer.cancel()
                    timer.join()
            elapsed = time.monotonic() - begin
            assert engine.alive(), "engine died"
            assert ids or cancel.is_set(), "request produced no tokens"
            assert len(ids) <= a.max_new, "token limit exceeded"
            for key, value in engine.last.items():
                if isinstance(value, float):
                    assert math.isfinite(value), f"nonfinite engine metric: {key}"
            results.append({"index": index, "prompt_tokens": len(prompt), "tokens": len(ids),
                            "cancelled": cancel.is_set(), "elapsed_s": elapsed,
                            "ttft_s": None if first is None else first - begin,
                            "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                            "metrics": dict(engine.last)})
            results[-1].update(input_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                               known_answer_ok=a.known_answer and not cancel.is_set(),
                               final_chars=sum(map(len, content)), reasoning_chars=sum(map(len, reasoning)),
                               done=done, engine_passes=engine.calls - calls_before)
            if a.fresh_parity and index == a.requests - 1 and not cancel.is_set():
                if engine.last.get("reused", 0) <= 0:
                    raise AssertionError("conversation parity did not exercise prefix reuse")
                before_text = "".join(content)
                # A different prefix invalidates the previous request's reusable prompt.
                reset_ids = tok.encode(template.render([{"role": "user", "content": "Return only OK."}],
                                                       enable_thinking=False), parse_special=True)
                for _ in engine.generate(reset_ids, 32, {"temperature": 0}, threading.Event()):
                    pass
                fresh = []
                fresh_done = None
                for kind, event in service.run(prompt, a.thinking, [], a.max_new, {"temperature": 0},
                                               threading.Event()):
                    if kind == "event" and event.kind == "content":
                        fresh.append(event.text)
                    elif kind == "done":
                        fresh_done = event
                same = "".join(fresh) == before_text
                results[-1]["fresh_parity"] = same
                results[-1]["fresh_reused"] = engine.last.get("reused")
                if not same or fresh_done is None or fresh_done["finish"] != "stop" or engine.last.get("reused") != 0:
                    raise AssertionError("fresh prefill final-answer parity failed")
            status = Path(f"/proc/{engine.proc.pid}/status").read_text()
            memory = {}
            for line in status.splitlines():
                fields = line.split()
                if fields and fields[0] in ("VmRSS:", "VmSwap:"):
                    memory[fields[0][:-1] + "_kib"] = int(fields[1])
            results[-1]["memory"] = memory
            persist()
            print(json.dumps({"request": index + 1, "tokens": len(ids), "elapsed_s": round(elapsed, 3),
                              "cancelled": int(cancel.is_set())}), flush=True)
        if httpd is not None:
            body = {"model": "pascal-local-test", "messages": [{"role": "user", "content":
                    "The identifier of the blue archive is 4199. Return only that identifier."}],
                    "max_tokens": a.max_new, "temperature": 0,
                    "reasoning_effort": "high" if a.thinking else "none"}
            if a.thinking:
                body.update(reasoning_budget_tokens=a.reasoning_budget, reasoning_min_answer_tokens=a.min_answer)
            for streaming in (False, True):
                request = urllib.request.Request(base + "/v1/chat/completions",
                                                 data=json.dumps({**body, "stream": streaming}).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if not streaming:
                        data = json.load(response)
                        assert data["object"] == "chat.completion"
                        assert len(data["choices"]) == 1
                        text = data["choices"][0]["message"]["content"]
                        assert isinstance(text, str)
                        check_final({"finish": data["choices"][0]["finish_reason"],
                                     "completion_tokens": data["usage"]["completion_tokens"]},
                                    text, "4199", a.max_new)
                        assert 0 < data["usage"]["completion_tokens"] <= a.max_new
                        report["http_completion_tokens"] = data["usage"]["completion_tokens"]
                    else:
                        fragments, finish, ended = [], None, False
                        for line in response:
                            if line.strip() == b"data: [DONE]":
                                ended = True
                            elif line.startswith(b"data: {"):
                                data = json.loads(line[6:])
                                for choice in data["choices"]:
                                    fragments.append(choice["delta"].get("content") or "")
                                    finish = choice.get("finish_reason") or finish
                        assert ended
                        check_final({"finish": finish, "completion_tokens": len(tok.encode("".join(fragments)))},
                                    "".join(fragments), "4199", a.max_new)
                        report["http_stream_gate"] = True
        measured = [r["elapsed_s"] for r in results[1:] if not r["cancelled"]]
        report["median_elapsed_s"] = statistics.median(measured) if measured else None
        if len(results) >= 100:
            rss_growth = results[-1]["memory"]["VmRSS_kib"] - results[9]["memory"]["VmRSS_kib"]
            report["post_warmup_rss_growth_kib"] = rss_growth
            assert rss_growth <= 512 * 1024, "engine RSS grew more than 512 MiB after warmup"
        report["status"] = "OK"
        report.pop("current_request", None)
        return 0
    except Exception as e:
        report["status"] = "ERROR"
        report["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        engine.close()
        if a.save_profile and a.adapt == "off":
            report["saved_profile_not_applicable"] = "adaptive tier disabled; engine does not learn a profile"
        elif a.save_profile and report["status"] == "OK":
            shape = server.profile_shape(a.output.with_suffix(".profile.bin"))
            if shape != (48, 512):
                report.update(status="ERROR", error=f"saved profile shape mismatch: {shape}")
                persist()
                raise AssertionError("saved profile shape mismatch")
            report["saved_profile_shape"] = list(shape)
        persist()


if __name__ == "__main__":
    sys.exit(main())
