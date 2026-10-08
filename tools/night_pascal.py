"""Sequential, deadline-bounded local validation. No prompt or answer text is printed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def stop_group(proc):
    # Each worker is launched in its own session; this group contains only its test processes.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def memory():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, *value = line.split()
        if key in ("MemAvailable:", "SwapFree:"):
            fields[key[:-1] + "_kib"] = int(value[0])
    return fields


def wait_gpu_idle():
    error = "GPU occupied"
    for attempt in range(3):
        try:
            probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20)
            if probe.returncode == 0 and not probe.stdout.strip():
                return True, None
            error = f"GPU probe status {probe.returncode}; active PID metadata: {probe.stdout.strip()}"
        except subprocess.TimeoutExpired:
            error = "GPU probe timed out during cleanup"
        if attempt < 2:
            time.sleep(10)
    return False, error


def run_case(command, output, deadline, timeout_s, min_free_kib=3 * 1024**2, gpu_monitor=False):
    started = time.time()
    result = {"command": command, "started_unix": started, "status": "RUNNING", "samples": []}
    proc = None
    low_since = None
    last_gpu = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".worker.log").open("w") as log:
        try:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while proc.poll() is None:
                now = time.time()
                sample = {"unix": now, **memory()}
                if gpu_monitor and now - last_gpu >= 10:
                    probe = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,temperature.gpu,utilization.gpu",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
                    if probe.returncode:
                        raise OSError(f"GPU monitoring failed with status {probe.returncode}")
                    sample["gpus"] = [[int(value.strip()) for value in line.split(",")]
                                      for line in probe.stdout.splitlines()]
                    last_gpu = now
                    if any(row[3] >= 85 for row in sample["gpus"]):
                        result["status"] = "BLOCKED_THERMAL"
                        result["samples"].append(sample)
                        break
                result["samples"].append(sample)
                if sample["MemAvailable_kib"] < min_free_kib:
                    low_since = low_since or now
                else:
                    low_since = None
                if now >= deadline or now - started >= timeout_s:
                    result["status"] = "TIMEOUT"
                    break
                if low_since is not None and now - low_since >= 10:
                    result["status"] = "BLOCKED_MEMORY"
                    break
                output.write_text(json.dumps(result, indent=2) + "\n")
                time.sleep(2)
            if result["status"] == "RUNNING":
                result["status"] = "PASS" if proc.returncode == 0 else "FAIL"
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            result.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        finally:
            if proc is not None:
                stop_group(proc)
                result["returncode"] = proc.returncode
            result["elapsed_s"] = time.time() - started
            output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def matrix():
    cases = []

    def add(name, flags, stage, timeout=1800):
        cases.append({"name": name, "flags": flags, "stage": stage, "timeout": timeout})

    known = ["--known-answer", "--requests", "3", "--max-new", "128", "--prompt-tokens", "8192"]
    for name, extra in (
        ("stable-controlled", ["--stable"]),
        ("candidate-controlled", []),
        ("reverse-order", ["--gpu-order", "0,1", "--vram-reserve-mib", "1800",
                           "--vram-reserve-later-mib", "512"]),
        ("serial", ["--pipeline", "0"]),
        ("adaptive-off", ["--adapt", "off"]),
        ("adaptive-blocking", ["--adapt", "blocking"]),
        ("catchup-all", ["--catchup-all"]),
        ("stager-yield", ["--stager-sleep", "0"]),
        ("split20", ["--split", "20"]),
        ("split24", ["--split", "24"]),
    ):
        add(name, known + extra, "screening")
    for budget in (1024, 2048, 4096):
        add(f"thinking-{budget}", ["--known-answer", "--thinking", "--reasoning-budget", str(budget),
                                   "--min-answer", "2048", "--max-new", str(budget + 2304),
                                   "--requests", "3"], "thinking")
    add("thinking-forced-wrap", ["--known-answer", "--thinking", "--reasoning-budget", "8",
                                 "--min-answer", "2048", "--max-new", "2304", "--requests", "3"],
        "thinking")
    for length in (4096, 16384, 24576, 30000):
        add(f"context-{length}", ["--known-answer", "--prompt-tokens", str(length), "--repeat-long",
                                  "--requests", "3", "--max-new", "128"], "long-context", 2400)
    add("thinking-long", ["--known-answer", "--thinking", "--reasoning-budget", "1024",
                         "--min-answer", "2048", "--max-new", "3328", "--requests", "3",
                         "--prompt-tokens", "24576"], "thinking", 2400)
    add("soak100", ["--known-answer", "--requests", "100", "--max-new", "128", "--cancel",
                    "--prompt-tokens", "8192"], "soak", 3600)
    add("conversation-reuse", ["--known-answer", "--requests", "8", "--max-new", "128",
                              "--prompt-tokens", "16384", "--conversation", "--fresh-parity"],
        "long-context", 2400)
    add("resident18", known + ["--resident-budget-gib", "18"], "screening")
    add("resident20", known + ["--resident-budget-gib", "20"], "screening")
    add("what-fits", known + ["--resident-budget-gib", "0"], "diagnosis", 2400)
    for context, length in ((65536, 49152), (65536, 60000), (131072, 98304), (131072, 120000),
                            (262144, 240000)):
        add(f"context-{length}", ["--known-answer", "--context", str(context),
                                  "--kv-resident", "32768", "--prompt-tokens", str(length),
                                  "--requests", "3", "--max-new", "128"], "extreme-context", 2400)
    for pair in range(5):
        for arm in ("stable", "candidate"):
            add(f"performance-{pair}-{arm}", ["--benchmark", "--requests", "3", "--max-new", "128"] +
                (["--stable"] if arm == "stable" else []), "performance", 1800)
    for name, extra in (("stable-explicit-off", ["--stable"]), ("candidate-explicit-off", [])):
        add(name, known + extra, "screening")
    add("pipeline1", known + ["--pipeline", "1"], "screening")
    add("serial-spec2", known + ["--pipeline", "0", "--spec", "2"], "screening")
    add("reasoning-arithmetic", ["--known-answer", "--thinking", "--reasoning-problem",
                                "--reasoning-budget", "1024", "--min-answer", "2048",
                                "--max-new", "3328", "--requests", "3"], "thinking")
    add("adaptive-off-checked", known + ["--adapt", "off"], "screening")
    add("adaptive-blocking-retry", known + ["--adapt", "blocking"], "screening")
    add("http-thinking", ["--known-answer", "--thinking", "--reasoning-budget", "8",
                         "--min-answer", "2048", "--max-new", "2304", "--requests", "3", "--http"],
        "thinking")
    reverse = ["--gpu-order", "0,1", "--vram-reserve-mib", "1800", "--vram-reserve-later-mib", "512"]
    add("reverse-thinking", ["--known-answer", "--thinking", "--reasoning-budget", "8",
                            "--min-answer", "2048", "--max-new", "2304", "--requests", "3", *reverse],
        "thinking")
    add("serial-soak100", ["--known-answer", "--requests", "100", "--max-new", "128", "--cancel",
                          "--prompt-tokens", "8192", "--pipeline", "0"], "soak", 3600)
    add("serial-context30k", ["--known-answer", "--prompt-tokens", "30000", "--repeat-long",
                            "--requests", "3", "--max-new", "128", "--pipeline", "0"], "long-context", 2400)
    add("serial-thinking", ["--known-answer", "--thinking", "--reasoning-budget", "8",
                           "--min-answer", "2048", "--max-new", "2304", "--requests", "3",
                           "--pipeline", "0", "--http"], "thinking")
    add("serial-conversation", ["--known-answer", "--requests", "8", "--max-new", "128",
                               "--prompt-tokens", "16384", "--conversation", "--fresh-parity",
                               "--pipeline", "0"], "long-context", 2400)
    add("reverse-soak100", ["--known-answer", "--requests", "100", "--max-new", "128", "--cancel",
                           "--prompt-tokens", "8192", *reverse], "soak", 3600)
    add("reverse-context30k", ["--known-answer", "--prompt-tokens", "30000", "--repeat-long",
                             "--requests", "3", "--max-new", "128", *reverse], "long-context", 2400)
    add("reverse-conversation", ["--known-answer", "--requests", "8", "--max-new", "128",
                                "--prompt-tokens", "16384", "--conversation", "--fresh-parity",
                                *reverse], "long-context", 2400)
    for pair in range(3):
        for arm, flags in (("normal", []), ("reverse", reverse)):
            add(f"order-performance-{pair}-{arm}", ["--benchmark", "--requests", "3",
                                                   "--max-new", "128", *flags], "performance")
    priorities = {"thinking": 0, "screening": 1, "long-context": 2, "soak": 3,
                  "diagnosis": 4, "performance": 5, "extreme-context": 6}
    return sorted(cases, key=lambda case: priorities[case["stage"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cases = matrix()
    if args.only:
        cases = [case for case in cases if case["name"] in args.only]
        if set(args.only) != {case["name"] for case in cases}:
            ap.error("unknown case name")
    if not args.execute:
        print(json.dumps({"cases": len(cases), "model_started": False}))
        return 0
    baseline = json.loads(args.baseline.read_text())
    deadline = baseline["deadline_unix"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    profile = output / "initial-profile.bin"
    source_profile = ROOT / "data/expert-profile.bin"
    if not profile.exists():
        profile.write_bytes(source_profile.read_bytes())
    config = json.loads((ROOT / "strata-swift-upstream-pascal.json").read_text())
    values = config["args"]
    values[values.index("--expert-profile") + 1] = str(profile)
    config.pop("expert_profile_save", None)
    isolated_config = output / "input.config.json"
    isolated_config.write_text(json.dumps(config, indent=2) + "\n")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
        "deadline_unix": deadline, "cases": [], "binary_sha256": {}}
    for name in ("engine/strata", "build-upstream-pascal/strata"):
        with (ROOT / name).open("rb") as binary:
            digest = hashlib.file_digest(binary, "sha256").hexdigest()
        old_digest = summary["binary_sha256"].get(name)
        if old_digest is not None and old_digest != digest:
            raise ValueError("cannot resume comparable results with a different binary")
        summary["binary_sha256"][name] = digest
    completed = {case["name"] for case in summary["cases"]}
    for case in cases:
        if case["name"] in completed:
            continue
        if time.time() >= deadline - 1200:
            summary["cases"].append({**case, "status": "INCOMPLETE_BUDGET"})
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            continue
        idle, idle_error = wait_gpu_idle()
        if not idle:
            summary["cases"].append({**case, "status": "BLOCKED_GPU_OCCUPIED", "error": idle_error})
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            break
        flags = list(case["flags"])
        stable = "--stable" in flags
        if stable:
            flags.remove("--stable")
        # Zero requests what-fits by removing the explicit budget override, not an invalid zero CLI value.
        base_budget = ["--resident-budget-gib", "16"]
        if "--resident-budget-gib" in flags and flags[flags.index("--resident-budget-gib") + 1] == "0":
            index = flags.index("--resident-budget-gib")
            del flags[index:index + 2]
            base_budget = []
            local_config = {**config, "args": list(config["args"])}
            index = local_config["args"].index("--resident-budget-gib")
            del local_config["args"][index:index + 2]
            case_config = output / "what-fits.config.json"
            case_config.write_text(json.dumps(local_config, indent=2) + "\n")
        else:
            case_config = isolated_config
        command = [sys.executable, "-m", "tools.test_upstream_pascal", "--config", str(case_config),
                   "--exe", "engine/strata" if stable else "build-upstream-pascal/strata",
                   "--output", str(output / (case["name"] + ".json")),
                   "--pipeline", "2", "--adapt", "async", "--gpu-order", "1,0", "--split", "22",
                   *base_budget, "--vram-reserve-mib", "512", "--vram-reserve-later-mib", "1800",
                   "--paced-pin", "--save-profile", "--branches", "off", *flags]
        result = run_case(command, output / (case["name"] + ".supervisor.json"),
                          deadline - 1200, case["timeout"], gpu_monitor=True)
        summary["cases"].append({**case, "status": result["status"], "elapsed_s": result["elapsed_s"],
                                 "returncode": result.get("returncode")})
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary["cases"][-1]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
