"""Offline negative controls for the overnight supervisor and prompt sizing."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from serve.test_server import ByteTokenizer
from serve.frontend import ChatTemplate
from tools.night_pascal import ROOT, matrix, run_case, wait_gpu_idle
from tools.test_upstream_pascal import check_final, sized_prompt
from tools.night_report_pascal import recommend, summarize


class SupervisorTests(unittest.TestCase):
    def test_success_and_failure_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case.json"
            for code in (0, 7):
                result = run_case([sys.executable, "-c", f"raise SystemExit({code})"],
                                  output, time.time() + 20, 10, min_free_kib=0)
                self.assertEqual(result["status"], "PASS" if code == 0 else "FAIL")
                self.assertEqual(json.loads(output.read_text())["returncode"], code)

    def test_timeout_kills_worker_and_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile = root / "child.pid"
            code = ("import subprocess,sys,time; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                    f"Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)")
            result = run_case([sys.executable, "-c", code], root / "result.json",
                              time.time() + 20, 1, min_free_kib=0)
            self.assertEqual(result["status"], "TIMEOUT")
            pid = int(pidfile.read_text())
            status = Path(f"/proc/{pid}/status")
            if status.exists():
                self.assertIn("State:\tZ", status.read_text())

    def test_gpu_probe_timeout_is_saved_and_worker_cleaned_up(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case.json"
            with patch("tools.night_pascal.subprocess.run",
                       side_effect=subprocess.TimeoutExpired("nvidia-smi", 15)):
                result = run_case([sys.executable, "-c", "import time; time.sleep(60)"],
                                  output, time.time() + 20, 10, min_free_kib=0, gpu_monitor=True)
            self.assertEqual(result["status"], "ERROR")
            self.assertIn("TimeoutExpired", result["error"])
            self.assertNotEqual(result["returncode"], 0)
            self.assertEqual(json.loads(output.read_text())["status"], "ERROR")

    def test_idle_probe_recovers_without_terminating_processes(self):
        with patch("tools.night_pascal.subprocess.run", side_effect=[
                subprocess.TimeoutExpired("nvidia-smi", 20),
                subprocess.CompletedProcess([], 0, stdout="1234\n"),
                subprocess.CompletedProcess([], 0, stdout="")]), patch("tools.night_pascal.time.sleep"):
            self.assertEqual(wait_gpu_idle(), (True, None))

    def test_idle_probe_stops_after_three_occupied_results(self):
        with patch("tools.night_pascal.subprocess.run", return_value=
                   subprocess.CompletedProcess([], 0, stdout="1234\n")) as probe, \
                patch("tools.night_pascal.time.sleep"):
            idle, error = wait_gpu_idle()
            self.assertFalse(idle)
            self.assertIn("1234", error)
            self.assertEqual(probe.call_count, 3)

    def test_dry_run_does_not_load_config_or_model(self):
        result = subprocess.run([sys.executable, "-m", "tools.night_pascal",
                                 "--baseline", "/not-present", "--output-dir", "/not-present"],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["model_started"])
        cases = matrix()
        self.assertEqual(len({case["name"] for case in cases}), len(cases))

    def test_exact_size_and_all_fact_positions(self):
        tok = ByteTokenizer()
        template = ChatTemplate(ROOT / "serve/chat_template.jinja")
        for position in (0, 1, 2):
            for thinking in (False, True):
                prompt = sized_prompt(tok, template, 4096, "4187", position, thinking)
                self.assertEqual(len(prompt), 4096)
                self.assertIn("4187", tok.decode(prompt))
        with self.assertRaises(ValueError):
            sized_prompt(tok, template, 4, "4187", 0, False)

    def test_final_answer_gate_rejects_truncation_and_wrong_answers(self):
        done = {"finish": "stop", "completion_tokens": 4}
        check_final(done, "4187", "4187", 128)
        for result, text in ((None, "4187"), ({**done, "finish": "length"}, "4187"),
                             (done, ""), (done, "4188"), (done, "4187 and 4188"),
                             ({**done, "completion_tokens": 129}, "4187")):
            with self.assertRaises(AssertionError):
                check_final(result, text, "4187", 128)

    def test_report_never_qualifies_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({"cases": [
                {"name": "soak100", "status": "PASS", "stage": "soak"}]}))
            (root / "soak100.json").write_text(json.dumps({
                "status": "RUNNING", "requested_requests": 100, "requests": []}))
            rows = summarize(root)
            self.assertEqual(rows[0]["status"], "INCOMPLETE_EVIDENCE")
            self.assertIsNone(recommend(root, rows)["qualified_order"])

    def test_report_requires_actual_wrap_http_and_fresh_parity(self):
        rows = [{"name": name, "status": "PASS"} for name in
                ("reasoning-arithmetic", "http-thinking", "soak100", "context-30000",
                 "thinking-forced-wrap", "conversation-reuse", "serial-thinking",
                 "serial-soak100", "serial-context30k", "serial-conversation")]
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(recommend(Path(directory), rows)["qualified_order"])

    def test_report_recovers_interrupted_case_and_supervisor_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary.json").write_text(json.dumps({"cases": [
                {"name": "pipeline1", "status": "ERROR"}]}))
            (root / "pipeline1.supervisor.json").write_text(json.dumps({
                "status": "ERROR", "error": "TimeoutExpired: GPU monitoring"}))
            (root / "context-16384.supervisor.json").write_text(json.dumps({"status": "RUNNING"}))
            (root / "context-16384.json").write_text(json.dumps({
                "status": "RUNNING", "requested_requests": 3, "requests": [
                    {"prompt_tokens": 16384, "cancelled": False, "elapsed_s": 5, "metrics": {}}]}))
            rows = {row["name"]: row for row in summarize(root)}
            self.assertEqual(rows["pipeline1"]["error"], "TimeoutExpired: GPU monitoring")
            self.assertEqual(rows["context-16384"]["status"], "INCOMPLETE_EVIDENCE")
            self.assertEqual(rows["context-16384"]["requests"], 1)
            self.assertEqual(rows["context-16384"]["prompt_max"], 16384)
            self.assertEqual(rows["soak100"]["status"], "NOT_RUN")

    def test_report_serial_recommendation_preserves_verified_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.config.json").write_text(json.dumps({
                "args": [], "env": {}, "gpu": [0, 1]}))
            rows = [{"name": name, "status": "PASS"} for name in
                    ("reasoning-arithmetic", "serial-thinking", "serial-soak100",
                     "serial-context30k", "serial-conversation", "serial")]
            rows[1].update(two_pass_requests=3, http_stream_gate=True, http_completion_tokens=40)
            rows[2].update(requests=100, cancelled=10, known_correct=90, rss_growth_kib=1024)
            rows[4]["fresh_parity"] = [True]
            result = recommend(root, rows)
            self.assertEqual(result["qualified_order"], "serial")
            config = json.loads(Path(result["config"]).read_text())
            self.assertEqual(config["gpu"], [1, 0])
            self.assertEqual(config["args"][config["args"].index("--pipeline-windows") + 1], "0")
            self.assertEqual(config["reasoning_budget_tokens"], 1024)
            self.assertEqual(config["reasoning_min_answer_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
