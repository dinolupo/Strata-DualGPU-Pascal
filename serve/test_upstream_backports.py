"""Targeted upstream reliability regressions; no GPU or model required."""
from __future__ import annotations

import contextlib
import io
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve import server


class SplitProbe(unittest.TestCase):
    def test_incomplete_speed_keeps_order(self):
        for speeds in ({0: 200.0}, {0: 200.0, 1: 0.0}, {0: float("nan"), 1: 100.0}):
            self.assertEqual(server.split_order(speeds, [0, 1]), [0, 1])

    def test_failed_probe_does_not_use_stdout(self):
        result = SimpleNamespace(returncode=1, stdout="gpu 0 sms=10 khz=1000000\n")
        with mock.patch.object(server.subprocess, "run", return_value=result), \
                contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertEqual(server.card_speeds("strata", [0, 1], {}), {})
        self.assertIn("exited 1", log.getvalue())

    def test_partial_probe_reports_warning(self):
        result = SimpleNamespace(returncode=0, stdout="gpu 0 sms=10 khz=1000000\n")
        with mock.patch.object(server.subprocess, "run", return_value=result), \
                contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertEqual(server.card_speeds("strata", [0, 1], {}), {})
        self.assertIn("incomplete", log.getvalue())

    def test_timeout_reports_warning(self):
        with mock.patch.object(server.subprocess, "run", side_effect=subprocess.TimeoutExpired("strata", 120)), \
                contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertEqual(server.card_speeds("strata", [0, 1], {}), {})
        self.assertIn("TimeoutExpired", log.getvalue())

    def test_full_probe_preserves_device_mapping(self):
        result = SimpleNamespace(returncode=0, stdout="gpu 0 sms=10 khz=1000000\ngpu 1 sms=20 khz=1000000\n")
        with mock.patch.object(server.subprocess, "run", return_value=result):
            self.assertEqual(server.card_speeds("strata", [1, 0], {}), {1: 10.0, 0: 20.0})

    def test_partial_display_keeps_reserves(self):
        args = ["--spec", "4"]
        self.assertEqual(server.split_reserves({0: True}, [0, 1], args), args)


class ReadyTimeout(unittest.TestCase):
    def test_timeout_overrides(self):
        key = "STRATA_ENGINE_READY_S"
        with mock.patch.dict(os.environ, {key: "0"}):
            self.assertIsNone(server._timeout_env(key, 10))
        with mock.patch.dict(os.environ, {key: "2.5"}):
            self.assertEqual(server._timeout_env(key, 10), 2.5)
        for value in ("-1", "nan", "inf", "invalid"):
            with mock.patch.dict(os.environ, {key: value}):
                with self.assertRaisesRegex(ValueError, key):
                    server._timeout_env(key, 10)

    def test_silent_engine_is_killed(self):
        procs = []
        real_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake_engine.py"
            script.write_text("import time\nprint('INFO engine=0.0.0', flush=True)\ntime.sleep(600)\n")

            def popen(cmd, **kwargs):
                proc = real_popen([sys.executable, str(script), *cmd[1:]], **kwargs)
                procs.append(proc)
                return proc

            try:
                with mock.patch.object(server.subprocess, "Popen", side_effect=popen), \
                        mock.patch.object(server, "ENGINE_READY_S", 0.3):
                    with self.assertRaisesRegex(RuntimeError, "did not report READY"):
                        server.StrataEngine("strata", [])
                self.assertIsNotNone(procs[0].poll())
            finally:
                for proc in procs:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=10)

    def test_ready_pump_leaves_request_lines_unread(self):
        output = io.StringIO("INFO engine=0.1.38\nREADY 4096 stop\nT 7\nDONE\n")
        got = queue.Queue()
        server.StrataEngine._ready_pump(SimpleNamespace(stdout=output), got)
        self.assertEqual(got.get_nowait(), "INFO engine=0.1.38\n")
        self.assertEqual(got.get_nowait(), "READY 4096 stop\n")
        self.assertTrue(got.empty())
        self.assertEqual(output.readline(), "T 7\n")

    def test_close_does_not_join_an_unstarted_pump(self):
        engine = server.StrataEngine.__new__(server.StrataEngine)
        engine.proc = SimpleNamespace(poll=lambda: 0, stdin=io.StringIO(), stdout=io.StringIO())
        engine.pump = threading.Thread()
        engine.log = None
        engine.close()
        self.assertIsNone(engine.proc)
        self.assertTrue(engine.ended)

    def test_quiet_request_cancellation_sends_stop_promptly(self):
        engine = server.StrataEngine.__new__(server.StrataEngine)
        engine.lines = queue.Queue()
        sent = []

        def write(line):
            sent.append(line)
            if line == "STOP\n":
                engine.lines.put("DONE 0 1 0 0 stop\n")

        engine.proc = SimpleNamespace(stdin=SimpleNamespace(write=write, flush=lambda: None))
        engine.can_stop = True
        engine.silence_s = 0
        cancel = threading.Event()
        cancel.set()
        started = time.monotonic()
        self.assertEqual(list(engine.generate([1], 1, {}, cancel)), [])
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(sent[-1], "STOP\n")


class VisionTimeout(unittest.TestCase):
    def test_silent_encoder_is_killed(self):
        release = threading.Event()
        killed = threading.Event()

        def readline():
            release.wait(5)
            return ""

        def kill():
            killed.set()
            release.set()

        vision = server.Vision.__new__(server.Vision)
        vision.stopped = False
        vision.proc = SimpleNamespace(stdout=SimpleNamespace(readline=readline), kill=kill,
                                     wait=lambda **kwargs: 0, poll=lambda: None)
        with self.assertRaisesRegex(RuntimeError, "said nothing"):
            vision._readline(0.1, "encoder")
        self.assertTrue(killed.is_set())
        self.assertFalse(vision.alive())

    def test_timely_read_and_unlimited_timeout(self):
        vision = server.Vision.__new__(server.Vision)
        vision.proc = SimpleNamespace(stdout=io.StringIO("OK 7\nOK 8\n"))
        self.assertEqual(vision._readline(5.0, "encoder"), "OK 7\n")
        self.assertEqual(vision._readline(None, "encoder"), "OK 8\n")


if __name__ == "__main__":
    unittest.main()
