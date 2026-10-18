"""CPU-only regression tests for the steering E2E engine invocation."""
import pathlib
import tempfile
import os
import subprocess
import unittest
from unittest.mock import Mock, patch

from tools import test_steering_e2e as runner
from tools import test_steering_server as server_runner
from serve.server import child_env, engine_args


class EngineInvocation(unittest.TestCase):
    def run_fake(self, tokens, last):
        engine = Mock()
        engine.generate.return_value = iter(tokens)
        engine.last = last

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)

        def create(extra_args):
            cfg = runner.runtime_config()
            args = engine_args(cfg) + extra_args
            self.assertIn("--layer-split", args)
            for flag, value in (
                ("--prefill", "auto"),
                ("--spec", "4"),
                ("--mtp", "/home/dino/proj/Strata-data/mtp/swift15-ft2"),
                ("--resident-budget-gib", "16"),
                ("--max-context", "2048"),
            ):
                self.assertEqual(args.count(flag), 1)
                self.assertEqual(args[args.index(flag) + 1], value)
            self.assertNotIn("--tokens", args)
            self.assertNotIn("--max-new", args)
            self.assertEqual(child_env(cfg)["CUDA_VISIBLE_DEVICES"], "1,0")
            self.assertEqual(child_env(cfg)["STRATA_PIN_GUARD"], "2")
            log = pathlib.Path(directory.name) / "engine.log"
            log.write_text("test stderr\n", encoding="utf-8")
            return engine, log

        return engine, patch.object(runner, "start_engine", side_effect=create)

    def test_runtime_matches_launcher_except_context(self):
        import json
        source = json.loads((runner.ROOT / "strata-swift-upstream-pascal.json").read_text())
        with patch.dict("os.environ", {}, clear=True):
            cfg = runner.runtime_config()
        expected = engine_args(source)
        expected[expected.index("--max-context") + 1] = "2048"
        self.assertEqual(engine_args(cfg), expected)
        self.assertEqual(cfg["gpu"], source["gpu"])
        self.assertEqual(cfg["env"], source["env"])

    def test_unbounded_or_unpaced_config_rejected(self):
        import json
        source = json.loads((runner.ROOT / "strata-swift-upstream-pascal.json").read_text())
        for kind in ("budget", "guard"):
            cfg = json.loads(json.dumps(source))
            if kind == "budget":
                cfg["args"][cfg["args"].index("--resident-budget-gib") + 1] = "20"
            else:
                cfg["env"].pop("STRATA_PIN_GUARD")
            with self.subTest(kind=kind), patch.object(runner.json, "loads", return_value=cfg):
                with self.assertRaises(ValueError):
                    runner.runtime_config()

    def test_server_exception_closes(self):
        engine = Mock()
        with patch.object(server_runner, "start_engine", return_value=(engine, pathlib.Path("unused"))), \
                patch.object(server_runner, "_run_suite", side_effect=RuntimeError("request failed")):
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                server_runner.run_suite(pathlib.Path("vector"), "test")
        engine.close.assert_called_once()

    def test_start_failure_retains_log(self):
        cfg = runner.runtime_config()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)

            def fail(exe, args, *, cwd, env, log):
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,0")
                self.assertIn("--resident-experts", args)
                pathlib.Path(log).write_text("startup failed\n")
                raise RuntimeError("startup failed")

            with patch.object(runner, "ROOT", root), patch.object(runner, "runtime_config", return_value=cfg), \
                    patch.object(runner, "StrataEngine", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "startup failed"):
                    runner.start_engine([])
            logs = list((root / "logs" / "steering-tests").glob("*.log"))
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0].read_text(), "startup failed\n")

    def test_shell_stops_after_first_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            script = root / "steering-test-run.sh"
            script.write_text((runner.ROOT / "steering-test-run.sh").read_text())
            bin_dir = root / ".venv" / "bin"
            bin_dir.mkdir(parents=True)
            fake_python = bin_dir / "python"
            fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> calls\nexit 1\n')
            fake_python.chmod(0o755)
            fake_pgrep = bin_dir / "pgrep"
            fake_pgrep.write_text("#!/bin/sh\nexit 1\n")
            fake_pgrep.chmod(0o755)
            env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
            result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertEqual((root / "calls").read_text().splitlines(), ["tools/test_steering_e2e.py"])

    def test_serve_client_generation_and_cleanup(self):
        engine, patched = self.run_fake([None, 7, None, 8], {"finish": "length", "generated": 2})
        with patched:
            _, tokens, stderr = runner.run_engine([], [1, 2], 2)
        self.assertEqual(tokens, [7, 8])
        self.assertTrue(stderr.startswith("test stderr\n"))                 # the engine log, then
        self.assertIn("[test-req] label=baseline", stderr)                # the request line log_request appends
        self.assertEqual(engine.generate.call_args.args[:3], ([1, 2], 2, {"seed": runner.steering_seed()}))
        engine.close.assert_called_once()

    def test_invalid_generation_fails_and_closes(self):
        for tokens, last in (
            ([], {"finish": "stop", "generated": 0}),
            ([7], {}),
            ([7], {"finish": "cancel", "generated": 1}),
            ([7], {"finish": "length", "generated": 2}),
            ([7, 8, 9], {"finish": "length", "generated": 3}),
        ):
            with self.subTest(tokens=tokens, last=last):
                engine, patched = self.run_fake(tokens, last)
                with patched, self.assertRaises(RuntimeError):
                    runner.run_engine([], [1], 2)
                engine.close.assert_called_once()

    def test_generation_exception_closes(self):
        engine, patched = self.run_fake([], {})
        engine.generate.side_effect = RuntimeError("protocol error")
        with patched, self.assertRaisesRegex(RuntimeError, "protocol error"):
            runner.run_engine([], [1])
        engine.close.assert_called_once()

    def test_server_cross_format_mismatch_fails(self):
        with patch.object(server_runner, "run_suite", side_effect=[
            (True, [1], [2], [3], [4]),
            (True, [1], [9], [3], [4]),
        ]), patch("builtins.print"), self.assertRaisesRegex(AssertionError, "parity failed"):
            server_runner.main()


if __name__ == "__main__":
    unittest.main()
