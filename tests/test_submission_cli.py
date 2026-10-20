"""The local launcher does not require models, devices or participant records."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "submission_runtime", *arguments],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )

    def test_doctor_without_configuration_never_connects_to_a_device(self):
        result = self.invoke("doctor")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"])
        self.assertFalse(report["devices_connected"])
        self.assertFalse(report["models_loaded"])

    def test_smoke_uses_a_real_isolated_synthetic_process(self):
        result = self.invoke("smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(report["scope"], "synthetic transport only")
        self.assertFalse(report["hardware_tested"])

    def test_configured_run_requires_explicit_configuration(self):
        result = self.invoke("run")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("config", result.stderr)

    def test_original_solver_and_delivery_preserve_mood_and_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = json.loads((ROOT / "examples/runtime/host.json").read_text())
            config["policy"]["command"][0] = sys.executable
            config["output"] = {
                "command": [sys.executable, "-m", "submission_runtime.smoke_worker"],
                "runtime": "synthetic-transport", "capabilities": ["present"],
            }
            path = directory / "host.json"
            path.write_text(json.dumps(config))
            journal = directory / "delivery.runtime.jsonl"
            command = [sys.executable, "-m", "submission_runtime", "run", "--config",
                       str(path), "--journal", str(journal)]
            packet = (ROOT / "examples/runtime/decision.jsonl").read_text()
            result = subprocess.run(command, cwd=ROOT, input=packet + packet,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            first, repeated = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(first, repeated)
            self.assertEqual(first["affect"]["model_id"], "synthetic-no-model")
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([r["kind"] for r in records],
                             ["decision.returned", "presentation.attempted", "presentation.delivered"])
            attempted = records[1]["payload"]
            self.assertEqual(attempted["mood"], first["mood"])
            self.assertEqual(attempted["text"], "Synthetic software fixture only.")
            prior = journal.read_bytes()
            retry = subprocess.run(command, cwd=ROOT, input=packet,
                                   capture_output=True, text=True, timeout=10)
            self.assertNotEqual(retry.returncode, 0)
            self.assertIn("previous runtime journal", retry.stderr)
            self.assertEqual(journal.read_bytes(), prior)

    def test_mean_only_perception_does_not_silently_run_the_policy(self):
        packet = json.loads((ROOT / "examples/runtime/decision.jsonl").read_text())
        del packet["affect"]["normalized"]
        result = subprocess.run(
            [sys.executable, "-m", "submission_runtime", "run", "--config", "examples/runtime/host.json"],
            cwd=ROOT, input=json.dumps(packet) + "\n", capture_output=True,
            text=True, timeout=90,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("normalized", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_stiffness_is_not_acknowledged_as_part_of_a_presentation(self):
        packet = json.loads((ROOT / "examples/runtime/decision.jsonl").read_text())
        packet["stiffness"] = 1
        result = subprocess.run(
            [sys.executable, "-m", "submission_runtime", "run", "--config", "examples/runtime/host.json"],
            cwd=ROOT, input=json.dumps(packet) + "\n", capture_output=True,
            text=True, timeout=90,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stiffness", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_concurrent_launch_cannot_share_delivery_journal(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = json.loads((ROOT / "examples/runtime/host.json").read_text())
            config["policy"]["command"][0] = sys.executable
            path = directory / "host.json"
            path.write_text(json.dumps(config))
            journal = directory / "session.runtime.jsonl"
            command = [sys.executable, "-m", "submission_runtime", "run", "--config",
                       str(path), "--journal", str(journal)]
            first = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                claim = journal.with_suffix(journal.suffix + ".run.lock")
                deadline = time.monotonic() + 5
                while not claim.exists() and first.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(claim.exists())
                second = subprocess.run(command, cwd=ROOT, input="", capture_output=True,
                                        text=True, timeout=10)
                self.assertNotEqual(second.returncode, 0)
                self.assertIn("owned", second.stderr)
                self.assertFalse(journal.exists())
                _, error = first.communicate("", timeout=90)
                self.assertEqual(first.returncode, 0, error)
            finally:
                if first.poll() is None:
                    first.terminate()
                    first.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
