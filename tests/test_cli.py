import hashlib
import importlib.util
from argparse import Namespace

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "run.py"), *args],
            capture_output=True,
            text=True,
        )

    def test_invalid_deadlines_fail_before_connecting(self):
        for deadline in ("0", "-1", "nan", "inf"):
            result = self.invoke(
                "baseline",
                "--questions",
                "missing.jsonl",
                "--output",
                "unused",
                "--deadline",
                deadline,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("positive, finite", result.stderr)

    def test_missing_questions_fail_before_connecting(self):
        result = self.invoke("priced", "--questions", "missing.jsonl", "--output", "unused")
        self.assertEqual(result.returncode, 2)
        self.assertIn("existing JSONL file", result.stderr)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            questions = Path(folder) / "questions.jsonl"
            questions.write_text("")
            result = self.invoke("baseline", "--questions", str(questions), "--output", folder)
            self.assertEqual(result.returncode, 2)
            self.assertIn("new directory", result.stderr)


class SelectedSourceTests(unittest.TestCase):
    def test_launcher_uses_the_evaluated_source_and_prompt(self):
        spec = importlib.util.spec_from_file_location("launcher", ROOT / "run.py")
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        args = Namespace(
            policy="answer-first",
            questions="questions.jsonl",
            output="new-run",
            model_url="http://model",
            grader_url="http://grader",
            seed=811,
            deadline=600,
        )
        command = launcher.command(args)
        self.assertEqual(Path(command[1]), ROOT / "src/prompt_solver.py")
        self.assertEqual(command[command.index("--prompt-style") + 1], "answer_first")
        self.assertEqual(command[command.index("--probe-policy") + 1], "none")
        source = (ROOT / "src/prompt_solver.py").read_bytes()
        self.assertEqual(
            hashlib.sha256(source).hexdigest(),
            "560ace4f758748819b76bf11bd3ce2786482f309768d2f50ea0d94924998d00d",
        )
