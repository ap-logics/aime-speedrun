import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prompt_comparison", ROOT / "research/prompt-experiment-code/prompt_comparison.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class PromptSelectionTests(unittest.TestCase):
    def test_recorded_screen_selects_the_frozen_prompt(self):
        state = json.loads((ROOT / "research/prompt-comparison-01/finished.json").read_text())
        screen = [row for row in state["runs"] if row["split"] == "screen"]
        self.assertEqual(experiment.select_prompt(screen), "answer_first")

    def test_failed_baseline_or_no_improvement_stops_selection(self):
        for baseline in (None, 100):
            screen = [
                {"style": "original", "time_to_18_s": baseline},
                {"style": "direct", "time_to_18_s": 100},
                {"style": "checkpoint", "time_to_18_s": None},
            ]
            self.assertIsNone(experiment.select_prompt(screen))

    def test_ties_preserve_the_fixed_screen_order(self):
        screen = [
            {"style": "original", "time_to_18_s": 120},
            {"style": "direct", "time_to_18_s": 100},
            {"style": "checkpoint", "time_to_18_s": 100},
        ]
        self.assertEqual(experiment.select_prompt(screen), "direct")

    def test_evaluation_reverses_execution_order(self):
        self.assertEqual(
            experiment.evaluation_order("answer_first"),
            [(811, "original"), (811, "answer_first"), (813, "answer_first"), (813, "original")],
        )


class HelperTests(unittest.TestCase):
    def load(self, name):
        spec = importlib.util.spec_from_file_location(
            name.replace("-", "_"), ROOT / "research/prompt-experiment-code" / name
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_helpers_import_without_starting_a_trial(self):
        from unittest.mock import patch
        import subprocess
        import urllib.request

        with patch.object(
            subprocess, "Popen", side_effect=AssertionError("process on import")
        ), patch.object(urllib.request, "urlopen", side_effect=AssertionError("network on import")):
            for name in (
                "restart_prompt_trial.py",
                "profile-prompt.py",
                "export_prompt_evaluation.py",
            ):
                self.assertTrue(callable(self.load(name).main))

    def test_monitor_stops_if_solver_launch_fails(self):
        from unittest.mock import Mock, patch

        module = self.load("profile-prompt.py")
        monitor = Mock()
        with patch.object(module.subprocess, "Popen", return_value=monitor), patch.object(
            module.subprocess, "run", side_effect=OSError("launch failed")
        ):
            with self.assertRaises(OSError):
                module.run_with_monitor(ROOT, ROOT / "tmp", ["test-run"])
        monitor.terminate.assert_called_once()
        monitor.communicate.assert_called_once_with(timeout=10)

    def test_profile_detects_remaining_work(self):
        module = self.load("profile-prompt.py")
        before = {"vllm:num_requests_running": 0, "vllm:num_requests_waiting": 0, "tokens_total": 5}
        after = {**before, "vllm:num_requests_waiting": 1, "tokens_total": 12}
        record = module.make_record("server", before, after)
        self.assertFalse(record["idle_verified"])
        self.assertEqual(record["delta"], {"tokens_total": 7})
