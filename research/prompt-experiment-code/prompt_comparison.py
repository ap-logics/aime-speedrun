"""Run the fixed prompt screen and, if qualified, two evaluation pairs."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/azureuser/aime-speedrun")
OUT = ROOT / "research/prompt-comparison-01"
PY = str(ROOT / ".venv/bin/python")


def select_prompt(screen):
    baseline = screen[0]["time_to_18_s"]
    if baseline is None:
        return None
    candidates = [
        row
        for row in screen[1:]
        if row["time_to_18_s"] is not None and row["time_to_18_s"] < baseline
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row["time_to_18_s"])["style"]


def evaluation_order(selected):
    return [(811, "original"), (811, selected), (813, selected), (813, "original")]


def main():
    import fcntl
    from audit_runs import audit
    import fresh_validation as fv

    os.chdir(ROOT)
    lock = (ROOT / "logs/autoresearch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir(exist_ok=False)
    state = {
        "status": "starting",
        "runs": [],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "src/prompt_solver.py",
                "scripts/prompt_comparison.py",
                "docs/prompt-comparison-protocol.md",
            )
        },
    }
    previous = "confirm01-baseline-s711"

    def save():
        (OUT / "state.tmp").write_text(json.dumps(state, indent=2))
        (OUT / "state.tmp").replace(OUT / "state.json")

    def trial(split, style, seed):
        nonlocal previous
        name = f"prompt01-{split}-{style}-s{seed}"
        state.update(status="restarting", active_run=name)
        save()
        subprocess.run(
            [PY, "scripts/restart_prompt_trial.py", previous, name], check=True, timeout=360
        )
        if split == "screen":
            config = "/home/azureuser/aime-validation-private/dev.json"
            questions = ROOT / "research/fresh-validation-01/dev-questions.jsonl"
        else:
            config = "/home/azureuser/aime-prompt-private/evaluation.json"
            questions = OUT / "evaluation-questions.jsonl"
        env = dict(os.environ, GRADER_CONFIG=config, AIME_QUESTIONS=str(questions))
        state["status"] = "running"
        save()
        with (OUT / (name + ".log")).open("x") as log:
            result = subprocess.run(
                [
                    PY,
                    "scripts/profile-prompt.py",
                    name,
                    "--seed",
                    str(seed),
                    "--concurrency",
                    "30",
                    "--max-tokens",
                    "12288",
                    "--finalize-tokens",
                    "32",
                    "--probe-policy",
                    "none",
                    "--prompt-style",
                    style,
                    "--deadline-s",
                    "600",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=690,
            )
        directory = ROOT / "experiments" / name
        summary = json.loads((directory / "summary.json").read_text())
        record = {
            "name": name,
            "split": split,
            "style": style,
            "seed": seed,
            "valid": False,
            **{
                key: summary[key]
                for key in (
                    "time_to_18_s",
                    "solved",
                    "outcome",
                    "completed_response_tokens",
                    "grader_busy_s",
                )
            },
        }
        state["runs"].append(record)
        save()
        audit(directory)
        assert result.returncode == 0
        record["valid"] = True
        profile = json.loads((ROOT / "execution-evidence" / name / "profile.json").read_text())
        record["idle_verified"] = profile["idle_verified"]
        previous = name
        save()
        assert profile["idle_verified"], "Post-run idle check failed"
        return record

    save()
    try:
        screen = [
            trial("screen", style, 801)
            for style in ("original", "direct", "checkpoint", "answer_first")
        ]
        selected = select_prompt(screen)
        choice = {
            "selected": selected,
            "rule": "fastest target-reaching variant faster than successful original",
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
        }
        (OUT / "selection.json").write_text(json.dumps(choice, indent=2))
        state["selection"] = choice
        save()
        if selected:
            subprocess.run(
                [
                    "/home/azureuser/private-grader/.venv/bin/python",
                    "scripts/export_prompt_evaluation.py",
                ],
                check=True,
                timeout=120,
            )
            for seed, style in evaluation_order(selected):
                trial("evaluation", style, seed)
        state.update(status="complete", active_run=None)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            subprocess.run(
                [PY, "scripts/restart_prompt_trial.py", previous, "prompt-cleanup"],
                check=True,
                timeout=360,
            )
            fv.restore_grader()
            state["default_grader_restored"] = fv.STATE.get("default_grader_restored", False)
        except Exception as exc:
            state["restore_error"] = str(exc)
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        (OUT / "finished.json").write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
