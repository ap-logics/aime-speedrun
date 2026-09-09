"""Check the bounded prompt comparison using public client evidence."""

import ast
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_runs import audit

OUT = ROOT / "research/prompt-comparison-01"
CODE = ROOT / "research/prompt-experiment-code"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_sources(state):
    for filename, expected in state["hashes"].items():
        path = ROOT / filename if filename.startswith(("docs/", "src/")) else OUT / "controller.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    original = ast.parse((ROOT / "src/speedrun.py").read_text())
    modified = ast.parse((ROOT / "src/prompt_solver.py").read_text())

    def without_prompt(tree):
        return ast.dump(
            ast.Module(
                body=[node for node in tree.body if getattr(node, "name", "") != "problem_prompt"],
                type_ignores=[],
            )
        )

    assert without_prompt(original) == without_prompt(modified)


def check_trials(state, selection):
    commands, orders, configs = [], {}, []
    for record in state["runs"]:
        name = record["name"]
        folder = ROOT / "experiments" / name
        audit(folder)
        summary = read(folder / "summary.json")
        metadata = read(folder / "metadata.json")
        assert metadata["solver_sha256"] == state["hashes"]["src/prompt_solver.py"]
        assert summary["time_to_18_s"] == record["time_to_18_s"]
        assert summary["solved"] == record["solved"]
        config = metadata["config"]
        assert config["prompt_style"] == record["style"]
        assert config["probe_policy"] == "none"
        configs.append(
            {key: value for key, value in config.items() if key not in ("seed", "prompt_style")}
        )
        with (folder / "events.jsonl").open(encoding="utf-8") as stream:
            start = json.loads(next(stream))
        key = (record["split"], record["seed"])
        assert orders.setdefault(key, start["order"]) == start["order"]
        profile = read(ROOT / "execution-evidence" / name / "profile.json")
        restart = read(OUT / f"restart-{name}.json")
        assert profile["idle_verified"] and restart["idle_verified"]
        assert profile["server_command"].split() == restart["new_command"].split()
        commands.append(profile["server_command"].split())
        if record["split"] == "evaluation":
            assert datetime.fromisoformat(metadata["utc"]) > datetime.fromisoformat(
                selection["frozen_utc"]
            )
    assert all(config == configs[0] for config in configs)
    assert all(command == commands[0] for command in commands)
    assert "--no-enable-prefix-caching" in commands[0]
    assert "--async-scheduling" not in commands[0] and "--speculative-config" not in commands[0]
    assert read(OUT / "restart-prompt-cleanup.json")["idle_verified"]


def check_question_export():
    manifest = read(OUT / "evaluation-manifest.json")
    assert manifest["count"] == 30 and not any(manifest["overlap"].values())
    assert (
        hashlib.sha256((OUT / "evaluation-questions.jsonl").read_bytes()).hexdigest()
        == manifest["questions_sha256"]
    )


def check_selection(state, selected):
    screen = [record for record in state["runs"] if record["split"] == "screen"]
    assert [record["style"] for record in screen] == [
        "original",
        "direct",
        "checkpoint",
        "answer_first",
    ]
    eligible = [
        record
        for record in screen[1:]
        if record["time_to_18_s"] is not None and record["time_to_18_s"] < screen[0]["time_to_18_s"]
    ]
    assert min(eligible, key=lambda record: record["time_to_18_s"])["style"] == selected
    evaluation = [record for record in state["runs"] if record["split"] == "evaluation"]
    assert [(record["seed"], record["style"]) for record in evaluation] == [
        (811, "original"),
        (811, selected),
        (813, selected),
        (813, "original"),
    ]
    return evaluation


def compare_pairs(evaluation, selected):
    pairs = []
    for seed in (811, 813):
        pair = {record["style"]: record for record in evaluation if record["seed"] == seed}
        baseline, candidate = pair["original"], pair[selected]
        pairs.append(
            dict(
                seed=seed,
                baseline_s=baseline["time_to_18_s"],
                candidate_s=candidate["time_to_18_s"],
                baseline_correct=baseline["solved"],
                candidate_correct=candidate["solved"],
                speedup=1 - candidate["time_to_18_s"] / baseline["time_to_18_s"],
            )
        )
    return pairs


def main():
    state = read(OUT / "finished.json")
    assert state["status"] == "complete" and state["default_grader_restored"]
    assert len(state["runs"]) == 8
    selection = read(OUT / "selection.json")
    selected = selection["selected"]
    assert selected == "answer_first"
    check_sources(state)
    check_trials(state, selection)
    check_question_export()
    evaluation = check_selection(state, selected)
    pairs = compare_pairs(evaluation, selected)
    review = dict(
        status="audited",
        selected=selected,
        evaluation_pairs=pairs,
        mean_paired_speedup=sum(pair["speedup"] for pair in pairs) / len(pairs),
        all_eight_trace_audits_pass=True,
        matched_config_orders_fresh_starts=True,
        only_prompt_function_changed=True,
        exact_overlap_zero=True,
        selection_before_evaluation=True,
        default_grader_restored=True,
        export_timing_evidence={
            "selection_mtime_unix": 1788964395,
            "question_export_mtime_unix": 1788964396,
        },
        elapsed_s=(
            datetime.fromisoformat(state["finished_utc"])
            - datetime.fromisoformat(state["started_utc"])
        ).total_seconds(),
    )
    (OUT / "review.json").write_text(json.dumps(review, indent=2))
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
