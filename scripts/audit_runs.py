"""Check recorded solver evidence without opening grader data or private logs."""

import hashlib
import json
from pathlib import Path
import sys


def audit(directory):
    summary = json.loads((directory / "summary.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert (
        hashlib.sha256((directory / "solver.py").read_bytes()).hexdigest()
        == metadata["solver_sha256"]
    )
    assert metadata["grader_health"]["queries_so_far"] == 0
    assert metadata["grader_health"]["cost_c"] >= 3
    pending = None
    solved = set()
    milestones = {}
    tokens = checks = wrong = 0
    seen_votes = {}
    for event in events:
        kind = event["event"]
        if kind == "submit":
            assert pending is None, "Overlapping verification requests"
            pending = event
            checks += 1
            if "votes_at_submission" in event:
                assert event["votes_at_submission"] == seen_votes.get(
                    (event["index"], event["candidate"]), 0
                )
        elif kind == "verdict":
            assert pending is not None
            assert (pending["index"], pending["candidate"]) == (event["index"], event["candidate"])
            assert event["elapsed_s"] - pending["elapsed_s"] >= 3, "Verification toll bypass"
            pending = None
            if event["correct"]:
                assert event["index"] not in solved, "Duplicate solution counted"
                solved.add(event["index"])
                milestones[str(len(solved))] = event["elapsed_s"]
            else:
                wrong += 1
            assert event["solved_count"] == len(solved)
        elif kind == "generation":
            tokens += event["usage"].get("completion_tokens", 0)
            if event["candidate"] is not None:
                key = (event["index"], event["candidate"])
                seen_votes[key] = seen_votes.get(key, 0) + 1
    assert pending is None, "Audit expects fully completed benchmark runs"
    assert len(solved) == summary["solved"]
    assert milestones == summary["milestones_s"]
    assert milestones.get("18") == summary["time_to_18_s"]
    assert checks == summary["counts"]["verification_requests"]
    assert wrong == summary["counts"].get("verified_incorrect", 0)
    assert tokens == summary["completed_response_tokens"]
    if (directory / "cache-setup.json").exists():
        setup = json.loads((directory / "cache-setup.json").read_text())
        assert setup["fresh_server"]
        flag = "--enable-prefix-caching" if setup["prefix_cache"] else "--no-enable-prefix-caching"
        assert flag in setup["server_command"].split()
        before = {}
        for line in (directory / "metrics-before.txt").read_text().splitlines():
            if line.startswith("vllm:"):
                key = line.split("{")[0].split()[0]
                before[key] = before.get(key, 0) + float(line.rsplit(" ", 1)[1])
        for key in (
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:prefix_cache_hits_total",
            "vllm:prefix_cache_queries_total",
        ):
            assert key in before and before[key] == 0, "Server was not empty before run"
        delta = json.loads((directory / "metrics-delta.json").read_text())
        assert (
            0 <= delta["vllm:prefix_cache_hits_total"] <= delta["vllm:prefix_cache_queries_total"]
        )
        if not setup["prefix_cache"]:
            assert delta["vllm:prefix_cache_hits_total"] == 0
    print(f"{directory.name}: PASS ({len(solved)} solved, {checks} serial checks)")


if __name__ == "__main__":
    for path in sorted(Path(sys.argv[1]).glob("*/summary.json")):
        audit(path.parent)
