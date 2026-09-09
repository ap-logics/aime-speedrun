"""Run a measured policy against a fresh grader."""

import argparse
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def command(args):
    policy = "priced" if args.policy == "priced" else "none"
    solver = "prompt_solver.py" if args.policy == "answer-first" else "speedrun.py"
    return [
        sys.executable,
        str(ROOT / "src" / solver),
        "--questions",
        args.questions,
        "--output",
        args.output,
        "--model-url",
        args.model_url,
        "--grader-url",
        args.grader_url,
        "--seed",
        str(args.seed),
        "--deadline-s",
        str(args.deadline),
        "--concurrency",
        "30",
        "--max-tokens",
        "12288",
        "--finalize-tokens",
        "32",
        "--probe-tokens",
        "8192",
        "--probe-policy",
        policy,
        "--prompt-style",
        "answer_first" if args.policy == "answer-first" else "original",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=("baseline", "answer-first", "priced"))
    parser.add_argument("--questions", required=True, help="Question-only JSONL file")
    parser.add_argument("--output", required=True, help="New directory for this run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline", type=float, default=600)
    parser.add_argument("--model-url", default="http://127.0.0.1:8000")
    parser.add_argument("--grader-url", default="http://127.0.0.1:8077")
    args = parser.parse_args()
    if not math.isfinite(args.deadline) or args.deadline <= 0:
        parser.error("--deadline must be a positive, finite number")
    if not Path(args.questions).is_file():
        parser.error("--questions must name an existing JSONL file")
    if Path(args.output).exists():
        parser.error("--output must be a new directory")
    try:
        return subprocess.run(command(args), check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
