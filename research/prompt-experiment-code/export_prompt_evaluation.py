"""Export evaluation questions after prompt selection, keeping answers private."""

import hashlib
import json
from pathlib import Path
import re

ROOT = Path("/home/azureuser/aime-speedrun")


def prepare_private_dataset(private):
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = private / "source.parquet"
    # Keep gold as an Arrow array for grader provisioning; never decode or print it.
    table = pq.read_table(source, columns=["problem", "answer"], filters=[("year", "=", 2022)])
    assert len(table) == 30
    indices = pa.array(list(range(1, 31)))
    table = table.add_column(0, "problem_idx", indices)
    evaluation = private / "evaluation.parquet"
    assert not evaluation.exists()
    pq.write_table(table, evaluation)
    evaluation.chmod(0o600)
    rows = table.select(["problem_idx", "problem"]).to_pylist()
    del table
    return rows, source, evaluation


def normalise(text):
    return re.sub(r"\s+", "", text).casefold()


def check_overlap(root, rows):
    new = {normalise(row["problem"]) for row in rows}
    assert len(new) == 30
    previous = [
        root / "data/questions.jsonl",
        root / "research/fresh-comparison-data/dev-questions.jsonl",
        *(root / "research/fresh-validation-01").glob("*-questions.jsonl"),
    ]
    overlap = {}
    for path in previous:
        old = {normalise(json.loads(line)["problem"]) for line in path.read_text().splitlines()}
        overlap[str(path.relative_to(root))] = len(new & old)
    assert not any(overlap.values()), "Evaluation question overlap"
    return overlap


def write_questions(out, rows):
    questions = out / "evaluation-questions.jsonl"
    with questions.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return questions


def write_grader_config(private, evaluation):
    config = {
        "dataset": {
            "source": str(evaluation),
            "format": "parquet",
            "idx_field": "problem_idx",
            "gold_field": "answer",
        },
        "cost_c": 3.0,
        "host": "127.0.0.1",
        "port": 8077,
        "audit_log": str(private / "evaluation-audit.jsonl"),
    }
    config_path = private / "evaluation.json"
    config_path.write_text(json.dumps(config))
    config_path.chmod(0o600)


def write_manifest(out, source, questions, overlap):
    (out / "evaluation-manifest.json").write_text(
        json.dumps(
            {
                "repository": "allenai/aime-2022-2025",
                "revision": "73e1eba765ad5847cdb5d1e2e7aaf7b22b585798",
                "year": 2022,
                "count": 30,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
                "overlap": overlap,
            },
            indent=2,
        )
    )


def main():
    out = ROOT / "research/prompt-comparison-01"
    assert (out / "selection.json").exists()
    private = Path("/home/azureuser/aime-prompt-private")
    rows, source, evaluation = prepare_private_dataset(private)
    overlap = check_overlap(ROOT, rows)
    questions = write_questions(out, rows)
    write_grader_config(private, evaluation)
    write_manifest(out, source, questions, overlap)
    print("Exported 30 question-only rows; no exact overlap with prior sets.")


if __name__ == "__main__":
    main()
