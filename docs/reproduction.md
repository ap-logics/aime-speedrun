# Setup

I ran this on an A100 PCIe 80GB with Python 3.12, vLLM 0.10.1 and Transformers 4.55.2. The package versions are in [a100-requirements.lock.txt](a100-requirements.lock.txt).

The model is `WeiboAI/VibeThinker-3B`, pinned to revision `77bd2cced09193c8b9a59a32bd8577bbd1f3e01c`.

Create a `.venv` with the server dependencies, then start the model:

```bash
bash scripts/start-a100-model.sh
```

Start the organizer's grader separately on port 8077. Each run needs a fresh grader with zero previous queries and a three-second check cost. The model runs on port 8000 as `aime-model`.

The question file is JSONL, with `problem_idx` and `problem` on each line. It must not contain answers.

```bash
python -m pip install -r requirements.txt
python run.py answer-first --questions questions.jsonl --output experiments/candidate-811 --seed 811

# Restart model and grader before running the control:
python run.py baseline --questions questions.jsonl --output experiments/baseline-811 --seed 811
```

Use a new output directory for each run. `--model-url`, `--grader-url` and `--deadline` override the defaults.

The launcher uses 30 concurrent problems, a 12,288-token reasoning limit, up to 32 tokens for finalization and a 600-second deadline. Sampling uses temperature 1 and top-p 0.95, with one attempt per problem. Early checks are off. The server uses BF16, a 32,768 context limit, 32 maximum sequences and 8,192 maximum batched tokens; caching, async scheduling and speculation are off.

The score is `time_to_18_s` in `summary.json`. Runs continue past 18 so I can inspect the later verdicts. Model startup is outside the timer; reasoning, finalization and grading are included.

For paired comparisons, use the same seed and settings and alternate which policy runs first. I used baseline first for seed 811 and the selected prompt first for 813.

```bash
python -m unittest discover -s tests
python scripts/review_results.py
```

The audit checks the eight saved runs without a GPU. The experiment controller in `research/` uses the original machine paths and its separately provisioned `fresh_validation` helper. It is a record of that experiment, not a portable service launcher. Use `run.py` for a new run. The original controller is retained beside the results for its recorded hash; the readable version extracts the selection rule without changing the trial plan.
