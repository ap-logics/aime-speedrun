# Prompt experiment

I wanted to test whether better instructions could save reasoning without adding more checks or scheduling logic. GEPA motivated the idea, but I only ran a fixed comparison of three prompts. There was no training or iterative prompt search.

Only the prompt changed. The model, reasoning limit, sampling and server settings stayed the same.

## Choosing the prompt

I screened the original and three variants on AIME 2024, seed 801:

| Prompt | Time to 18 | Correct / 30 | Completed output tokens |
|---|---:|---:|---:|
| Original | 168.110s | 23 | 273,306 |
| One clear method | 158.159s | 25 | 268,417 |
| Check progress after each step | 157.607s | 23 | 252,976 |
| Focus on the requested quantity | 111.265s | 24 | 228,526 |

The rule was to choose the fastest variant that reached 18 and beat the original. That selected `answer_first`:

> Identify the exact integer the problem asks for, then choose the shortest sound route to determine it. Compute only what is needed for that result and avoid unrelated intermediate quantities. Give your final integer answer in \boxed{}.

The name means identifying what needs to be calculated, not guessing an answer first.

## Separate evaluation

I froze the choice before exporting AIME 2022 questions, then ran two paired seeds:

| Seed | Original | Selected prompt | Faster by | Correct: original / selected |
|---|---:|---:|---:|---:|
| 811 | 200.356s | 189.729s | 5.30% | 19 / 19 |
| 813 | 228.448s | 183.410s | 19.71% | 18 / 18 |

Both runs improved, but two seeds on one year are not enough to call this a dependable gain. The second pair only just reached the target. Selection also used only one seed, so I could have picked a different prompt with another seed.

The selected prompt produced fewer output tokens in both evaluation runs. That fits the idea of doing less unnecessary work, but it does not prove why it was faster. I made no further prompt changes after evaluation.

## Checks

Each trial started with fresh model and grader processes. Question order matched within each pair, and I reversed the policy order for the second pair. All eight trace audits passed. The full round took 35m09s.

The evaluation data came from `allenai/aime-2022-2025`, revision `73e1eba765ad5847cdb5d1e2e7aaf7b22b585798`, filtered to 2022. There was no normalized exact-text overlap with the earlier question sets. That does not rule out the model having seen these problems during training.

The selection timestamp, question-export record and source hashes are in `research/prompt-comparison-01/`. Answers stayed in the separate grader; only question text and IDs went to the solver.

```bash
python scripts/review_results.py
```

The [original protocol](prompt-comparison-protocol.md) is kept unchanged because its hash was recorded before the experiment.
