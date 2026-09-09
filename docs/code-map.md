# Code

Start with `run.py`. It sends the fixed settings to one of two solver files:

- `src/speedrun.py`: original prompt.
- `src/prompt_solver.py`: selected prompt.

They differ only in `problem_prompt`. I kept both measured versions so their hashes still match the runs.

Inside either file, `run` manages concurrent generation and a single grader request. `infer` generates the reasoning and asks for a short final answer if it hits the limit. `extract_answer` reads the boxed integer. `main` connects to the services and writes the trace.

The selected policy does not use the older early-check, overlap or multiple-sample branches. Those remain in the measured source.

`scripts/review_results.py` checks the final experiment: source hashes, verdict timing, settings, question order and server restarts.

The prompt controller now separates selection (`select_prompt`) from evaluation order (`evaluation_order`). Its original source is retained as `research/prompt-comparison-01/controller.py`, which the audit checks against the recorded hash. The helper scripts separate model restart, profiling and question export into functions. Each has a short `main()` and can be imported without starting a process or reading a dataset. No new GPU runs were made for this cleanup.
