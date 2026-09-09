# Bounded prompt comparison

Fixed before runs on 9 September 2026. This is a small GEPA-inspired comparison, not GEPA: three hand-specified general prompts, no iterative reflection, mutation or Pareto search. Hypothesis: reduce unnecessary reasoning without adding early verification/resumption overhead.

All arms use fixed12k+32finalizer, no early checks, original VibeThinker-3B revision and serving settings, concurrency30, one attempt/sample, temperature1, top-p.95, cache/async/speculation off. Only prompt_style changes. The original prompt remains byte-identical in its branch. The separate prompt_solver.py copy leaves the measured main solver intact. No per-problem prompt adaptation or answer examples.

Screen: AIME2024 (previous development set), seed801, order original/direct/checkpoint/answer_first. Four trials at most600s each, fresh server before each. Select the lowest time-to18 among variants that reach18 and are faster than the original (tie order direct/checkpoint/answer_first). If original misses18 or no variant wins, stop after four and report no qualifying candidate. No retries or prompt editing.

Evaluation: previously unused AIME2022 from allenai/aime-2022-2025 revision73e1eba765ad5847cdb5d1e2e7aaf7b22b585798. Source privately staged; only schema inspected before selection. Freeze choice before exporting2022 question text. Compare original then selected on seed811, selected then original on813. Up to four more trials, max eight overall. No additional runs or overall cutoff. Fresh servers and graders; stop on operational/audit failure. Report all misses, accuracy, paired gains and costs. Two evaluation pairs are limited replication, not conclusive evidence.

Provisioning preserves answer arrays only for the separate trusted grader, never prints or exposes answer/solution fields to the solver or analysis. Export questions+IDs only and check normalized exact overlap with previous question exports; stop on overlap. No proof of absent training contamination. Restore original grader/model at the end. Keep this round separate from earlier priced-check results. Existing submission remains unchanged until the completed review.

## Frozen prompt additions

{
  "direct": "Solve using one clear method. Avoid repeating derivations or rechecking a result already established. Stop when the answer is justified. Give your final integer answer in \\boxed{}.",
  "checkpoint": "Solve the problem step by step. At the end of each major step, assess whether the requested integer is already determined. If it is, finish instead of exploring another approach. Give your final integer answer in \\boxed{}.",
  "answer_first": "Identify the exact integer the problem asks for, then choose the shortest sound route to determine it. Compute only what is needed for that result and avoid unrelated intermediate quantities. Give your final integer answer in \\boxed{}."
}