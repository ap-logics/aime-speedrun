# Time log

I estimate that I spent approximately four hours of active work across these sessions. This is a retrospective estimate, not a stopwatch measurement. It covers setup, implementation, reviewing experiments and preparing the submission. I did not record separate durations for each activity. Waiting between sessions and unattended GPU runs are excluded.

| Date | Work |
|---|---|
| Sep 5â€“6 | Tried the supplied VM, then a university A4000 while the VM was unreachable. Worked through SSH access and container setup. |
| Sep 7 | Set up the replacement A100, driver and vLLM. Built the runner, question export and initial tests. |
| Sep 7 | Established the baseline and tested reasoning limits, finalisation, early checks, overlap, prefix caching, multiple samples and serving settings. An additional serving screen took 16m32s; neither tested candidate qualified for confirmation. |
| Sep 8 | Compared policies on fresh questions, then ran six new paired seeds for priced early checks. The confirmation took 56m11s and gave mixed results. |
| Sep 9 | Screened three prompt variants against the original on AIME 2024, then evaluated the selected prompt on AIME 2022. All eight trials took 35m09s. |
| Sep 9 | Reviewed the traces, checked the runnable submission, prepared slides and reduced the repository to the final experiment evidence. |

The dates show when work happened, not continuous working days. GPU runtimes include unattended execution and should not be added to the four-hour estimate, since some active work also took place while experiments ran.
