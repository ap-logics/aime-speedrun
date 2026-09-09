#!/usr/bin/env bash
# Run from the remote project. Arguments: run-name, followed by solver options.
set -euo pipefail
cd /home/azureuser/aime-speedrun
name=${1:?Supply a unique run name}
shift
if [[ ! "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo 'Run name must contain letters, digits, underscores, or hyphens' >&2
  exit 1
fi
if [[ -e "experiments/$name" ]]; then
  echo 'Run already exists; choose a new name' >&2
  exit 1
fi

# Only stop the grader process started and recorded by this harness.
if [[ -f logs/grader.pid ]]; then
  grader_pid=$(cat logs/grader.pid)
  if [[ "$grader_pid" =~ ^[0-9]+$ ]] && kill -0 "$grader_pid" 2>/dev/null; then
    if ! tr '\0' ' ' < "/proc/$grader_pid/cmdline" | grep -q '/home/azureuser/private-grader/server.py'; then
      echo 'Recorded PID is not our grader; refusing to stop it' >&2
      exit 1
    fi
    kill "$grader_pid"
    for attempt in {1..50}; do
      if ! kill -0 "$grader_pid" 2>/dev/null; then break; fi
      sleep 0.1
    done
  fi
fi
nohup /home/azureuser/private-grader/.venv/bin/python \
  /home/azureuser/private-grader/server.py \
  > logs/grader-stdout.log 2>&1 < /dev/null &
echo "$!" > logs/grader.pid
for attempt in {1..100}; do
  if curl -fsS http://127.0.0.1:8077/health > /dev/null 2>&1; then break; fi
  sleep 0.2
done

exec .venv/bin/python src/prompt_solver.py \
  --questions "${AIME_QUESTIONS:-data/questions.jsonl}" --output "experiments/$name" "$@"
