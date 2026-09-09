"""Restart the model between completed trials on the original machine."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path("/home/azureuser/aime-speedrun")


def check_no_solver():
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            words = path.read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        assert (
            b"src/speedrun.py" not in words
            and b"src/prompt_solver.py" not in words
            and b"scripts/dynamic_solver.py" not in words
        ), "Solver still active"


def stop_model(root):
    pid = int((root / "logs/model.pid").read_text())
    path = Path(f"/proc/{pid}/cmdline")
    words = path.read_bytes().split(b"\0")
    assert b"/home/azureuser/aime-speedrun/.venv/bin/vllm" in words and b"serve" in words
    assert words[words.index(b"--port") + 1] == b"8000"
    os.kill(pid, signal.SIGTERM)
    for _ in range(120):
        if not path.exists() or not path.read_bytes():
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("Model did not stop")
    return pid, b" ".join(words).decode()


def start_model(root, label):
    env = dict(
        os.environ,
        AIME_MAX_SEQS="32",
        AIME_BATCHED_TOKENS="8192",
        AIME_PREFIX_CACHE="0",
        AIME_ASYNC_SCHEDULING="0",
        AIME_NGRAM_TOKENS="0",
        HF_HUB_OFFLINE="1",
    )
    with (root / "logs" / ("clean-restart-" + label + ".log")).open("w") as log:
        process = subprocess.Popen(
            ["bash", "scripts/start-a100-model.sh"],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    (root / "logs/model.pid").write_text(str(process.pid))
    return process


def wait_until_idle(process):
    for _ in range(480):
        assert process.poll() is None, "Model exited during startup"
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            raw = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=2).read().decode()
            gauges = {}
            for line in raw.splitlines():
                if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
                    k = line.split("{")[0]
                    gauges[k] = gauges.get(k, 0) + float(line.rsplit(" ", 1)[1])
            assert gauges["vllm:num_requests_running"] == gauges["vllm:num_requests_waiting"] == 0
            break
        except Exception:
            time.sleep(0.5)
    else:
        raise RuntimeError("Fresh server not ready and idle")


def main():
    previous, label = sys.argv[1:]
    assert previous.startswith(("clean01-", "confirm01-", "prompt01-"))
    assert label.replace("-", "").replace("_", "").isalnum()
    assert (ROOT / "experiments" / previous / "summary.json").is_file()
    check_no_solver()
    pid, command = stop_model(ROOT)
    record = {"preceding_run": previous, "old_pid": pid, "old_command": command}
    process = start_model(ROOT, label)
    wait_until_idle(process)
    evidence = ROOT / "research/prompt-comparison-01" / ("restart-" + label + ".json")
    record.update(
        new_pid=process.pid,
        new_command=Path(f"/proc/{process.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(),
        idle_verified=True,
    )
    evidence.write_text(json.dumps(record, indent=2))
    print("Fresh server ready for " + label, flush=True)


if __name__ == "__main__":
    main()
