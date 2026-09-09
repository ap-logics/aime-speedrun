"""Profile a solver run without changing its inference requests."""

import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path("/home/azureuser/aime-speedrun")


def snapshot(evidence, label):
    raw = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=3).read().decode()
    (evidence / (label + ".prom")).write_text(raw)
    values = {}
    for line in raw.splitlines():
        if line.startswith("vllm:"):
            key = line.split("{")[0].split()[0]
            if key.endswith(("_total", "_sum", "_count")) or key in (
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
            ):
                values[key] = values.get(key, 0) + float(line.rsplit(" ", 1)[1])
    return values


def run_with_monitor(root, evidence, arguments):
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm",
            "--format=csv",
            "-l",
            "1",
            "-f",
            str(evidence / "gpu.csv"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        result = subprocess.run(["bash", "scripts/run-prompt-experiment.sh", *arguments], cwd=root)
    finally:
        monitor.terminate()
        monitor.communicate(timeout=10)
    return result.returncode


def make_record(command, before, after):
    return {
        "server_command": command,
        "metrics_before": before,
        "metrics_after": after,
        "idle_verified": after["vllm:num_requests_running"]
        == after["vllm:num_requests_waiting"]
        == 0,
        "delta": {
            k: v - before.get(k, 0)
            for k, v in after.items()
            if k.endswith(("_total", "_sum", "_count"))
        },
    }


def main():
    arguments = sys.argv[1:]
    name = arguments[0]
    evidence = ROOT / "execution-evidence" / name
    evidence.mkdir(parents=True, exist_ok=False)
    before = snapshot(evidence, "before")
    assert before["vllm:num_requests_running"] == before["vllm:num_requests_waiting"] == 0
    pid = int((ROOT / "logs/model.pid").read_text())
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "--no-enable-prefix-caching" in command.split()
    exit_code = run_with_monitor(ROOT, evidence, arguments)
    time.sleep(12)  # Let serving statistics flush after the timed run.
    after = snapshot(evidence, "after")
    record = make_record(command, before, after)
    (evidence / "profile.json").write_text(json.dumps(record, indent=2))
    print(f"Profile complete for {name}; idle_verified={record['idle_verified']}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
