#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
export HF_HOME="$PWD/models/huggingface"
export VLLM_NO_USAGE_STATS=1
mkdir -p logs models

# Capture the exact model revision and download it before starting timed work.
.venv/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
repo = 'WeiboAI/VibeThinker-3B'
revision_file = Path('models/revision.txt')
revision = '77bd2cced09193c8b9a59a32bd8577bbd1f3e01c'
revision_file.write_text(revision + '\n')
print('Model revision:', revision, flush=True)
snapshot_download(repo, revision=revision, allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model'])
PY

cache_flag=--no-enable-prefix-caching
schedule_flags=()
if [[ "${AIME_ASYNC_SCHEDULING:-0}" == 1 ]]; then
  schedule_flags+=(--async-scheduling)
fi
if [[ "${AIME_NGRAM_TOKENS:-0}" != 0 ]]; then
  schedule_flags+=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${AIME_NGRAM_TOKENS},\"prompt_lookup_min\":3,\"prompt_lookup_max\":5}")
fi
if [[ "${AIME_PREFIX_CACHE:-0}" == 1 ]]; then
  cache_flag=--enable-prefix-caching
fi
exec .venv/bin/vllm serve WeiboAI/VibeThinker-3B \
  --revision "$(cat models/revision.txt)" \
  --served-model-name aime-model \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 \
  --max-num-seqs "${AIME_MAX_SEQS:-32}" --max-num-batched-tokens "${AIME_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm --disable-log-requests "$cache_flag" "${schedule_flags[@]}"
