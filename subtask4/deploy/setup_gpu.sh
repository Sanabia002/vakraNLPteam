#!/usr/bin/env bash
# One-time setup of a fresh GPU machine (Thunder "ollama" template, Ubuntu) for Subtask 4 runs.
# Usage: bash setup_gpu.sh [branch]     -> ~/vakra on that branch (default roshan/subtask4), ready for
#        python -m subtask4.deploy.services start
set -euo pipefail
BRANCH="${1:-roshan/subtask4}"
cd ~

# 1. Code: the team repo at the branch's latest commit (runs record the commit they ran from).
[ -d vakra ] || git clone -q --branch "$BRANCH" https://github.com/Sanabia002/vakraNLPteam.git vakra
cd vakra

# 2. Ollama with the settings services.py uses; then the agent's model (aliased to the name the runner uses)
#    and the self-hosted judge. Pulls run while the environments install and the data downloads.
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=-1 \
  setsid nohup ollama serve > ~/ollama.log 2>&1 < /dev/null &
until curl -sf localhost:11434/api/tags > /dev/null; do sleep 1; done
# The judge (65 GB) must load without mmap: with mmap, a machine with less RAM than the model thrashes and the load
# never finishes (seen on an A100 with 62 GB RAM, 2026-11-12).
( ollama pull qwen3.5:4b && ollama cp qwen3.5:4b qwen3.5-4b && ollama pull gpt-oss:120b \
  && printf 'FROM gpt-oss:120b\nPARAMETER use_mmap false\n' > ~/judge.Modelfile \
  && ollama create gpt-oss:120b -f ~/judge.Modelfile ) > ~/pull.log 2>&1 &
PULLS=$!

# 3. Python: the agent side as on the team's machines (Python 3.14, pinned), the servers apart (mcp 1.x).
uv venv -q -p 3.14 .venv
uv pip install -q -p .venv/bin/python -r requirements_working.txt pytest
uv venv -q -p 3.12 .venv-env
uv pip install -q -p .venv-env/bin/python fastapi "uvicorn[standard]" chromadb sentence-transformers "mcp<2" \
  httpx pyyaml pandas python-dotenv jsonref prometheus-fastapi-instrumentator huggingface_hub

# 4. Data: the public VAKRA release (Subtasks 3-4 questions, databases, document index; ~35 GB).
.venv-env/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ibm-research/VAKRA", repo_type="dataset", local_dir="data",
                  allow_patterns=["train/capability_3_multihop_reasoning/*", "train/capability_4_multiturn/*",
                                  "databases/*", "indexed_documents/*"])
EOF

wait "$PULLS"  # the model pulls (a bare `wait` would also wait for ollama serve)
.venv/bin/python -m pytest -q tests/unit -k s4 --ignore tests/unit/test_initialize_active_data.py  # imports a missing module
echo "SETUP DONE"
