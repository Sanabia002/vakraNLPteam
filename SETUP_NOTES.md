# Setup notes (capability 3)

## Environment that works
- Ubuntu 26.04, Python 3.14.4, Intel Arc B580 (12 GB), Docker (docker.io + compose v2)
- Packages: see `requirements_working.txt` (mcp==2.3.0, litellm==1.106.6, langgraph==1.2.18, langchain-core==1.7.1)
- Clone, then: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[init]" && pip install -r requirements_benchmark.txt && pip install langchain-litellm`
- `export TMPDIR=~/tmp` before big pip installs if /tmp is small.

## Data
- `make download` (about 35 GB). Only the public TRAIN split downloads without gated access.
- The runner hardcodes `data/test/...`. `make download` leaves an empty `data/test`; replace it:
  `rmdir data/test && ln -s train data/test`.
  **Remove this link (`rm data/test`) before downloading the real test set.** Label all runs on this data as train/dev.

## Containers
`make build && docker compose up -d` -> 4 containers (capability_1..4). Cap 3 uses `capability_3_multihop_reasoning`.

## Model server (local)
llama.cpp build b11960 (Vulkan), model bartowski/Qwen_Qwen3.5-4B-GGUF:Q4_K_M:
`./llama-server -hf bartowski/Qwen_Qwen3.5-4B-GGUF:Q4_K_M --alias qwen3.5-4b --jinja -ngl 99 -c 16384 -n 1024 --reasoning off --port 8080 --api-key "$LITELLM_API_KEY"`
`.env` needs `LITELLM_BASE_URL=http://localhost:8080/v1` (lowercase /v1) and `LITELLM_API_KEY=...`. Never commit `.env` or logs.

## Known pitfalls (cost us days)
1. **mcp 2.x renamed `inputSchema` -> `input_schema`.** `agents/mcp_tool_wrapper.py` silently skipped the schema, so the model saw tools with only `args`/`config` and every call failed validation ("... is a required property"). Fixed in the wrapper (accepts both names). All agent runs before the fix are invalid. `mcp 1.30` conflicts with litellm 1.106.6 (ImportError CacheableResult).
2. Without `-n`, a model can generate until the timeout. We use `-n 1024`.
3. Thinking models burn tokens; we serve with `--reasoning off`.
4. `--top-k-tools 40` needs about 18.8k tokens of prompt (> 16k context).
5. The `*_tools.json` "shortlisted_tools" is computed by the runner, not by the agent; it does not reflect the ORACLE_TOOLS experiment override.

## Running and logging
`python run_logged.py --out output/NAME --cap 3 --domain a,b --model openai/qwen3.5-4b --top-k 20 --max-iter 12 --temperature 0 --thinking off --build b11960 --notes "..."` appends a row to `runs_log.csv`. Use a new `--out` each time. Dev split: `python make_dev_split.py` -> `splits/dev_split.json` (seed 0, hockey excluded). Do not tune on dev domains.

## Scoring
- Proxy (NOT the VAKRA metric): `python proxy_score.py output/RUN`
- Official-format: `evaluator/evaluator.py` with `JUDGE_BACKEND=groq` and `API_KEY=<groq key>` (key kept in `~/.groq_key`, never in git).
- Retrieval study: `python shortlist_recall.py` (needs `tool_dumps/` from list_tools.py).
