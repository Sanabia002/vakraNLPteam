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

## Frozen test domains and disclosures (Nov 6)
- Frozen course test set (S3): bike_share_1, computer_student, disney, olympics, restaurant, student_loan. Never tune, run, or read outputs on these. hockey is analysis-only. Tune pool = remaining non-dev domains.
- Disclosure: after the freeze, Claude Code ran two one-question smoke tests on bike_share_1 (it found a generic bug: tools return JSON objects like {"count": 0}, not bare lists). Nothing was tuned on it; the question may be dropped from the test set.
- An early tune run (output/base_tune75_k20) used a sample that included frozen-test domains and hockey. It was quarantined and not scored.
- Judge runs on Groq's free on-demand tier: a 413 (request over the 8000 tokens/min cap) and a hidden `{e}` error made 2 items unjudgeable on the first pass; they were re-judged into results/eval_tune_plan_merged.json (check_eval.py: VALID). Always run check_eval.py: judge failures score 0 silently.
- Tune comparison (68 q, k=20 baseline vs PlanExecuteAgent with all flags): 16/68 vs 23/68. Tune result only, not a reported score.

## Nov 8: retrieval, server, and runs
- Tier 1 retrieval (retrieval_experiments.py, tune pool 191 q, dev 82 q): hybrid of MiniLM (tool text = words + description + parameters, question clauses, max over clauses) + BM25, fused by reciprocal rank (k=60). All-gold-tools-in-top-20: tune 103 -> 140 of 191, dev 45 -> 58 of 82 (finalist chosen on tune; dev viewed once). Recall, not accuracy. Needs `pip install rank-bm25==0.2.2`.
- Flags: RETRIEVER=hybrid (default base), SAVE_TRAJECTORY=1 (writes <domain>_trajectory.json with plan, per-step offered tools, retriever, retriever_build_s; build ~0.7 s for a 155-tool domain), BASELINE_GROUND_ANSWER=1.
- Model server: llama.cpp b11960, Qwen3.5-4B Q4_K_M, -c 16384 (4 slots, unified KV), -n 1024, --reasoning off, --cache-ram 1024, started with `setsid nohup ... &` so it survives the terminal. An earlier server grew to ~7.9 GiB resident and was OOM-killed on Nov 8 at 19:40; three runs made around then were INVALID (server down, items "succeeded" with no model) and were deleted. After the cap the server stayed at 1.4-2.7 GiB.
- run_hybrid4.sh now checks server health before/after each run and calls validate_run.py (items, frozen domains, error types, empty-answer and zero-tool-call rates). Always run validate_run.py and check_eval.py before using a run.
- Context overflow (16K): 1 item in each ReAct run (different questions). Plan agent: 0. The ReAct loop does not truncate tool results; the plan agent cuts each to 600 chars (confound).
- Runs waiting for evaluator scores (Groq free tier cap of 200k tokens/day blocks them): tune68_baseline_k20_ground, tune68_plan_rep2, tune68_plan_rep3, tune68_hyb_plan, tune68_hyb_plan_nostep, tune68_hyb_base, tune68_hyb_base_ground, and the 24 unjudged items of tune68_abl_norecord.
