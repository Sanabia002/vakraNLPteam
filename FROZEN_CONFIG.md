# Frozen configuration: candidate F (decided Nov 11 by the rule in results/freeze_rule.txt; output in results/decide_final2.txt)
Code frozen at git tag final-config = 70a1d6e (2026-11-14). This document and the run/score scripts were added after the tag
without touching agents/ or benchmark_runner.py (run_set.sh aborts if they differ from the tag).

## System
F = D with fallback. Per question: run D; if D's status is not "success" (step limit, context overflow, API error, timeout), use C's result (merge_fallback.py).
- D: LangGraphReActAgent, RETRIEVER=hybrid BASELINE_GROUND_ANSWER=1 SAVE_TRAJECTORY=1, --top-k 30 --max-iter 12 --temperature 0.
  Model qwen/qwen3.8-27b on Groq (Preview) via LiteLLM, base URL https://api.groq.com/openai/v1,
  LLM_EXTRA_KWARGS={"extra_body":{"reasoning_effort":"none"},"max_tokens":1024}. Pool run done Nov 11 2026. Hosted, NOT deterministic.
- C: PlanExecuteAgent, PLAN_AGENT=1 STEP_RETRIEVAL=1 VERIFY=1 GROUND_PROMPT=1 RETRIEVER=hybrid SAVE_TRAJECTORY=1, 12 iterations, temperature 0.
  Model Qwen3.5-4B Q4_K_M (bartowski/Qwen_Qwen3.5-4B-GGUF), llama.cpp b11960: -c 32768 -n 1024 --cache-ram 1024 --reasoning off.
- Retriever: hybrid MiniLM (all-MiniLM-L6-v2, max over question clauses) + BM25 (rank-bm25 0.2.2), reciprocal rank fusion k=60; tool text = name words + description + parameters.
- Plain baseline for comparison: provided ReAct agent, --top-k 20, no flags, Qwen3.5-4B on the same local server.
- Procedure: run_set.sh dev|test.

## Evidence
Pool (191 tune questions, evaluator VALID, 0 unjudged): A 74, D 97, E 84, F 104; B 70, C 69. Rule: best score; ties within 3 -> earlier of A, D, E, F.
E's score was seen before E was declared (disclosed in results/freeze_rule.txt).

## Data use
Tune pool (191 q, 15 domains) for all choices. Dev (82 q): retrieval recall confirmation once, then the reporting runs after this freeze.
Frozen course test (bike_share_1, computer_student, disney, olympics, restaurant, student_loan): after this freeze only, two systems, one run each.
Disclosures: two one-question smoke tests on bike_share_1 after the test freeze; early tune run base_tune75_k20 used frozen domains and was quarantined, never scored.

## Environment
Python 3.14.4; packages in results/pip_freeze_final.txt and requirements_working.txt (mcp==2.3.0; agents/mcp_tool_wrapper.py fix required).
Judge: gpt-oss-120b on Groq (JUDGE_BACKEND=groq). Evaluator patch: Judge Error f-string fix.
Rules: no edits to agents/ or the runner after the tag. Reruns only for infrastructure failures caught by validate_run.py before scoring, never after seeing a score.
