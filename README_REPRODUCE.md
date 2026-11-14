# Reproducing the final dev and test results

This covers the frozen system **F** and the plain **baseline** on the dev set (82 questions) and the frozen course
test set. The configuration itself is defined in `FROZEN_CONFIG.md`; this file is the how-to. Code is frozen at git
tag `final-config`; `run_set.sh` refuses to run if `agents/` or `benchmark_runner.py` differ from that tag.

No secrets are in this repository. Keys live in files outside it (see "Keys").

## 1. Setup

- Ubuntu, Python 3.14.4, Docker (with compose v2). GPU used: Intel Arc B580, 12 GB.
- Python environment:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -e ".[init]" && pip install -r requirements_benchmark.txt && pip install langchain-litellm
  pip install rank-bm25==0.2.2
  ```
  Exact versions: `results/pip_freeze_final.txt` (also `requirements_working.txt`). `mcp` must be **2.3.0**;
  `agents/mcp_tool_wrapper.py` contains a required fix for mcp 2.x (`inputSchema` was renamed `input_schema`).
- Containers: `make build && docker compose up -d` gives 4 containers (capability_1..4). `run_set.sh` checks that at
  least 4 are running.
- Evaluation sets: `python make_eval_sets.py` writes `tmp_gt_dev/`, `tmp_gt_test/` (ground truth for the scorer) and
  `results/dev_uuids.json`, `results/test_uuids.json` (the question lists the runner and validator use).
  Dev domains come from `splits/dev_split.json`; test is the six frozen course domains.

## 2. Local model server (used by C and the baseline)

llama.cpp build b11960, model `bartowski/Qwen_Qwen3.5-4B-GGUF:Q4_K_M`, served with the frozen settings
(context 32768, at most 1024 generated tokens, reasoning off) and logged to `~/llama_server.log`:

```bash
./llama-server -hf bartowski/Qwen_Qwen3.5-4B-GGUF:Q4_K_M --alias qwen3.5-4b --jinja -ngl 99 \
  -c 32768 -n 1024 --cache-ram 1024 --reasoning off --port 8080 --api-key "$LITELLM_API_KEY" \
  > ~/llama_server.log 2>&1
```

`run_set.sh` reads that log for `n_ctx_slot = 32768` and aborts if the server was started with another context size.
(`SETUP_NOTES.md` shows the earlier 16384-context command; the frozen runs use 32768.)

## 3. Keys

Three secrets, none of them in git:

| What | Where it lives | Used by |
|---|---|---|
| Local server key | `LITELLM_API_KEY` in `.env` (with `LITELLM_BASE_URL=http://localhost:8080/v1`, lowercase `/v1`) | C, baseline, server health check |
| Groq key for the agent (D) | `~/.groq_pilot_key` (one line, the key only) | `run_set.sh`, D runs |
| Groq key for the judge | `~/.groq_key` | `score_dev.sh`, `score_test.sh` (exported as `API_KEY`, with `JUDGE_BACKEND=groq`) |

`.env` and the key files are read by the scripts at run time. Never commit them, print them, or paste them anywhere.
Keep the key files readable only by you (`chmod 600`).

## 4. Running dev or test

```bash
./run_set.sh dev     # then, only after dev has validated and been scored:
./run_set.sh test
```

Each call runs, in order, all through `run_logged.py` (one row per run in `runs_log.csv`):

1. `final_<set>_D`: ReAct agent, hybrid retriever, grounded final answer, `--top-k 30 --max-iter 12 --temperature 0`,
   `qwen/qwen3.8-27b` on Groq.
2. `final_<set>_C`: plan-execute agent (all plan flags, hybrid retriever) on the local 4B model.
3. `final_<set>_F` = `merge_fallback.py final_<set>_D final_<set>_C`: D's answer, or C's where D's status is not
   `success`. Written to `results/merge_log_final_<set>_F.json`.
4. `final_<set>_baseline`: the provided ReAct agent, `--top-k 20`, no flags, local 4B model.

Before starting, it aborts if: a feature flag is already set in your shell; the `final-config` tag is missing;
`agents/` or the runner changed since the tag; tracked files have uncommitted changes; the eval sets are missing;
the local server is not at 32768 context or not answering; the Groq key or connection fails; fewer than 4 containers
are up. Before and after each run it checks that run's backend again.

Progress goes to `runset_<set>.out`; `runset_<set>.done` appears when everything finished. To stop between runs,
`touch STOP_RUNSET`. Per-run logs are `<run>.log`; outputs are in `output/<run>/`.

For `test`, the script sets `ALLOW_FROZEN=1` so the validator accepts the frozen domains. Per `FROZEN_CONFIG.md`,
test is run **once**, for these two systems only, after the freeze.

## 5. Scoring

```bash
./score_dev.sh  final_dev_F  final_dev_baseline
./score_test.sh final_test_F final_test_baseline
```

Each run is scored by `evaluator/evaluator.py` (judge `gpt-oss-120b` on Groq) into `results/eval_<run>.json`, then
checked by `check_eval.py`. If any item hit a judge error, the script waits 60 s, re-judges only those items
(`rejudge_dev.py` / `rejudge_test.py`) and writes `results/eval_<run>_merged.json`. A run that is already scored and
`VALID` is skipped. Report the merged file when one exists.

Rule (from `FROZEN_CONFIG.md`): rerun only for infrastructure failures caught by `validate_run.py` **before** scoring,
never after seeing a score.

## 6. What the validator messages mean

`validate_run.py RUN N` (run automatically after each run) prints `OK` or `PROBLEM: ...` with one or more of:

| Message | Meaning | What to do |
|---|---|---|
| `OK` | All checks passed | Continue |
| `items X or sample mismatch` | The run's questions are not exactly the N questions in `SAMPLE_FILE` (missing, extra or different) | Infrastructure failure: fix and rerun before scoring |
| `frozen domains [...]` | Output contains a frozen test domain (or hockey) and `ALLOW_FROZEN` is not set | On dev or tune this is a leak: stop. On test, `run_set.sh` sets `ALLOW_FROZEN=1` |
| `K errors other than step limit/timeout` | More than `MAX_OTHER_ERR` (8 in `run_set.sh`) failed items whose error is not a step limit or timeout, e.g. API errors, crashed tool calls | Check the backend; rerun before scoring if it was infrastructure |
| `E/S successful items with empty answers` | Over 30% of successful items returned an empty answer | Usually a broken model server or prompt; investigate |
| `C/S successful items with no tool calls` | Over 30% of successful items made no tool call | Usually broken tool schemas (see the mcp note) or a server ignoring tools |

After each run `run_set.sh` also logs counts of schema errors (`is a required property`, tool schemas not reaching
the model), context-window errors (`ContextWindowExceeded`) and 429 rate-limit lines.

`check_eval.py FILE` prints the number of turns, judge errors (with the five most common), `n_correct / n_samples`,
and finally `VALID` (no judge errors: the score can be reported) or `NOT VALID: judge errors present; do not report
this score` (judge failures silently score 0).

## 7. Known caveats

- D runs on a hosted model (Groq preview) and is **not deterministic**; reruns can differ.
- Groq's free tier limits tokens per minute and per day; 429s on D or judge errors while scoring are expected and
  are handled by the validator and the re-judge step.

## Relaunching after a failure
`run_set.sh` skips any stage whose output folder already passes `validate_run.py` ("skip ...: already valid" in `runset_<set>.out`).
A stage folder that exists but fails validation makes the script abort; delete only that stage's folder (`output/final_<set>_<D|C|baseline>`)
and relaunch. Never delete or rerun a stage after seeing its score: the test protocol (results/freeze_rule.txt) allows reruns only for
infrastructure failures found before scoring. `MAX_OTHER_ERR` (default 8) can be raised from the shell for hosted-API 400s.
