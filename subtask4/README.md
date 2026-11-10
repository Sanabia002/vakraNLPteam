# Subtask 4 (Team 9)

VAKRA's ReAct agent, plus three parts that each decide one thing before the model sees the question. Each part has
its own switch in the run config, so its effect can be measured on its own.

| Part | Decides | Hypothesis | Setting |
|---|---|---|---|
| Rule + topic | which rule applies (topic rules: the model names the domain's category once) | | always on when P1 or P2 is |
| **P1 router** | hides the tools an applicable rule forbids | H7: enforcing a rule beats stating it | `router: true` |
| **P2 gate** | under an applicable documents-only rule: refuse at once, or search once and refuse if the documents do not answer | H3: search that can say "nothing found" | `gate: refuse` / `evidence` |
| **P3 picker** | the tool shortlist (Melissa's dense + BM25 retriever) | H4 (partly) | `picker: hybrid` |

All parts off gives the intermediate report's baseline (20 tools, rule delivered for single-turn input).

## Layout

```
subtask4/
  rules.py        the four rule templates and the 12 topic categories
  topics.py       domain -> category (model, zero-shot); tune gold labels to check it
  agent.py        Subtask4Agent: the decision, P2 gate, P1 router, P3 picker
  config.py       RunConfig: one YAML file per run; run ID = name + hash of the settings
  data.py         tune / dev / course-test split; the frozen test split is SHA-256 checked and locked
  run.py          runs a config through VAKRA's runner in-process and writes the run record
  evaluate.py     VAKRA's evaluator with a chosen judge and a persistent judge cache
  scoreboard.py   per-dialogue table, summary and paired comparison
  configs/        baseline, p1, p1_oracle, p2_refuse, p2_evidence, p1_p2_*, p1_p2_p3, smoke
  deploy/         services.py (start/check the servers), nothink_proxy.py, mcp_native.yaml
```

Changes to VAKRA's code (both behaviour-preserving, tested): `benchmark_runner.AGENT_FACTORY` and a `domain` argument
to `_get_agent`; `_select_tools` and `_input_messages` hooks on `LangGraphReActAgent`.

## Running

```bash
python -m subtask4.deploy.services start                                   # Ollama, proxy, REST, retriever
python -m subtask4 topics-check --base-url http://localhost:8080/v1 --model qwen3.5-4b
python -m subtask4 run subtask4/configs/smoke.yaml --allow-dirty --mcp-config subtask4/deploy/mcp_native.yaml
python -m subtask4 run subtask4/configs/p1.yaml --split dev --mcp-config subtask4/deploy/mcp_native.yaml
python -m subtask4 eval <run_id> --judge local --mcp-config subtask4/deploy/mcp_native.yaml
python -m subtask4 score <run_id>
python -m subtask4 compare <baseline_run_id> <run_id>
```

Each run writes `output/subtask4/<run_id>/`: predictions, shortlists, trajectories (first entry: the agent's
decision), `run.log`, `run.json`, then `eval_<judge>.json` and `score_<judge>.json`. Copy finished runs to the
project's `results/subtask4/runs/`.

## Reproducibility

- **One config, one run.** Every behaviour-changing setting is in the YAML file; the run ID hashes them.
- **Code version.** `run.json` records the commit; a run refuses uncommitted changes unless `--allow-dirty`
  (smoke runs only), and then records a hash of the difference.
- **Environment.** Python and package versions, and the model server's models and Ollama digests.
- **Frozen test set.** `splits/course_test_split.json` is checked against its SHA-256 on every use; its domains
  can only be run or scored with `--final`, and the oracle can never touch them.
- **Judge.** Replies are cached by exact prompt and judge settings, so re-scoring gives the same verdicts.
- **No GPU to rebuild numbers.** `score` and `compare` read only stored files. A test checks that the scoreboard
  reproduces the intermediate report (47.7 / 41.7 at k=20) from the stored runs.

## Tests

`python -m pytest tests/unit -k s4` (31 tests: rule parsing over all non-test training rules, the split and its
guard, run IDs, router, gate, rule delivery, the runner seam, the scoreboard against the report, the judge cache).

## What the data says (tune pool unless noted)

- Every rule is one of four templates; a topic rule often names a category unrelated to the domain.
- Under an applicable documents-only rule the gold answer is a refusal in 83 of 85 tune dialogues (and 23 of 23 in
  dev). `gate: refuse` is therefore a strong, simple policy; `evidence` must beat it to matter.
- Baseline agent (k=20, dev): breaks an applicable documents-only rule in 20 of 23 dialogues; 18 of 23 with the
  rule delivered.
- The scorer gives automatic credit to a refusal only when no tool was called: a refusal after searching goes to
  the judges.

## Known issues

- Upstream `_parse_json_tool_call` crashes on a bare-number answer (e.g. "1990"); `Subtask4Agent` guards it.
- The MCP server needs `mcp<2` in `.venv-env`; the runner side needs `mcp==2.3.0` (Melissa's wrapper fix).
- Topic classifier with a 3B stand-in (llama3.2): 14/17 tune domains; to be measured with Qwen.
