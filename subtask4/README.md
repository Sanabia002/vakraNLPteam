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

## The Decision Graph

Three systems are compared: **Baseline** (VAKRA's ReAct agent), **Rule Gate** (Tool Filter + refuse at once: the
`p1_p2_refuse` config of the interim report) and the **Decision Graph** (`subtask4/system/`): one LangGraph state graph
in which every decision is made where its information exists. Its components, in three stages, are also its config
switches (`SystemConfig` in `config.py`) and its graph nodes:

| Stage | Component | Switch | What it does |
|---|---|---|---|
| Read | Rule Reader | `rule_reader` | which tools the rule allows; the model reads a rule in new wording |
| | Topic Check | `topic_check` | does a topic rule apply to this question? |
| | Tool Filter | `tool_filter` | hides the tools the rule forbids |
| Act | Tool Ranker | `tool_ranker` | the 20 tools shown at each step |
| | Search on Demand | `search_on_demand` | document search shown only when it fits |
| | Rule Guard | (always on) | blocks a forbidden or made-up tool |
| | Value Check | `value_check` | warns once about values that appear nowhere |
| | Empty-Result Recovery | `empty_recovery` | retries other forms of the values, then a hint |
| | Loop Guard | `loop_guard` | stops repeated calls |
| | Step Budget | `max_iterations` | the step limit (12) |
| | Verified Lookup | `verified_lookup` | a document search inside the loop, fact-checked |
| Answer | Evidence Check | `evidence_check` | documents-only rule: answer from cited sentences, or refuse |
| | Grounded Answer | `grounded_answer` | the final answer rewritten from the tool results |

`spec.py` is the graph as designed: Figure 1 is drawn from it, and `tests/unit/test_s4_system.py` checks the built
graph against it.

| Config | What |
|---|---|
| `decision_graph.yaml` | the final system |
| `rebuild_check.yaml` | every component off: must score like the Rule Gate |
| `ablate_<component>.yaml` | the final system without one component (9 files) |
| `order_test.yaml` | the shown tools shuffled: does the model choose by position? |
| `decision_graph_smoke.yaml` | 6 dialogues each of books, cars, hockey |

Retired names: P1 = Tool Filter, P2 = Refusal Gate (replaced by the Evidence Check), P3 = hybrid tool picker (dropped);
the design document's C1-C15 map to the components above.

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
  configs/        baseline, p1, p1_oracle, p2_refuse, p2_evidence, p1_p2_*, p1_p2_p3, smoke, decision_graph*, rebuild_check, ablate_*, order_test
  deploy/         services.py (start/check the servers), nothink_proxy.py, mcp_native.yaml
  system/         the Decision Graph: spec.py (the design's graph), graph.py (nodes, edges, Evidence Check sub-graph),
                  checks.py (plain checks), prompts.py, agent.py (GraphAgent, for the runner), studio.py

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
