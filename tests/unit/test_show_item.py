"""show_item.py and save_trajectories on fake run dirs."""
import json
from pathlib import Path

import pytest

import show_item as si
from benchmark.runner_helpers import BenchmarkResult, save_trajectories

U = "aaaa-0001"


def write(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def record(uuid=U, domain="books", calls=(("get_a", {"x": 1}),), answer="42"):
    return {"uuid": uuid, "domain": domain, "status": "success", "error": "", "duration_s": 1.0,
            "output": [{"turn_id": 0, "query": "How many?", "answer": answer,
                        "sequence": {"tool_call": [{"name": n, "arguments": a} for n, a in calls]}}]}


def plan_trajectory(long_result="R" * 500):
    return [
        {"type": "HumanMessage", "content": "How many?"},
        {"type": "AIMessage", "content": "[...]", "plan": ["find a", "count b"], "tool_calls": []},
        {"type": "AIMessage", "content": "Step 1: find a", "step_index": 0, "step": "find a",
         "offered_tools": ["get_a", "get_c"], "no_tool_call": False, "tool_calls": []},
        {"type": "AIMessage", "content": "", "step_index": 0, "attempt": "llm",
         "tool_calls": [{"id": "s0", "name": "get_a", "args": {"x": "foo "}}]},
        {"type": "ToolMessage", "tool_name": "get_a", "tool_call_id": "s0", "result": "[]", "step_index": 0,
         "attempt": "llm", "verdict": "empty", "kept": False},
        {"type": "AIMessage", "content": "", "step_index": 0, "attempt": "strip",
         "tool_calls": [{"id": "s0_r1", "name": "get_a", "args": {"x": "foo"}}]},
        {"type": "ToolMessage", "tool_name": "get_a", "tool_call_id": "s0_r1", "result": long_result,
         "step_index": 0, "attempt": "strip", "verdict": None, "kept": True},
        {"type": "AIMessage", "content": "Step 2: count b | no tool call: hmm", "step_index": 1, "step": "count b",
         "offered_tools": ["get_c"], "no_tool_call": True, "tool_calls": []},
        {"type": "AIMessage", "content": "42", "final": True, "tool_calls": [], "stop_reason": "completed",
         "final_fallback": False, "elapsed_s": 3.0, "n_executions": 2},
    ]


@pytest.fixture
def env(tmp_path):
    run, gold = tmp_path / "output" / "r1", tmp_path / "gold"
    write(run / "books.json", [record()])
    write(run / "books_tools.json", [{"uuid": U, "domain": "books", "query": "How many?",
                                      "all_tools": ["get_a", "get_b", "get_c"],
                                      "shortlisted_tools": ["get_a", "get_c"]}])
    write(gold / "books.json", [{"uuid": U, "domain": "books", "output": [{"turn_id": 0, "query": "How many?",
          "answer": [[42]], "sequence": {"tool_call": [{"name": "get_a", "arguments": {"x": "foo"}},
                                                       {"name": "get_b", "arguments": {"y": 2}}],
                                         "tool_response": [["a"], [[42]]]}}]}])
    split = tmp_path / "split.json"
    write(split, {"capability_3": {"dev_domains": ["genes"]}})
    return {"run": run, "gold": gold, "split": split,
            "args": [str(run), U, "--gold-dir", str(gold), "--split", str(split)]}


def test_without_trajectory_uses_tools_union(env, capsys):
    si.main(env["args"])
    out = capsys.readouterr().out
    assert "results/plan/verdicts not saved" in out
    assert "get_a({\"x\": 1})" in out
    assert "GOLD TOOLS NOT OFFERED [union of offered tools" in out and ": get_b" in out
    assert "GOLD TOOLS CALLED: get_a=yes, get_b=no" in out
    assert "GOLD ANSWER: [[42]]" in out


def test_with_trajectory_shows_plan_steps_attempts(env, capsys):
    write(env["run"] / "books_trajectory.json", [{"uuid": U, "domain": "books", "trajectory": plan_trajectory()}])
    si.main(env["args"])
    out = capsys.readouterr().out
    assert "PLAN:\n  1. find a\n  2. count b" in out
    assert "STEP 1: find a\n  offered: get_a, get_c" in out
    assert "[llm] get_a({\"x\": \"foo \"}) verdict=empty" in out
    assert "[strip] get_a({\"x\": \"foo\"}) verdict=ok KEPT" in out
    assert "no tool call: hmm" in out
    assert "FINAL: stop_reason=completed" in out
    assert "[per-step offered tools (trajectory)]: get_b" in out
    assert "R" * 200 + "...(+300)" in out and "R" * 201 not in out


def test_cut():
    assert si.cut("x" * 10, 5) == "xxxxx...(+5)"
    assert si.cut({"a": 1}) == '{"a": 1}'


def test_frozen_and_dev_domain_files_not_opened(env, capsys, monkeypatch):
    write(env["run"] / "disney.json", [record(uuid="f1", domain="disney")])
    write(env["run"] / "genes.json", [record(uuid="d1", domain="genes")])
    opened = []
    real = si.by_uuid
    monkeypatch.setattr(si, "by_uuid", lambda p: (opened.append(Path(p).name), real(p))[1])
    si.main(env["args"][:2] + ["f1", "d1"] + env["args"][2:])
    out = capsys.readouterr().out
    assert "disney.json" not in opened and "genes.json" not in opened
    assert "f1: not found" in out and "d1: not found" in out
    assert "skipped unopened domain files: disney, genes" in out


def test_dev_domain_allowed_with_flag(env, capsys):
    write(env["run"] / "genes.json", [record(uuid="d1", domain="genes")])
    si.main([str(env["run"]), "d1", "--gold-dir", str(env["gold"]), "--split", str(env["split"]), "--allow-dev"])
    assert "=== d1  [genes]" in capsys.readouterr().out


def test_seen_uuid_refused(env):
    with pytest.raises(SystemExit, match="seen"):
        si.main([str(env["run"]), "10050cfaf017-32fe3953461a"])


def test_save_trajectories_writes_side_file(tmp_path):
    r = BenchmarkResult(uuid=U, domain="books", query="q", status="success", trajectory=plan_trajectory())
    save_trajectories([r], tmp_path)
    data = json.loads((tmp_path / "books_trajectory.json").read_text())
    assert data == [{"uuid": U, "domain": "books", "status": "success", "trajectory": plan_trajectory()}]
    assert not (tmp_path / "books.json").exists()
