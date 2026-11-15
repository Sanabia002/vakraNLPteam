"""subtask4.deploy.monitor: reading a run's log and trajectories. Fake run folder; no servers."""
import json

from subtask4.deploy import monitor

LOG = """[10:00:00.000] # DOMAIN: books (3 items)
[10:00:05.000]     Status: success | Tools: 1 | Trajectory steps: 4 | Time: 5.00s
[10:03:00.000]     Status: success | Tools: 2 | Trajectory steps: 9 | Time: 175.00s
[10:08:00.000]     Status: timeout after 300.0s
"""


def fake_run(tmp_path):
    run = tmp_path / "system-abc"
    run.mkdir()
    (run / "run.log").write_text(LOG)
    (run / "run.json").write_text(json.dumps({"status": "running"}))
    calls = [{"type": "AIMessage", "tool_calls": [{"name": "t", "args": {"a": 1}}]}] * 3
    trajectories = [
        {"uuid": "1", "trajectory": [{"type": "Subtask4Decision", "mode": "none"}, *calls,
                                     {"type": "Subtask4Log", "events": [{"node": "model", "forced": True},
                                                                        {"node": "guard", "unknown": True}]}]},
        {"uuid": "2", "trajectory": [{"type": "Subtask4Decision", "mode": "docs"},
                                     {"type": "Subtask4Log", "events": []}]},
    ]
    (run / "books_trajectory.json").write_text(json.dumps(trajectories))
    (run / "books.json").write_text(json.dumps([{"output": [{"answer": "24"}]}, {"output": [{"answer": ""}]}]))
    return run


def test_run_progress(tmp_path):
    p = monitor.run_progress(fake_run(tmp_path))
    assert (p["done"], p["timeouts"], p["errors"], p["slow"], p["items_started"]) == (2, 1, 0, 1, 3)


def test_behaviour(tmp_path):
    b = monitor.behaviour(fake_run(tmp_path))
    assert b["dialogues"] == 2 and b["forced at step limit"] == 1 and b["made-up tools per dialogue"] == 1
    assert b["3+ identical calls in a row"] == 1 and b["empty answers"] == 1
    assert b["path none"] == 1 and b["path docs"] == 1


def test_parse_ollama_ps():
    text = """NAME                 ID              SIZE      PROCESSOR          CONTEXT    UNTIL
qwen3.5-4b:latest    d8b0f5e9760c    4.2 GB    100% GPU           32768      Forever
gpt-oss:120b         cd490ef50ef3    64 GB     12%/88% CPU/GPU    32768      Forever
"""
    assert monitor.parse_ollama_ps(text) == {"qwen3.5-4b": "100% GPU", "gpt-oss:120b": "12%/88% CPU/GPU"}
