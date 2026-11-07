"""compare_runs.py on fake evaluator JSONs."""
import json

import pytest

import compare_runs as cr


def dialogue(uuid, score, gt=2, query="q", judge_error=False):
    expl = {"answer": "Judge Error: timeout" if judge_error else "fine", "exactmatch": "", "groundedness": ""}
    return {"uuid": uuid, "score": score,
            "details": {"per_turn": [{"query": query + " " + uuid, "score": score,
                                      "metadata": {"gt_steps": gt, "score_explanation": expl}}]}}


def eval_json(tmp_path, name, domains):
    p = tmp_path / name
    p.write_text(json.dumps({"domains": {d: {"dialogues": dl} for d, dl in domains.items()}, "summary": {}}))
    return str(p)


@pytest.fixture
def split(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"capability_3": {"dev_domains": ["genes"]}}))
    return str(p)


@pytest.fixture
def pair(tmp_path):
    a = eval_json(tmp_path, "a.json", {
        "books": [dialogue("u1", 1.0, 2), dialogue("u2", 1.0, 3), dialogue("u3", 0.0, 2), dialogue("u4", 0.0, 2)],
        "cars": [dialogue("u5", 1.0, 2), dialogue("u6", 0.0, 3), dialogue("ua", 1.0, 2)]})
    b = eval_json(tmp_path, "b.json", {
        "books": [dialogue("u1", 1.0, 2), dialogue("u2", 0.0, 3), dialogue("u3", 1.0, 2), dialogue("u4", 0.0, 2)],
        "cars": [dialogue("u5", 1.0, 2, judge_error=True), dialogue("u6", 1.0, 3), dialogue("ub", 1.0, 2)]})
    return a, b


def test_flip_counts_exclusion_and_unpaired(pair):
    res = cr.compare(cr.load(pair[0]), cr.load(pair[1]))
    assert res["excluded"] == ["u5"]
    assert res["only_a"] == ["ua"] and res["only_b"] == ["ub"]
    assert dict(res["overall"]) == {"both": 1, "A_only": 1, "B_only": 2, "neither": 1}
    assert dict(res["by_gt_steps"][3]) == {"A_only": 1, "B_only": 1}
    assert dict(res["by_gt_steps"][2]) == {"both": 1, "B_only": 1, "neither": 1}
    assert dict(res["by_domain"]["cars"]) == {"B_only": 1}


def test_judge_error_in_a_only_also_excludes(tmp_path):
    a = eval_json(tmp_path, "a.json", {"books": [dialogue("u1", 0.0, judge_error=True)]})
    b = eval_json(tmp_path, "b.json", {"books": [dialogue("u1", 1.0)]})
    res = cr.compare(cr.load(a), cr.load(b))
    assert res["excluded"] == ["u1"] and res["rows"] == []


def test_threshold(tmp_path):
    a = eval_json(tmp_path, "a.json", {"books": [dialogue("u1", 0.5)]})
    b = eval_json(tmp_path, "b.json", {"books": [dialogue("u1", 0.5)]})
    assert dict(cr.compare(cr.load(a), cr.load(b), 1.0)["overall"]) == {"neither": 1}
    assert dict(cr.compare(cr.load(a), cr.load(b), 0.5)["overall"]) == {"both": 1}


def test_report_lists_flips_with_query_prefix(pair, split, capsys):
    cr.main([pair[0], pair[1], "--split", split])
    out = capsys.readouterr().out
    assert "A right -> B wrong (A_only): 1" in out and "u2  [books, gt=3]  q u2" in out
    assert "A wrong -> B right (B_only): 2" in out
    assert "EXCLUDED (Judge Error): u5" in out


@pytest.mark.parametrize("dom", ["disney", "sales_in_weather"])
def test_frozen_domain_refused(tmp_path, split, dom):
    a = eval_json(tmp_path, "a.json", {dom: [dialogue("u1", 1.0)]})
    with pytest.raises(SystemExit, match="frozen"):
        cr.main([a, a, "--split", split])


def test_dev_domain_refused_unless_allowed(tmp_path, split, capsys):
    a = eval_json(tmp_path, "a.json", {"genes": [dialogue("u1", 1.0)]})
    with pytest.raises(SystemExit, match="dev domain"):
        cr.main([a, a, "--split", split])
    cr.main([a, a, "--split", split, "--allow-dev"])
    assert "paired kept: 1" in capsys.readouterr().out


def test_seen_uuid_refused(tmp_path, split):
    a = eval_json(tmp_path, "a.json", {"books": [dialogue("10050cfaf017-32fe3953461a", 1.0)]})
    with pytest.raises(SystemExit, match="seen"):
        cr.main([a, a, "--split", split])
