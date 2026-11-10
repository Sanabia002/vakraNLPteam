"""subtask4: the scoreboard reproduces the intermediate report; topic prompt and labels; the judge cache."""
import os
from pathlib import Path

import pytest
from langchain_core.outputs import Generation

from subtask4 import data, scoreboard, topics
from subtask4.evaluate import SQLiteCache
from subtask4.rules import CATEGORIES

# The team's stored dev runs live next to the repository (08_VARKA/results); TEAM9_RESULTS points elsewhere.
RESULTS = Path(os.environ.get("TEAM9_RESULTS", data.REPO.parent / "results")) / "subtask4"
needs_results = pytest.mark.skipif(not RESULTS.exists() or not data.TRAIN.exists(), reason="needs stored runs and data")


def table(run: str, evaluation: str):
    return scoreboard.rows(RESULTS / evaluation, RESULTS / "predictions" / run)


@needs_results
def test_scoreboard_reproduces_the_intermediate_report():
    k20 = scoreboard.summary(table("dev4_k20_full", "eval_dev4_k20_full_selfjudge.json"))
    fix = scoreboard.summary(table("dev4_k20_policyfix", "eval_dev4_k20_policyfix_selfjudge.json"))
    assert (round(100 * k20["official"], 1), round(100 * k20["refusal_checked"], 1)) == (47.7, 41.7)
    assert (round(100 * fix["official"], 1), round(100 * fix["refusal_checked"], 1)) == (50.3, 42.4)
    assert [round(100 * x, 1) for x in k20["official_ci"]] == [39.9, 55.6]
    assert k20["causes"] == {"correct": 63, "credited, never refused": 9, "should refuse, answered": 14,
                             "agent error": 1, "right data, judged ungrounded": 8, "wrong tool": 40,
                             "right tools, wrong arguments": 16}
    assert k20["refused_when_should"] == [0, 23]


@needs_results
def test_paired_comparison_matches_the_report():
    result = scoreboard.compare(table("dev4_k20_full", "eval_dev4_k20_full_selfjudge.json"),
                                table("dev4_k20_policyfix", "eval_dev4_k20_policyfix_selfjudge.json"))
    assert round(100 * result["difference"], 1) == 0.7 and result["p"] == 1.0


def test_statistics():
    assert scoreboard.mcnemar_exact(0, 8) == pytest.approx(0.0078125)
    assert scoreboard.mcnemar_exact(0, 0) == 1.0
    lo, hi = scoreboard.wilson(72, 151)
    assert (round(lo, 3), round(hi, 3)) == (0.399, 0.556)


def test_topic_prompt_and_answer():
    prompt = topics.build_prompt("hockey", ["query_hockey", "get_b", "get_a"])
    assert "'hockey'" in prompt and "get_a, get_b" in prompt and "query_hockey" not in prompt
    assert all(name in prompt for name in CATEGORIES)
    names = list(CATEGORIES)
    assert topics.parse_answer("10") == names[9]
    assert topics.parse_answer("<think>maybe 3</think> 7.") == names[6]
    assert topics.parse_answer("none") is None and topics.parse_answer("13") is None


@pytest.mark.skipif(not data.TRAIN.exists(), reason="needs data/train")
def test_stored_labels_are_the_tune_gold_labels():
    assert topics.load_labels() == topics.gold_labels(data.domains("tune"))
    with pytest.raises(data.FrozenSplitError):
        topics.gold_labels(sorted(data.test_domains()))


def test_judge_cache_round_trip(tmp_path):
    cache = SQLiteCache(tmp_path / "c.sqlite")
    assert cache.lookup("p", "judge") is None
    cache.update("p", "judge", [Generation(text="score: 1")])
    again = SQLiteCache(tmp_path / "c.sqlite")
    assert again.lookup("p", "judge")[0].text == "score: 1"
    assert again.lookup("p", "other judge") is None
