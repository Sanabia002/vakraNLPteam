"""subtask4: rule parsing over the real training rules, the split and its guard, run configs."""
import hashlib
import json
import shutil

import pytest

from subtask4 import data
from subtask4.config import RunConfig
from subtask4.rules import CATEGORIES, Rule, Source, UnknownRule, is_retriever, parse_rule

pytestmark = pytest.mark.skipif(not data.TRAIN.exists(), reason="needs data/train")

DOCS_TOPIC = ("If a user's query pertains to Sports & Athletics, which is/are about Topics covering various sports, "
              "athletes, games, and statistics from different leagues, make sure you try answering them by only "
              "using document retrievers. Do not use other types of tools.")
APIS_TOPIC = ("If a user's query pertains to Health & Medicine, which is/are about Topics that pertain to health data, "
              "medical conditions, surveys, and synthetic patients, make sure you do not use document retrievers to "
              "try answering those questions. Use other types of tools.")


# ---------- rules ----------

def test_the_four_templates():
    assert parse_rule("Use document retrievers to answer questions. Do not use any other type of tool.") == Rule(Source.DOCS)
    assert parse_rule("Do not use document retrievers to answer questions. Use other types of tools.") == Rule(Source.APIS)
    assert parse_rule(DOCS_TOPIC) == Rule(Source.DOCS, "Sports & Athletics")
    assert parse_rule(APIS_TOPIC) == Rule(Source.APIS, "Health & Medicine")
    assert parse_rule("  " + DOCS_TOPIC.replace(" ", "\n ", 3)) == Rule(Source.DOCS, "Sports & Athletics")
    assert parse_rule(None) is None and parse_rule(" ") is None


def test_unknown_rule_is_an_error():
    with pytest.raises(UnknownRule):
        parse_rule("Only use tools whose names start with get_.")


def test_every_training_rule_outside_the_test_split_parses():
    kinds = {}
    for domain in data.domains("tune") + data.domains("dev"):
        for item in data.inputs(domain).values():
            rule = parse_rule(item["additional_instructions"])
            if rule:
                kinds[rule.kind] = kinds.get(rule.kind, 0) + 1
                assert rule.category is None or rule.category in CATEGORIES
    assert set(kinds) == {"docs", "apis", "docs|topic", "apis|topic"}


def test_retriever_names():
    assert is_retriever("query_books") and is_retriever("retriever_books")
    assert not is_retriever("get_books_by_author")


# ---------- split ----------

def test_split_partitions_the_domains():
    tune, dev, test = (set(data.domains(s)) for s in data.SPLIT_NAMES)
    assert tune | dev | test == set(data.all_domains())
    assert not (tune & dev or tune & test or dev & test)
    assert "hockey" in tune
    sizes = {s: sum(len(data._load("input", d)) for d in data.domains(s)) for s in data.SPLIT_NAMES}
    assert sizes == {"tune": 559, "dev": 151, "test": 188}


def test_frozen_domains_are_refused_outside_a_final_run():
    frozen = sorted(data.test_domains())[0]
    with pytest.raises(data.FrozenSplitError):
        data.gold(frozen)
    with pytest.raises(data.FrozenSplitError):
        data.check_access(["books", frozen])
    data.check_access([frozen], final=True)


def test_a_changed_test_split_is_detected(tmp_path, monkeypatch):
    shutil.copytree(data.SPLITS, tmp_path, dirs_exist_ok=True)
    split = json.loads((tmp_path / "course_test_split.json").read_text())
    split["capability_4"]["dev_domains"].remove("sales_in_weather")
    (tmp_path / "course_test_split.json").write_text(json.dumps(split))
    monkeypatch.setattr(data, "SPLITS", tmp_path)
    with pytest.raises(data.FrozenSplitError):
        data.test_domains()


# ---------- config ----------

def test_run_id_follows_behaviour_not_notes(tmp_path):
    a, b = RunConfig("x", router=True), RunConfig("x", router=True, notes="first try")
    assert a.run_id == b.run_id and a.run_id.startswith("x-")
    assert RunConfig("x").run_id != a.run_id
    assert RunConfig("x", split="dev", router=True).run_id != a.run_id


def test_uuids_file_enters_the_run_id_by_content(tmp_path):
    path = tmp_path / "u.json"
    path.write_text('["a"]')
    first = RunConfig("x", uuids_file=str(path)).run_id
    path.write_text('["a", "b"]')
    assert RunConfig("x", uuids_file=str(path)).run_id != first
    assert RunConfig("x", uuids_file=str(path)).settings()["uuids_file"] == hashlib.sha256(b'["a", "b"]').hexdigest()[:12]


def test_config_files_load_and_override_split():
    for path in (data.REPO / "subtask4" / "configs").glob("*.yaml"):
        config = RunConfig.load(path)
        assert RunConfig.load(path, split="dev").split == "dev"
        assert config.name


def test_oracle_is_refused_on_test():
    with pytest.raises(ValueError):
        RunConfig("x", split="test", router=True, topics="oracle")
