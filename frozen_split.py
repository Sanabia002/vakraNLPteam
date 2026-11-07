"""Domains and questions that must never enter a tuning sample (capability 3).

FROZEN_TEST: held-out test domains.
SEEN_UUIDS: questions already looked at during development; excluded from tuning and test reporting.
  10050cfaf017-32fe3953461a (bike_share_1): ran in two PlanExecuteAgent smoke runs on 2026-11-06
  (output/plan_smoke_bike1_v2, _v3) and its trajectory was inspected, before bike_share_1 was frozen.
"""
import glob
import json
import os
from typing import Dict, Iterable

FROZEN_TEST = {"bike_share_1", "computer_student", "disney", "olympics", "restaurant", "student_loan"}
SEEN_UUIDS = {"10050cfaf017-32fe3953461a"}


def uuid_domains(cap_dir: str = "data/train/capability_3_multihop_reasoning") -> Dict[str, str]:
    """uuid -> domain, from the input files."""
    out = {}
    for f in glob.glob(os.path.join(cap_dir, "input", "*.json")):
        for it in json.load(open(f)):
            out[it["uuid"]] = os.path.basename(f)[:-5]
    return out


def assert_no_frozen(uuids: Iterable[str], domain_of: Dict[str, str]) -> None:
    """Fail if any uuid belongs to a frozen test domain, is a seen question, or has no known domain."""
    uuids = list(uuids)
    unknown = [u for u in uuids if u not in domain_of]
    frozen = sorted({(domain_of[u], u) for u in uuids if u in domain_of and domain_of[u] in FROZEN_TEST})
    seen = sorted(set(uuids) & SEEN_UUIDS)
    assert not unknown, f"{len(unknown)} uuids have no known domain, e.g. {unknown[:3]}"
    assert not frozen, f"{len(frozen)} uuids belong to frozen test domains, e.g. {frozen[:3]}"
    assert not seen, f"seen uuids in sample: {seen}"
