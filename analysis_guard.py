"""Domain guards shared by the analysis scripts (show_item.py, compare_runs.py).

Frozen test domains and seen uuids are always refused. Dev domains are refused unless allow_dev.
"""
import json
import os
from typing import Iterable, Set

from frozen_split import FROZEN_TEST, SEEN_UUIDS

FROZEN_ALL = set(FROZEN_TEST) | {"sales_in_weather"}  # + capability 4 frozen domain
DEFAULT_SPLIT = "splits/dev_split.json"


def dev_domains(split_path: str = DEFAULT_SPLIT) -> Set[str]:
    if not os.path.exists(split_path):
        return set()
    return set(json.load(open(split_path)).get("capability_3", {}).get("dev_domains", []))


def domain_allowed(domain: str, dev: Set[str], allow_dev: bool) -> bool:
    return domain not in FROZEN_ALL and (allow_dev or domain not in dev)


def check(domains: Iterable[str], uuids: Iterable[str], dev: Set[str], allow_dev: bool) -> None:
    """Raise SystemExit if any domain is frozen (or dev without allow_dev) or any uuid was seen."""
    domains, uuids = set(domains), set(uuids)
    if domains & FROZEN_ALL:
        raise SystemExit(f"refused: frozen test domain(s) {sorted(domains & FROZEN_ALL)}")
    if not allow_dev and domains & dev:
        raise SystemExit(f"refused: dev domain(s) {sorted(domains & dev)} (pass --allow-dev only if asked)")
    if uuids & SEEN_UUIDS:
        raise SystemExit(f"refused: seen uuid(s) {sorted(uuids & SEEN_UUIDS)}")
