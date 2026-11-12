"""Conservative, optional retry policies for VAKRA Subtask 3.

This module is intentionally independent of LangChain and the VAKRA runner.
It does not perform tool calls. It only recommends when/how to retry.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


ID_LIKE_NAMES = {"id", "uuid", "key", "code", "hash", "isbn", "doi", "url", "uri", "year", "date"}


def should_retry(
    verdict: Optional[str],
    *,
    retry_empty: bool = False,
    retry_zero: bool = False,
) -> bool:
    """Retry definite errors by default; treat empty/zero as potentially valid.

    ``verdict`` is the output of Melissa's existing ``classify_result``.
    Change ``retry_empty``/``retry_zero`` only for controlled ablations.
    """
    if verdict == "error":
        return True
    if verdict == "empty":
        return retry_empty
    if verdict == "zero":
        return retry_zero
    return False


def _id_like_parameter(name: str) -> bool:
    parts = set(re.findall(r"[a-z0-9]+", name.lower()))
    return bool(parts & ID_LIKE_NAMES)


def single_argument_variants(
    args: Dict[str, Any], limit: int = 2
) -> List[Tuple[str, Dict[str, Any]]]:
    """Generate up to ``limit`` candidates, changing only one string field each.

    Whitespace trimming is attempted first. Case normalization is restricted to
    alphabetic values and non-ID-like parameter names. Numerical, date-like,
    and non-string fields are never modified automatically.
    """
    if limit <= 0:
        return []
    if not isinstance(args, dict):
        raise TypeError("args must be a dictionary")

    seen = {json.dumps(args, sort_keys=True, default=str)}
    out: List[Tuple[str, Dict[str, Any]]] = []

    # Low-risk transformations before case changes.
    for name, value in args.items():
        if not isinstance(value, str):
            continue
        trimmed = value.strip()
        if trimmed != value:
            updated = {**args, name: trimmed}
            key = json.dumps(updated, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out.append((f"strip:{name}", updated))
                if len(out) >= limit:
                    return out

    for name, value in args.items():
        if not isinstance(value, str) or not value.strip():
            continue
        if _id_like_parameter(name) or any(char.isdigit() for char in value):
            continue
        if not any(char.isalpha() for char in value):
            continue
        for label, transformed in (("title", value.title()), ("lower", value.lower())):
            if transformed == value:
                continue
            updated = {**args, name: transformed}
            key = json.dumps(updated, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append((f"{label}:{name}", updated))
            if len(out) >= limit:
                return out
    return out
