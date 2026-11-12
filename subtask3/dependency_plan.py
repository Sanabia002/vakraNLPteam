"""Optional structured input-dependency hints for the existing VAKRA planner.

This module only parses and formats a plan. It does NOT execute tools, validate
answer correctness, or reject inferred/transformed arguments. Safe to use in a
prompt-only ablation with DEPENDENCY_PLAN=1.
"""

from __future__ import annotations

import json
import re
from typing import Any


DEPENDENCY_PLAN_SYSTEM = (
    "Split the question into the minimum ordered list of database API lookups "
    "(usually 2 or 3; never more than 5). Return ONLY a JSON list of objects. "
    "Each object has a short 'goal' and an 'inputs' list. Each input is an "
    "object with 'name', 'source', and optional 'transform'. 'source' must be "
    "'question' or 'step_N' where N is an EARLIER 1-based step number. "
    "Use 'question' for constraints present in the user's question and "
    "'step_N' for values obtained from that earlier API result. "
    "Use 'transform' to explain necessary formatting, extraction, or "
    "normalization; do NOT assume exact string copying is always valid. "
    "Describe information needed, not guessed API results or tool names. "
    "Do not add validation steps."
)

DEPENDENCY_PLAN_EX_A = json.dumps(
    [
        {
            "goal": "Find the artist with the most awards",
            "inputs": [],
        },
        {
            "goal": "Count albums released by that artist",
            "inputs": [
                {
                    "name": "artist",
                    "source": "step_1",
                    "transform": "extract the returned artist identifier",
                }
            ],
        },
    ],
    ensure_ascii=False,
)


def _parse_json(raw: str) -> Any:
    """Parse JSON, allowing surrounding markdown fences or reasoning tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", str(raw), flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError):
        # Some model backends wrap their JSON in a brief introductory phrase.
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except ValueError:
                pass
        return None


def parse_dependency_plan(raw: str, max_steps: int = 5) -> list[dict[str, Any]]:
    """Return normalized step specs, or [] so caller can use the old parser.

    Bad references are omitted rather than used as hard failures. This is
    source guidance, not provenance verification.
    """
    parsed = _parse_json(raw)
    if isinstance(parsed, dict):
        parsed = parsed.get("steps")
    if not isinstance(parsed, list) or not parsed or max_steps < 1:
        return []

    result: list[dict[str, Any]] = []
    for index, item in enumerate(parsed[:max_steps], start=1):
        if not isinstance(item, dict):
            return []
        goal = item.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            return []
        goal = goal.strip()
        inputs = item.get("inputs", [])
        if not isinstance(inputs, list):
            inputs = []
        valid_inputs: list[dict[str, str]] = []
        for entry in inputs:
            if not isinstance(entry, dict):
                continue
            name, source = entry.get("name"), entry.get("source")
            if not isinstance(name, str) or not name.strip():
                continue
            if source != "question":
                if not isinstance(source, str):
                    continue
                m = re.fullmatch(r"step_([1-9][0-9]*)", source)
                if m is None or int(m.group(1)) >= index:
                    continue
            normalized = {"name": name.strip(), "source": source}
            transform = entry.get("transform")
            if isinstance(transform, str) and transform.strip():
                normalized["transform"] = transform.strip()
            valid_inputs.append(normalized)
        result.append({"goal": goal, "inputs": valid_inputs})
    return result


def dependency_hint(step: dict[str, Any]) -> str:
    """Format soft input-source hints to append to an execution prompt."""
    hints = []
    for entry in step.get("inputs", []):
        name, source = entry["name"], entry["source"]
        origin = "the question" if source == "question" else source.replace("step_", "step ") + "'s result"
        hint = f"{name}: use {origin}"
        if entry.get("transform"):
            hint += f" ({entry['transform']})"
        hints.append("- " + hint)
    if not hints:
        return ""
    return "Input-source guidance (not hard constraints):\n" + "\n".join(hints) + "\n"
