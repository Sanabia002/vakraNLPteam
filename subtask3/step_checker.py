"""Conservative checks for VAKRA Subtask 3 intermediate API results.

These are structural/provenance checks, not a semantic correctness oracle.
No model, database, MCP server, or external dependency is required.
"""
from dataclasses import dataclass, asdict
import json
from typing import Any, Mapping, Optional


@dataclass
class StepCheck:
    valid: bool
    reason: str
    warnings: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return json.loads(stripped)
        except (ValueError, TypeError):
            return value
    return value


def _collect_dicts(value: Any) -> list[dict]:
    """Inspect nested objects, including JSON strings and MCP content blocks."""
    value = _decode(value)
    found = []
    if isinstance(value, dict):
        found.append(value)
        for child in value.values():
            found.extend(_collect_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_collect_dicts(child))
    return found


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


def _has_field(value: Any, field: str) -> bool:
    """Support dotted fields in nested JSON objects, e.g. 'user.manager_id'."""
    for obj in _collect_dicts(value):
        current = obj
        ok = True
        for part in field.split('.'):
            if not isinstance(current, dict) or part not in current:
                ok = False
                break
            current = _decode(current[part])
        if ok and not _is_empty(current):
            return True
    return False


def _scalar_values(value: Any):
    """Yield primitive leaf values from JSON, MCP blocks, or nested lists."""
    value = _decode(value)
    if isinstance(value, dict):
        for child in value.values():
            yield from _scalar_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _scalar_values(child)
    elif value is not None and not isinstance(value, (dict, list)):
        yield value


def _normalized(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _appears_in_question(value: Any, question: str) -> bool:
    """Literal evidence only; not a semantic/entity-matching system."""
    if value is None or isinstance(value, bool):
        return False
    needle = _normalized(value)
    return bool(needle) and needle in _normalized(question)


def trace_argument_provenance(
    arguments: Mapping[str, Any],
    *,
    question: str = "",
    previous_results: Optional[list[Any]] = None,
) -> dict[str, str]:
    """Classify each argument as question, previous_response, both, or unknown.

    This is an evidence heuristic. It cannot rule out legitimate constants,
    paraphrases, transformations, or tool-derived values.
    """
    seen = list(_scalar_values(previous_results or []))
    origins = {}
    for key, value in arguments.items():
        if isinstance(value, (dict, list)):
            origins[key] = "complex_unchecked"
            continue
        from_question = _appears_in_question(value, question)
        from_tools = any(value == item or _normalized(value) == _normalized(item)
                         for item in seen)
        if from_question and from_tools:
            origins[key] = "question_and_previous_response"
        elif from_question:
            origins[key] = "question"
        elif from_tools:
            origins[key] = "previous_response"
        else:
            origins[key] = "unknown"
    return origins


def check_step(
    result: Any,
    *,
    required_fields: Optional[list[str]] = None,
    next_arguments: Optional[Mapping[str, Any]] = None,
    previous_results: Optional[list[Any]] = None,
    question: str = "",
) -> StepCheck:
    """Validate an API response and conservatively flag unsupported arguments.

    next_arguments refer to the call being checked, and previous_results must
    contain responses from earlier calls (NOT the current call). Unknown sources
    are warnings only. Required fields must be supplied explicitly.
    """
    if _is_empty(result):
        return StepCheck(False, "Empty API response", [])

    decoded = _decode(result)
    if isinstance(decoded, str) and decoded.strip().lower().startswith((
        "error:", "exception:", "traceback (most recent call last):"
    )):
        return StepCheck(False, "API returned an error message", [])

    for obj in _collect_dicts(decoded):
        if obj.get("isError") is True or obj.get("is_error") is True:
            return StepCheck(False, "API returned an error flag", [])
        if obj.get("error") not in (None, "", False):
            return StepCheck(False, "API returned an error field", [])

    missing = [field for field in (required_fields or []) if not _has_field(decoded, field)]
    if missing:
        return StepCheck(False, "Missing required fields: " + ", ".join(missing), [])

    warnings = []
    if next_arguments:
        origins = trace_argument_provenance(
            next_arguments, question=question, previous_results=previous_results
        )
        for name, origin in origins.items():
            if origin == "unknown":
                warnings.append(
                    f"Argument {name}={next_arguments[name]!r} has no literal match "
                    "in the question or earlier API responses; manual review needed"
                )
    return StepCheck(True, "ok", warnings)
