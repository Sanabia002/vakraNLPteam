"""Plain checks used by the graph's nodes: no model, no tools, so each one is unit-tested on its own.

  is_empty           an API result with no rows (Empty-Result Recovery)
  case_variants      the same arguments in UPPER, Title and lower case
  value_variants     other forms of the values to retry after an empty result (Empty-Result Recovery): letter case
                     (all arguments, then one at a time), date and number formats, two text values swapped, and
                     close spellings among values the agent has already seen
  seen_values        the quoted text values in what the agent has seen (tool results, earlier turns)
  untraceable        text values that appear nowhere in what the agent has seen (Value Check)
  schema_problems    missing required values, values of the wrong type (Value Check)
  passages           the passage texts in a document-search result (Evidence Check)
  citation_found     a cited sentence is in the passages, allowing small copy differences (Evidence Check)
  answer_in          the answer's names and numbers appear in the cited sentences (Evidence Check)
  back_refers        the question points back to earlier turns ("those", "them", ...) (Tool Ranker rewrite)
  list_items         the items of a list-shaped result (Grounded Answer list check)
  parse_json         the first JSON object in a model reply
"""
from __future__ import annotations

import collections
import difflib
import json
import re

STOPWORDS = frozenset("""a an the of in on at to for from by with and or but is are was were be been being do does did
what which who whom whose when where why how many much list name names give show tell me us find all any each this that
these those it its there their they them he she his her as than then so if not no yes can could would should will
please about into over under per""".split())
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
_BACK = re.compile(r"\b(those|these|them|that one|this one|they|their|theirs|it|its|he|she|his|her|the same|"
                   r"among them|of them|the above|mentioned|previous|earlier|former|latter)\b", re.I)


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split()


def is_empty(result: str) -> bool:
    """True for an API result with no rows: [], {}, null, "", or an object whose values are all empty."""
    text = str(result).strip()
    if text in ("", "[]", "{}", "null", "None"):
        return True
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return False
    if isinstance(value, dict):
        return bool(value) and all(v in ([], None, {}, "") for v in value.values())
    return value in ([], None, {}, "")


def case_variants(args: dict) -> list[dict]:
    """The arguments with every text value in UPPER, Title and lower case; variants equal to `args` are dropped."""
    out = []
    for convert in (str.upper, str.title, str.lower):
        variant = {k: convert(v) if isinstance(v, str) else v for k, v in args.items()}
        if variant != args and variant not in out:
            out.append(variant)
    return out


_DATE = re.compile(r"^(\d{2,4})([/-])(\d{1,2})\2(\d{1,4})$")


def format_forms(value: str) -> list[str]:
    """Other common forms of a date or number: zero padding removed or added, / and - swapped, periods dropped."""
    forms, v = [], value.strip()
    match = _DATE.match(v)
    if match:
        a, sep, b, c = match.groups()
        other = "-" if sep == "/" else "/"
        unpad = [str(int(a)) if len(a) <= 2 else a, str(int(b)), str(int(c)) if len(c) <= 2 else c]
        pad = [a if len(a) > 2 else a.zfill(2), b.zfill(2), c if len(c) > 2 else c.zfill(2)]
        for parts in (unpad, pad):
            forms += [sep.join(parts), other.join(parts)]
        forms.append(other.join([a, b, c]))
    elif re.fullmatch(r"0+\d+", v):
        forms.append(v.lstrip("0") or "0")
    if "." in v and not re.fullmatch(r"[\d.]+", v):
        forms.append(re.sub(r"\.\s*", " ", v).replace("  ", " ").strip())
    return [f for f in dict.fromkeys(forms) if f != value]


def seen_values(texts: list[str], limit: int = 20_000) -> list[str]:
    """Quoted text values (3-80 characters) in tool results and earlier turns: the only legitimate source of the
    database's own spellings, since the agent may not read the database directly."""
    out: dict[str, None] = {}
    for text in texts:
        for a, b in re.findall(r"\"([^\"\n]{3,80})\"|'([^'\n]{3,80})'", str(text)):
            out[a or b] = None
            if len(out) >= limit:
                return list(out)
    return list(out)


def close_value(value: str, candidates: list[str], cutoff: float = 0.85) -> str | None:
    """The candidate closest in spelling to `value` (case-insensitive), if close enough and not the same value."""
    lowered = {c.lower(): c for c in candidates if c.strip() and c.lower() != value.lower()}
    match = difflib.get_close_matches(value.lower(), list(lowered), n=1, cutoff=cutoff)
    return lowered[match[0]] if match else None


def value_variants(args: dict, seen: list[str] = (), limit: int = 12) -> list[dict]:
    """Other forms of the values to try when a call returns nothing, most specific first, at most `limit`."""
    out: list[dict] = []

    def add(variant: dict) -> None:
        if variant != args and variant not in out:
            out.append(variant)
    texts = [k for k, v in args.items() if isinstance(v, str) and v.strip()]
    for variant in case_variants(args):                       # every text value in one case
        add(variant)
    for key in texts:                                          # a close spelling the agent has seen
        match = close_value(args[key], list(seen))
        if match:
            add({**args, key: match})
    for key in texts:                                          # date and number formats
        for form in format_forms(args[key]):
            add({**args, key: form})
    if len(texts) == 2:                                        # two text values swapped (first and last name)
        a, b = texts
        add({**args, a: args[b], b: args[a]})
    for key in texts:                                          # one value's case at a time
        for convert in (str.upper, str.title, str.lower):
            add({**args, key: convert(args[key])})
    return out[:limit]


def _lenient(value: str) -> bool:
    """Values the agent builds rather than copies: patterns, dates, numbers, very short strings."""
    v = value.strip()
    return len(v) <= 2 or "%" in v or bool(re.fullmatch(r"[\d\s.,:/+-]+", v))


def untraceable(args: dict, seen_text: str) -> list[str]:
    """Text values whose words do not all appear in what the agent has seen (question, dialogue, results)."""
    seen = set(words(seen_text))
    missing = []
    for value in args.values():
        if isinstance(value, str) and not _lenient(value):
            tokens = [w for w in words(value) if w not in STOPWORDS] or words(value)
            if tokens and not all(t in seen for t in tokens):
                missing.append(value)
    return missing


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


def schema_problems(args: dict, schema: dict) -> tuple[dict, list[str]]:
    """(arguments with harmless type fixes, problems the model must fix). Fixes: "5" -> 5 for an integer field."""
    props, fixed, problems = schema.get("properties", {}), dict(args), []
    for name in schema.get("required", []):
        if fixed.get(name) in (None, ""):
            problems.append(f"the required value '{name}' is missing")
    for name, value in args.items():
        want = props.get(name, {}).get("type")
        if want not in _TYPES or value is None or isinstance(value, _TYPES[want]) and not (
                want in ("integer", "number") and isinstance(value, bool)):
            continue
        try:
            fixed[name] = {"integer": int, "number": float, "string": str}[want](value)
        except (KeyError, ValueError, TypeError):
            problems.append(f"'{name}' should be of type {want}, not {value!r}")
    return fixed, problems


def passages(result: str) -> list[str]:
    """Passage texts in a document-search result (a JSON list of {"text": ...}); the raw text otherwise."""
    try:
        value = json.loads(result)
    except (ValueError, TypeError):
        return [str(result)] if str(result).strip() else []
    if isinstance(value, dict):
        value = value.get("results") or value.get("documents") or [value]
    texts = [v.get("text") or v.get("document") or v.get("content") for v in value if isinstance(v, dict)] \
        if isinstance(value, list) else []
    return [t for t in texts if t] or ([str(result)] if str(result).strip() else [])


def citation_found(sentence: str, texts: list[str], threshold: float = 0.9) -> bool:
    """The best window of each passage, as long as the sentence, shares >= threshold of its words."""
    cited = words(sentence)
    if not cited:
        return False
    need, n = collections.Counter(cited), len(cited)
    for text in texts:
        w = words(text)
        if len(w) < n:
            windows = [w]
        else:
            windows = (w[i:i + n] for i in range(0, len(w) - n + 1))
        for window in windows:
            if sum((need & collections.Counter(window)).values()) >= threshold * n:
                return True
    return False


def answer_in(answer: str, cited: list[str]) -> bool:
    """Every number in the answer, and its names (or half its content words when it has none), are cited."""
    text = " ".join(cited)
    have, have_numbers = set(words(text)), set(_NUMBER.findall(text.replace(",", "")))
    numbers = set(_NUMBER.findall(str(answer).replace(",", "")))
    if numbers - have_numbers:
        return False
    names = [w for w in words(" ".join(re.findall(r"\b[A-Z][\w'-]*", str(answer)))) if w not in STOPWORDS]
    content = [w for w in words(answer) if w not in STOPWORDS and not _NUMBER.fullmatch(w)]
    if names:
        return all(w in have for w in names)
    return not content or sum(w in have for w in content) >= len(content) / 2


def back_refers(question: str) -> bool:
    return bool(_BACK.search(question))


def list_items(result: str) -> list[str]:
    """The items of a list-shaped API result: each row's first value, as text. Empty for anything else."""
    try:
        value = json.loads(result)
    except (ValueError, TypeError):
        return []
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    if not isinstance(value, list) or len(value) < 2:
        return []
    items = []
    for row in value:
        if isinstance(row, dict) and row:
            row = next(iter(row.values()))
        elif isinstance(row, (list, tuple)) and row:
            row = row[0]
        if isinstance(row, (str, int, float)):
            items.append(str(row))
    return items


def parse_json(text: str) -> dict | None:
    """The first JSON object in a model reply (inside ``` fences or bare); None if there is none."""
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S)
    for candidate in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S) + re.findall(r"\{.*\}", text, re.S):
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None
