"""The new system's graph, as designed: the single description that Figure 1 is drawn from
(src/analysis/s4_system_graph.py) and that tests/unit/test_s4_graph_structure.py checks the built graph against.

Every box is a node of the built graph, under the same name. Diamonds (kind "decision") are conditional edges, not
nodes. The "evidence_check" cluster is one sub-graph, added to the main graph as the single node "evidence_check".
"input" and "output" are the graph's START and END.
"""
from __future__ import annotations

# id: (cluster, kind, title, detail)   kind: io | read | act | decision | answer
NODES = {
    "input":                 (None, "io", "Dialogue + tool rule", "earlier turns, the last question, the rule text"),
    "rule_reader":           ("read", "read", "Rule Reader", "which tools the rule allows;\nthe model reads a rule in new wording"),
    "topic_check":           ("read", "read", "Topic Check", "is this question about the rule's topic?"),
    "which_rule":            ("read", "decision", "Which rule applies?", ""),
    "tool_filter":           ("read", "read", "Tool Filter", "hides the tools the rule forbids"),

    "retrieve":              ("evidence_check", "act", "Retrieve", "search the domain's own documents (up to 3 phrasings);\npropose an answer and cite up to 3 sentences"),
    "check_citations":       ("evidence_check", "act", "Check Citations", "the cited sentences are in the passages,\nand the answer is in them"),
    "critic":                ("evidence_check", "act", "Critic", "the same model, a fresh conversation:\nsupported, as asked? if not, what is missing"),
    "is_supported":          ("evidence_check", "decision", "Citations and critic\nboth say supported?", ""),
    "supported":             ("evidence_check", "act", "Supported", "the answer, with its citations"),
    "not_found":             ("evidence_check", "act", "Not Found", "nothing verified after the last search"),

    "plan_step":             ("act", "act", "Plan Step", "the question (or its standalone rewrite)\n+ what the last step found"),
    "tool_ranker":           ("act", "act", "Tool Ranker", "the 20 tools shown for this step;\nsearch competes for a place (Search on Demand)"),
    "model":                 ("act", "act", "Model", "calls one shown tool, or answers"),
    "loop_guard":            ("act", "act", "Loop Guard", "the same call again? point to its result;\ntwice: answer now"),
    "rule_guard":            ("act", "act", "Rule Guard", "a forbidden or made-up tool is blocked"),
    "api_or_docs":           ("act", "decision", "API or\ndocuments?", ""),
    "value_check":           ("act", "act", "Value Check", "values present, right type, traceable"),
    "call_api":              ("act", "act", "Call API", "recorded as a tool call"),
    "empty":                 ("act", "decision", "Empty\nresult?", ""),
    "empty_recovery":        ("act", "act", "Empty-Result Recovery", "a hint: another form of the value, or another tool"),
    "add_result":            ("act", "act", "Add Result", "the result joins what is known"),

    "answer_from_documents": (None, "answer", "Answer from Documents", "from the cited sentences"),
    "refuse":                (None, "answer", "Refuse", "\u201cI can not answer.\u201d"),
    "grounded_answer":       ("act", "answer", "Grounded Answer", "rewritten from the tool results only;\nevery list item kept"),
    "output":                (None, "io", "Output", "answer or refusal, every tool call,\nand the decision log"),
}

# (from, to, label, style)   style: main | back | caller | caller_free | free | invisible
EDGES = [
    ("input", "rule_reader", "", "main"),
    ("rule_reader", "topic_check", "", "main"),
    ("topic_check", "which_rule", "", "main"),
    ("which_rule", "retrieve", "documents-only rule", "main"),
    ("which_rule", "tool_filter", "APIs-only rule,\nor no rule", "main"),
    ("tool_filter", "plan_step", "", "main"),

    ("retrieve", "check_citations", "", "main"),
    ("check_citations", "critic", "", "main"),
    ("critic", "is_supported", "", "main"),
    ("critic", "retrieve", "missing: search again\n(at most 2 more)", "back"),
    ("is_supported", "supported", "yes", "main"),
    ("is_supported", "not_found", "no", "main"),
    ("supported", "answer_from_documents", "documents-only rule", "caller"),
    ("not_found", "refuse", "documents-only rule", "caller"),
    ("supported", "add_result", "Verified Lookup: the fact", "caller_free"),
    ("not_found", "add_result", "Verified Lookup: \u201cnot found\u201d", "caller_free"),

    ("plan_step", "tool_ranker", "", "main"),
    ("tool_ranker", "model", "", "main"),
    ("model", "loop_guard", "calls a tool", "main"),
    ("loop_guard", "model", "same call again:\npoint to its result", "back"),
    ("loop_guard", "rule_guard", "new call", "main"),
    ("rule_guard", "model", "forbidden or made-up:\nblocked", "back"),
    ("rule_guard", "api_or_docs", "allowed", "main"),
    ("api_or_docs", "value_check", "API", "main"),
    ("api_or_docs", "retrieve", "documents:\nVerified Lookup", "caller_free"),
    ("value_check", "model", "untraceable value:\none warning", "back"),
    ("value_check", "call_api", "", "main"),
    ("call_api", "empty", "", "main"),
    ("empty", "call_api", "retry other forms\nof the values (once)", "back"),
    ("empty", "empty_recovery", "yes", "main"),
    ("empty", "add_result", "no", "main"),
    ("empty_recovery", "add_result", "", "main"),
    ("add_result", "plan_step", "next step", "back"),
    ("model", "grounded_answer", "answers", "free"),
    ("add_result", "grounded_answer", "", "invisible"),

    ("answer_from_documents", "output", "", "main"),
    ("refuse", "output", "", "main"),
    ("grounded_answer", "output", "", "main"),
]

SUBGRAPH = "evidence_check"


def code_edges() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(main-graph edges, evidence sub-graph edges) that the built graph must have, derived from EDGES:
    decisions collapsed into the edges that pass through them, the evidence cluster collapsed into one node."""
    real = [(a, b) for a, b, _, style in EDGES if style != "invisible"]
    kind = {n: v[1] for n, v in NODES.items()}
    through = {(a, c) for a, b in real if kind[b] == "decision" for b2, c in real if b2 == b}
    edges = {(a, b) for a, b in real if "decision" not in (kind[a], kind[b])} | through
    rename = {"input": "__start__", "output": "__end__"}
    main, sub = set(), set()
    for a, b in edges:
        a, b = rename.get(a, a), rename.get(b, b)
        in_a, in_b = NODES.get(a, (None,))[0] == SUBGRAPH, NODES.get(b, (None,))[0] == SUBGRAPH
        if in_a and in_b:
            sub.add((a, b))
        elif in_b:  # a caller enters the sub-graph at its first node
            main.add((a, SUBGRAPH))
            sub.add(("__start__", b))
        elif in_a:  # the sub-graph's outcome goes back to its caller
            main.add((SUBGRAPH, b))
            sub.add((a, "__end__"))
        else:
            main.add((a, b))
    return main, sub
