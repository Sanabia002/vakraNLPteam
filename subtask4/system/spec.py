"""The new system's graph, as designed: the single description that Figure 1 is drawn from
(src/analysis/s4_system_graph.py) and that tests/unit/test_s4_graph_structure.py checks the built graph against.

Every box is a node of the built graph, under the same name. Diamonds (kind "decision") are conditional edges, not
nodes. The "evidence" cluster is one sub-graph, added to the main graph as the single node "evidence".
"input" and "output" are the graph's START and END.
"""
from __future__ import annotations

# id: (cluster, kind, title, detail)   kind: io | before | inside | decision | after
NODES = {
    "input":       (None, "io", "Dialogue + tool rule", "earlier turns, the last question, the rule text"),
    "rule_reader": ("before", "before", "Rule reader", "4 sentence patterns; the model classifies new wording"),
    "topic_check": ("before", "before", "Topic check", "is this question about the rule's topic?"),
    "which_rule":  ("before", "decision", "Which rule applies?", ""),
    "tool_filter": ("before", "before", "Tool filter", "APIs-only rule: API tools only\nno rule: API tools + the document lookup"),

    "retriever":   ("evidence", "inside", "Retriever (Qwen)", "search the domain's own documents, 1-3 phrasings;\npropose an answer and cite up to 3 sentences"),
    "code_check":  ("evidence", "inside", "Code check", "cited sentences found in the passages (fuzzy);\nthe answer appears in them"),
    "critic":      ("evidence", "inside", "Critic (same Qwen, fresh conversation)", "supported, including scope?\nif not, name what is missing"),
    "supported":   ("evidence", "decision", "Code check and critic\nboth say supported?", ""),
    "verified":    ("evidence", "inside", "Verified answer", "with its citations"),
    "nothing":     ("evidence", "inside", "Nothing verified", "after the last search round"),

    "subgoal":     ("loop", "inside", "1. Sub-goal for this hop", "the question (or its standalone rewrite)\n+ what earlier hops found"),
    "rank":        ("loop", "inside", "2. Rank the options", "API tools and the document lookup compete\nfor the 20 places shown"),
    "model":       ("loop", "inside", "3. Model", "picks one shown option, or writes the answer"),
    "guard":       ("loop", "inside", "4. Rule guard", "is the choice allowed?"),
    "api_or_docs": ("loop", "decision", "API or\ndocuments?", ""),
    "arg_check":   ("loop", "inside", "5a. Argument check", "values present, right type, traceable"),
    "run_api":     ("loop", "inside", "6a. Run the API", "recorded as a tool call"),
    "empty":       ("loop", "decision", "7a. Empty\nresult?", ""),
    "hint":        ("loop", "inside", "Hint", "try another form or tool; if still nothing,\nanswer “no matching records”"),
    "merge":       ("loop", "inside", "Result joins what is known", ""),

    "answer_docs": (None, "after", "Answer from the cited sentences", ""),
    "refuse":      (None, "after", "Refuse: “I can not answer”", ""),
    "grounded":    ("loop", "after", "After the loop: grounded final answer", "rewritten from tool results only;\nevery list item kept"),
    "output":      (None, "io", "Output", "answer or refusal, every tool call,\nand every component's decision (logged)"),
}

# (from, to, label, style)   style: main | back | caller
EDGES = [
    ("input", "rule_reader", "", "main"),
    ("rule_reader", "topic_check", "", "main"),
    ("topic_check", "which_rule", "", "main"),
    ("which_rule", "retriever", "documents-only rule", "main"),
    ("which_rule", "tool_filter", "APIs-only rule,\nor no rule", "main"),
    ("tool_filter", "subgoal", "", "main"),

    ("retriever", "code_check", "", "main"),
    ("code_check", "critic", "", "main"),
    ("critic", "supported", "", "main"),
    ("critic", "retriever", "missing: search again\n(at most 2 more)", "back"),
    ("supported", "verified", "yes", "main"),
    ("supported", "nothing", "no", "main"),
    ("verified", "answer_docs", "documents-only rule", "caller"),
    ("nothing", "refuse", "documents-only rule", "caller"),
    ("verified", "merge", "lookup: verified fact", "caller_free"),
    ("nothing", "merge", "lookup: \u201cnot found\u201d", "caller_free"),

    ("subgoal", "rank", "", "main"),
    ("rank", "model", "", "main"),
    ("model", "guard", "picks a tool", "main"),
    ("guard", "model", "forbidden: blocked,\nthe model is told", "back"),
    ("guard", "api_or_docs", "allowed", "main"),
    ("api_or_docs", "arg_check", "API", "main"),
    ("api_or_docs", "retriever", "documents:\nthe sub-question", "caller_free"),
    ("arg_check", "model", "untraceable value:\none warning", "back"),
    ("arg_check", "run_api", "", "main"),
    ("run_api", "empty", "", "main"),
    ("empty", "run_api", "retry variant: same API,\nother letter cases (once)", "back"),
    ("empty", "hint", "yes", "main"),
    ("empty", "merge", "no", "main"),
    ("hint", "merge", "", "main"),
    ("merge", "subgoal", "next hop", "back"),
    ("model", "grounded", "writes the answer", "free"),
    ("merge", "grounded", "", "invisible"),

    ("answer_docs", "output", "", "main"),
    ("refuse", "output", "", "main"),
    ("grounded", "output", "", "main"),
]


SUBGRAPH = "evidence"


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
