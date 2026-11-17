"""subtask4.system: the new system's agent graph. Structure against the design (spec.py), every path through the
graph, and the plain checks. A fake model answers each prompt by its role; fake tools; no servers."""
import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from subtask4.config import RunConfig, SystemConfig
from subtask4.system import checks, spec
from subtask4.system.graph import REFUSAL, System, initial_state

DOCS = "Use document retrievers to answer questions. Do not use any other type of tool."
APIS = "Do not use document retrievers to answer questions. Use other types of tools."
DOCS_SPORTS = ("If a user's query pertains to Sports & Athletics, which is/are about Topics covering various sports, "
               "athletes, games, and statistics from different leagues, make sure you try answering them by only "
               "using document retrievers. Do not use other types of tools.")
PASSAGE = "Stanley Cup. The Montreal Canadiens have won the Stanley Cup 24 times, more than any other team."
REBUILD = SystemConfig(rule_reader="patterns", topic_check="domain", tool_filter="rule", tool_ranker="once",
                       search_on_demand="off", evidence_check="refuse", verified_lookup=False, value_check=False,
                       empty_recovery="off", loop_guard=False, grounded_answer=False)  # = configs/rebuild_check.yaml


def tool(name, reply, log):
    def fn(query: str = "", team: str = "") -> str:
        log.append((name, query or team))
        return reply(query or team) if callable(reply) else reply
    return StructuredTool.from_function(fn, name=name, description=f"{name} tool")


class FakeLLM:
    """Answers each prompt by its role. `steps` scripts the agent's think-act steps: each is a tool call
    (name, args) or a final answer (str)."""

    def __init__(self, steps=(), retriever=None, critic=True, topic="10"):
        self.steps, self.retriever, self.critic, self.topic = list(steps), retriever, critic, topic
        self.bound, self.prompts = None, []

    def bind_tools(self, tools):
        self.bound = [t.name for t in tools]
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        text = str(messages[-1].content)
        self.prompts.append(text)
        if "Reply with JSON only: {\"answer\"" in text:
            return AIMessage(content=json.dumps(self.retriever or {"answer": None, "citations": []}))
        if "You check another assistant's work" in text:
            return AIMessage(content=json.dumps({"supported": self.critic, "missing": "" if self.critic else "the year"}))
        if "Write a short keyword search query" in text or "Write one new search query" in text:
            return AIMessage(content="stanley cup wins")
        if "Which ONE category" in text:
            return AIMessage(content=self.topic)
        if "Rewrite the last question" in text:
            return AIMessage(content="How many Stanley Cups have the Canadiens won?")
        if "step limit" in text:
            return AIMessage(content="I could not finish.")
        if "repeating calls" in text:
            return AIMessage(content="Stopped.")
        step = self.steps.pop(0) if self.steps else "done"
        if isinstance(step, str):
            return AIMessage(content=step)
        name, args = step
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{len(self.prompts)}"}])


class Topics:
    def __init__(self, category):
        self.category_ = category

    async def category(self, domain, names):
        return self.category_


def run(rule, llm, config=SystemConfig(), api_reply="[]", search_reply=None, history=(), max_steps=20,
        question="How many Stanley Cups have the Canadiens won?", topics="Sports & Athletics"):
    log = []
    search_reply = search_reply or json.dumps([{"text": PASSAGE}])
    tools = [tool("get_team_wins", api_reply, log), tool("query_hockey", search_reply, log),
             tool("query_books", search_reply, log)]
    system = System(llm=llm, tools=tools, domain="hockey", config=config, topics=Topics(topics), max_steps=max_steps)
    state = initial_state(question=question, history=list(history), rule_text=rule,
                          messages=[HumanMessage(content=question)])
    final = asyncio.run(system.build().ainvoke(state, config={"recursion_limit": system.recursion_limit()}))
    return final, log


def nodes(final):
    return [e["node"] for e in final["log"]]


# ---------------------------------------------------------------- structure

def test_built_graph_matches_the_design():
    app = System(llm=None, tools=[], domain="hockey", config=SystemConfig(), topics=None).build()
    main = {(e.source, e.target) for e in app.get_graph().edges}
    sub = {(e.source.split(":", 1)[1], e.target.split(":", 1)[1]) for e in app.get_graph(xray=True).edges
           if e.source.startswith("evidence_check:") and e.target.startswith("evidence_check:")}
    want_main, want_sub = spec.code_edges()
    assert main == want_main
    assert sub == want_sub


def test_every_box_in_the_design_is_a_node():
    app = System(llm=None, tools=[], domain="hockey", config=SystemConfig(), topics=None).build()
    built = set(app.get_graph(xray=True).nodes)
    for name, (cluster, kind, *_) in spec.NODES.items():
        if kind not in ("decision", "io"):
            assert (f"evidence_check:{name}" if cluster == "evidence_check" else name) in built, name


# ---------------------------------------------------------------- documents-only rule

def test_documents_answer_gives_a_cited_answer():
    llm = FakeLLM(retriever={"answer": "24 times", "citations": ["The Montreal Canadiens have won the Stanley Cup 24 times"]})
    final, log = run(DOCS, llm)
    assert final["answer"] == "24 times" and final["mode"] == "docs"
    assert [c["tool_name"] for c in final["tool_calls"]] == ["query_hockey", "query_hockey"]  # as asked + keywords
    assert "answer_from_documents" in nodes(final)


def test_documents_without_the_answer_refuse_after_searching_again():
    final, log = run(DOCS, FakeLLM(retriever={"answer": None, "citations": []}))
    assert final["answer"] == REFUSAL
    assert nodes(final).count("retrieve") == 3  # first round + 2 more (critic+research)
    assert final["tool_calls"]  # every search is reported


def test_made_up_citation_fails_the_code_check():
    llm = FakeLLM(retriever={"answer": "30 times", "citations": ["The Canadiens won it 30 times."]})
    final, _ = run(DOCS, llm, config=SystemConfig(evidence_check="critic"))
    assert final["answer"] == REFUSAL
    assert not final["ev"]["code_ok"]


def test_critic_can_reject_a_cited_answer():
    llm = FakeLLM(retriever={"answer": "24 times", "citations": ["The Montreal Canadiens have won the Stanley Cup 24 times"]},
                  critic=False)
    final, _ = run(DOCS, llm, config=SystemConfig(evidence_check="critic"))
    assert final["answer"] == REFUSAL


def test_parity_refuses_at_once_without_tools():
    final, log = run(DOCS, FakeLLM(), config=REBUILD)
    assert final["answer"] == REFUSAL and final["tool_calls"] == [] and log == []


def test_topic_rule_for_another_topic_does_not_apply():
    final, _ = run(DOCS_SPORTS, FakeLLM(steps=["Six."], topic="3"))  # 3 = Food: not Sports
    assert final["mode"] == "none" and final["answer"] == "Six."


def test_topic_rule_checked_per_domain_in_parity():
    final, _ = run(DOCS_SPORTS, FakeLLM(), config=REBUILD, topics="Sports & Athletics")
    assert final["mode"] == "docs" and final["answer"] == REFUSAL


def test_rule_in_new_wording_is_read_by_the_model():
    llm = FakeLLM(steps=["Six."])
    llm.topic = "10"
    original = llm.ainvoke

    async def reply(messages):
        if "Which type is it?" in str(messages[-1].content):
            return AIMessage(content="apis")
        return await original(messages)
    llm.ainvoke = reply
    final, _ = run("Please never search the documents; rely on the APIs.", llm)
    assert final["rule"] == {"allowed": "apis", "category": None, "source": "model"} and final["mode"] == "apis"


# ---------------------------------------------------------------- the controlled loop

def test_guard_blocks_a_forbidden_search_under_apis_only():
    llm = FakeLLM(steps=[("query_hockey", {"query": "cups"}), ("get_team_wins", {"team": "Canadiens"}), "24."])
    final, log = run(APIS, llm, api_reply='[{"wins": 24}]')
    assert ("query_hockey", "cups") not in log
    assert {"node": "rule_guard", "blocked": "query_hockey", "unknown": False} in final["log"]
    told = [m.content for m in final["messages"] if m.__class__.__name__ == "ToolMessage"][0]
    assert told.startswith("Blocked: 'query_hockey' is not allowed by the tool rule. Use one of [get_team_wins]")
    assert final["tool_calls"][0]["tool_name"] == "get_team_wins"


def test_guard_names_the_valid_tools_for_a_made_up_one():
    llm = FakeLLM(steps=[("get_team_cups", {"team": "Canadiens"}), "24."])
    final, _ = run(APIS, llm)
    told = [m.content for m in final["messages"] if m.__class__.__name__ == "ToolMessage"][0]
    assert told == "Error: get_team_cups is not a valid tool, try one of [get_team_wins]."


def test_empty_result_retries_with_other_letter_cases():
    reply = lambda team: '[{"wins": 24}]' if team == "CANADIENS" else "[]"
    llm = FakeLLM(steps=[("get_team_wins", {"team": "Canadiens"}), "24."])
    final, _ = run(APIS, llm, api_reply=reply)
    assert [c["arguments"]["team"] for c in final["tool_calls"]] == ["Canadiens", "CANADIENS"]
    told = [m.content for m in final["messages"] if m.__class__.__name__ == "ToolMessage"][0]
    assert "CANADIENS" in told and "24" in told


def test_empty_result_without_variants_gets_the_hint_twice_at_most():
    llm = FakeLLM(steps=[("get_team_wins", {"team": y}) for y in ("1990", "1991", "1992")] + ["No matching records."])
    final, _ = run(APIS, llm, question="Who won in 1990, 1991 or 1992?")
    told = [m.content for m in final["messages"] if m.__class__.__name__ == "ToolMessage"]
    assert sum("No rows matched" in t for t in told) == 2


def test_untraceable_value_is_warned_once_then_runs():
    llm = FakeLLM(steps=[("get_team_wins", {"team": "Bruins"}), ("get_team_wins", {"team": "Bruins"}), "7."])
    final, log = run(APIS, llm, api_reply='[{"wins": 7}]')
    assert log == [("get_team_wins", "Bruins")]
    assert any(e.get("warned") == ["Bruins"] for e in final["log"])


def test_step_budget_ends_with_an_answer():
    llm = FakeLLM(steps=[("get_team_wins", {"team": "Canadiens"})] * 10)
    final, _ = run(APIS, llm, max_steps=3, api_reply='[{"wins": 24}]',
                   config=SystemConfig(grounded_answer=False, loop_guard=False))
    assert final["answer"] == "I could not finish." and final["hop"] == 4


def test_document_lookup_in_the_loop_returns_a_verified_fact():
    llm = FakeLLM(steps=[("query_hockey", {"query": "Stanley Cup wins Canadiens"}), "24."],
                  retriever={"answer": "24", "citations": ["The Montreal Canadiens have won the Stanley Cup 24 times"]})
    final, _ = run("", llm, config=SystemConfig(grounded_answer=False))
    told = [m.content for m in final["messages"] if m.__class__.__name__ == "ToolMessage"][0]
    assert told.startswith("From the documents: 24")
    assert final["answer"] == "24."  # the loop never refuses


def test_document_lookup_not_found_is_not_a_refusal():
    llm = FakeLLM(steps=[("query_hockey", {"query": "coach salary"}), "No answer found."])
    final, _ = run("", llm, config=SystemConfig(grounded_answer=False))
    told = [m.content for m in final["messages"] if m.__class__.__name__ == "ToolMessage"][0]
    assert told == "Not found in the documents." and final["answer"] != REFUSAL


def test_rule_reader_off_sends_a_documents_only_rule_down_the_no_rule_path():
    final, _ = run(DOCS, FakeLLM(steps=["Six."]), config=SystemConfig(rule_reader="off"))
    assert final["rule"] is None and final["mode"] == "none" and "retrieve" not in nodes(final)
    assert final["answer"] == "Six."


def test_tool_filter_off_shows_and_allows_a_forbidden_search():
    llm = FakeLLM(steps=[("query_hockey", {"query": "cups"}), "24."])
    final, log = run(APIS, llm, config=SystemConfig(tool_filter="off"))
    assert final["mode"] == "apis" and "query_hockey" in final["allowed"]
    assert not any(e["node"] == "rule_guard" and e.get("blocked") for e in final["log"])


def test_own_search_only_in_the_no_rule_path():
    final, _ = run("", FakeLLM(steps=["Six."]))
    assert "query_books" not in final["allowed"] and "query_hockey" in final["allowed"]
    final, _ = run("", FakeLLM(steps=["Six."]), config=REBUILD)
    assert "query_books" in final["allowed"]


def test_back_referring_question_is_rewritten_for_ranking():
    history = [{"role": "user", "content": "Tell me about the Canadiens."}, {"role": "assistant", "content": "A team."}]
    final, _ = run("", FakeLLM(steps=["24."]), history=history, question="How many cups have they won?")
    assert final["subgoal"] == "How many Stanley Cups have the Canadiens won?"


def test_grounded_answer_keeps_the_draft_when_the_rewrite_drops_list_items():
    llm = FakeLLM(steps=[("get_team_wins", {"team": "Canadiens"}), "Canadiens, Bruins, Leafs"])
    original = llm.ainvoke

    async def reply(messages):
        if messages and "every item" in str(messages[0].content):
            return AIMessage(content="Canadiens")
        return await original(messages)
    llm.ainvoke = reply
    rows = json.dumps([{"team": "Canadiens"}, {"team": "Bruins"}, {"team": "Leafs"}])
    final, _ = run(APIS, llm, api_reply=rows, question="Which teams did the Canadiens play?")
    assert final["answer"] == "Canadiens, Bruins, Leafs"


# ---------------------------------------------------------------- checks

@pytest.mark.parametrize("text,empty", [("[]", True), ("{}", True), ('{"rows": []}', True), ("", True),
                                        ('[{"a": 1}]', False), ("Error: bad", False), ('{"n": 0}', False)])
def test_is_empty(text, empty):
    assert checks.is_empty(text) is empty


def test_case_variants():
    assert checks.case_variants({"team": "New Castle", "year": 1990}) == [
        {"team": "NEW CASTLE", "year": 1990}, {"team": "new castle", "year": 1990}]


def test_citation_found_allows_small_copy_differences():
    assert checks.citation_found("The Montreal Canadiens have won the Stanley Cup 24 times.", [PASSAGE])
    assert checks.citation_found("the montreal canadiens have won stanley cup 24 times", [PASSAGE])
    assert not checks.citation_found("The Boston Bruins won the cup 30 times", [PASSAGE])


def test_answer_in_citations():
    cited = ["The Montreal Canadiens have won the Stanley Cup 24 times"]
    assert checks.answer_in("24", cited) and checks.answer_in("The Montreal Canadiens", cited)
    assert not checks.answer_in("25", cited) and not checks.answer_in("Boston Bruins", cited)


def test_untraceable_and_lenient_values():
    seen = "How many games did the Canadiens win in 2014?"
    assert checks.untraceable({"team": "Canadiens", "date": "2014-08%", "n": "12"}, seen) == []
    assert checks.untraceable({"team": "Boston Bruins"}, seen) == ["Boston Bruins"]


def test_schema_problems():
    schema = {"properties": {"year": {"type": "integer"}, "team": {"type": "string"}}, "required": ["team"]}
    assert checks.schema_problems({"year": "1990", "team": "X"}, schema) == ({"year": 1990, "team": "X"}, [])
    assert checks.schema_problems({"year": "soon"}, schema)[1] == [
        "the required value 'team' is missing", "'year' should be of type integer, not 'soon'"]


def test_list_items_and_parse_json():
    assert checks.list_items('[{"name": "a"}, {"name": "b"}]') == ["a", "b"]
    assert checks.list_items('[{"n": 3}]') == []
    assert checks.parse_json('<think>x</think>```json\n{"answer": "24"}\n```') == {"answer": "24"}
    assert checks.parse_json("no json") is None


def test_system_settings_change_the_run_id_but_not_older_ones():
    plain = RunConfig(name="x")
    assert "system" not in plain.settings()
    assert RunConfig(name="x", system=SystemConfig()).run_id != plain.run_id


def test_graph_agent_runs_a_dialogue_for_the_runner():
    from agents.agent_interface import Message
    from subtask4.system.agent import GraphAgent
    log = []
    tools = [tool("get_team_wins", '[{"wins": 24}]', log), tool("query_hockey", "[]", log)]
    llm = FakeLLM(steps=[("get_team_wins", {"team": "Canadiens"}), "24."])
    agent = GraphAgent(config=RunConfig(name="t", system=SystemConfig(grounded_answer=False)), domain="hockey",
                       topics=Topics(None), llm=llm, tools=tools, top_k_tools=20, max_iterations=20)
    dialogue = [Message(role="user", content="Tell me about the Canadiens."), Message(role="assistant", content="A team."),
                Message(role="user", content="How many games did the Canadiens win?")]
    response = asyncio.run(agent.run(dialogue, additional_instructions=APIS))
    assert response.content == "24."
    assert response.tool_calls == [{"tool_name": "get_team_wins", "arguments": {"team": "Canadiens"},
                                    "result": '[{"wins": 24}]'}]
    assert response.trajectory[0]["type"] == "Subtask4Decision" and response.trajectory[0]["mode"] == "apis"
    assert response.trajectory[-1]["type"] == "Subtask4Log"
    assert llm.bound == ["get_team_wins"]  # the search tool is never shown under APIs-only


def test_studio_loads_training_dialogues_but_never_the_course_test():
    from subtask4 import data
    from subtask4.system import studio
    question, rule = studio.dialogue("books", None)
    assert question and isinstance(rule, str)
    with pytest.raises(data.FrozenSplitError):
        studio.dialogue(sorted(data.test_domains())[0], None)


def test_helper_prompts_get_a_short_history_but_the_argument_check_sees_all():
    from subtask4.system.graph import HISTORY_CHARS, HISTORY_TURNS
    history = [{"role": "user", "content": "Which papers?"}, {"role": "assistant", "content": "x" * 100_000 + " Canadiens"}]
    llm = FakeLLM(steps=[("get_team_wins", {"team": "Canadiens"}), "24."])
    final, log = run(APIS, llm, history=history, question="How many did they win?", api_reply='[{"wins": 24}]')
    rewrite = next(p for p in llm.prompts if "Rewrite the last question" in p)
    assert len(rewrite) < HISTORY_TURNS * (HISTORY_CHARS + 50) + 600
    assert log == [("get_team_wins", "Canadiens")]  # traced to the end of the long answer: not warned


def test_loop_guard_points_to_the_earlier_result_then_stops():
    llm = FakeLLM(steps=[("get_team_wins", {"team": "Canadiens"})] * 5)
    final, log = run(APIS, llm, api_reply='[{"wins": 24}]', config=SystemConfig(grounded_answer=False))
    assert log == [("get_team_wins", "Canadiens")]  # run once, never repeated
    told = [m.content for m in final["messages"] if m.__class__.__name__ == "ToolMessage"]
    assert told[1].startswith("You already made this exact call")
    assert final["answer"] == "Stopped." and final["stop"]


def test_loop_guard_lets_a_warned_call_through():
    llm = FakeLLM(steps=[("get_team_wins", {"team": "Bruins"}), ("get_team_wins", {"team": "Bruins"}), "7."])
    final, log = run(APIS, llm, api_reply='[{"wins": 7}]')
    assert log == [("get_team_wins", "Bruins")] and not any(e["node"] == "loop_guard" for e in final["log"])


def test_retry_uses_a_spelling_seen_in_an_earlier_result():
    reply = lambda team: ('[{"team": "Montreal Canadiens"}]' if team == "Montreal" else
                          '[{"wins": 24}]' if team == "Montreal Canadiens" else "[]")
    llm = FakeLLM(steps=[("get_team_wins", {"team": "Montreal"}), ("get_team_wins", {"team": "Montreal Canadians"}), "24."])
    final, _ = run(APIS, llm, api_reply=reply, question="How many cups have the Montreal Canadians won?")
    assert final["tool_calls"][-1]["arguments"] == {"team": "Montreal Canadiens"}
    assert '"wins": 24' in final["tool_calls"][-1]["result"]


def test_value_variants():
    swapped = checks.value_variants({"first_name": "Pedro", "last_name": "Pierluisi"})
    assert {"first_name": "Pierluisi", "last_name": "Pedro"} in swapped
    assert {"phone_brand": "vivo", "device_model": "X5Pro"} in checks.value_variants({"phone_brand": "vivo", "device_model": "X5pro"})
    assert "2018/8/13" in checks.format_forms("2018/08/13")
    seen = checks.seen_values(['{"names": ["Stanford University"]}'])
    assert {"school": "Stanford University"} in checks.value_variants({"school": "Standford University"}, seen)


def test_tool_order_shuffled_keeps_the_same_tools():
    final, _ = run("", FakeLLM(steps=["Six."]), config=SystemConfig(tool_order="shuffled"))
    ranked, _ = run("", FakeLLM(steps=["Six."]))
    assert sorted(final["shortlist"]) == sorted(ranked["shortlist"])


def test_refusal_is_the_reference_wording():
    assert REFUSAL == "I can not answer."


def test_answer_check_items_come_from_tune_and_dev_and_are_scored_by_the_known_value():
    from subtask4 import answer_check, data
    items = json.loads(answer_check.ITEMS.read_text())
    assert items and {x["split"] for x in items} <= {"tune", "dev"}
    assert not {x["domain"] for x in items} & data.test_domains()
    assert answer_check.outcome("verified", "It is Ukraine.", ["Ukraine"]) == "correct"
    assert answer_check.outcome("verified", "Poland", ["Ukraine"]) == "wrong"
    assert answer_check.outcome("nothing", None, ["Ukraine"]) == "refused"
