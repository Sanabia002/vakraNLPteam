"""Every prompt the new system sends to its model (Qwen3.5-4B in all roles). One place, so a run's behaviour can be
read from here and changes show up in review."""

API_FIRST = """Prefer the database API tools. Use a document search only when no API tool can provide the information."""

BUDGET = """You have reached the step limit. Answer the question now, using only the tool results above."""

BLOCKED = """Blocked: the tool '{name}' is not allowed here (the tool rule, or it does not exist). Choose one of the tools you were given, or answer."""

ARG_PROBLEMS = """The call to '{name}' was not run: {problems}. Fix the arguments and call it again."""

ARG_WARNING = """Check before running '{name}': the value(s) {values} do not appear in the question, the dialogue or earlier tool results. The database matches values exactly. If you are sure, make the same call again and it will run."""

EMPTY_HINT = """No rows matched. The database matches values exactly; it may store this value differently (UPPERCASE, an abbreviation, a code). Try another form or another tool. If other forms also return nothing, answer that no matching records were found."""

NOT_FOUND = "Not found in the documents."

VERIFIED = "From the documents: {answer} (source: \"{citation}\")"

RULE_TYPES = """Here is a rule about which tools an assistant may use:
"{rule}"

Which type is it? Reply with one line:
docs - only document retrievers may be used, for every question
apis - document retrievers may not be used, for every question
docs N - only document retrievers, but only for questions about category N
apis N - no document retrievers, but only for questions about category N
other - none of these

Categories:
{menu}"""

QUESTION_TOPIC = """A user asked this question to an assistant for the '{domain}' database:
"{question}"
{context}
Which ONE category below is this question about?
{menu}
Reply with the category number only."""

STANDALONE = """Earlier turns of a dialogue:
{history}

Last question: "{question}"

Rewrite the last question so that it can be understood without the earlier turns (replace words such as "those" or "it" with what they refer to). Reply with the rewritten question only."""

KEYWORDS = """Write a short keyword search query (3 to 8 words) for finding documents that answer this question:
"{question}"
Reply with the query only."""

RETRIEVER = """Question: {question}
{history}
Passages found in the documents:
{passages}

Answer the question using ONLY these passages. Copy up to 3 sentences from the passages, word for word, that support your answer.
If the passages do not contain the answer, use null for the answer.
Reply with JSON only: {{"answer": "<short answer or null>", "citations": ["<copied sentence>", ...]}}"""

CRITIC = """You check another assistant's work. Question: {question}

Passages:
{passages}

Proposed answer: {answer}
Cited sentences:
{citations}

Do the cited sentences, which come from the passages, really support this answer to the question as asked?{scope}
Reply with JSON only: {{"supported": true or false, "missing": "<if not supported: what information is missing>"}}"""

SCOPE = """
Mind the scope: a question about records in a database (counts of rows, lists of records, IDs, values stored in tables) is NOT answered by a general passage that merely mentions the topic."""

RESEARCH = """We are looking for documents that answer: "{question}"
The last search did not find enough. What is missing: {missing}
Write one new search query (3 to 12 words) that could find it. Reply with the query only."""

GROUND_LIST = """If the tool results contain a list, include every item of the list in your answer."""
