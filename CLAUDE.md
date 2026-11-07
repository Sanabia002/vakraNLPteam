# Rules for this repo (capability 3 project)
- Work only on branch melissa/solution. Review diffs with me before any commit. Never push to main.
- Never read, print, or commit: .env, ~/.groq_key, any *.log, output/, tool_dumps/. Never echo key values.
- Develop and tune ONLY on non-dev domains. Dev domains are in splits/dev_split.json. Do not run the
  evaluator or benchmark runs on dev domains unless I explicitly ask.
- Do not touch data/ or the data/test symlink. Do not modify tools or the evaluator's scoring.
- Every benchmark run goes through run_logged.py with a NEW --out name. Ask before any run over 30 questions.
- The model server is local (llama.cpp, port 8080, ~30 tok/s). Keep LLM calls per question small.
- Every new behavior is a flag (env var or CLI), default OFF, so the baseline stays reproducible.
- Use the venv: source .venv/bin/activate. mcp must stay 2.3.0; agents/mcp_tool_wrapper.py has a required fix.
- FROZEN TEST domains, never run, score, sample from, or read outputs of: bike_share_1, computer_student, disney,
  olympics, restaurant, student_loan (cap 3); also sales_in_weather (cap 4).
- hockey is analysis-only. Dev domains (splits/dev_split.json) only on request.
- Smoke tests: use a tune domain, e.g. books, cars, or authors.
- The tune sample was regenerated; use results/tune_sample_uuids.json only after it is rewritten by plan_recall.py.
