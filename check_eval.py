import json, sys, collections
ev = json.load(open(sys.argv[1]))
n = bad = 0; kinds = collections.Counter()
for v in ev["domains"].values():
    for dl in v["dialogues"]:
        for t in dl["details"]["per_turn"]:
            n += 1
            ex = (t.get("metadata") or {}).get("score_explanation") or {}
            for k in ("answer", "groundedness"):
                s = str(ex.get(k) or "")
                if "Judge Error" in s:
                    bad += 1; kinds[s[:110]] += 1
print(f"{n} turns; judge errors: {bad}")
for s, c in kinds.most_common(5): print(" ", c, s)
print("summary:", ev["summary"].get("n_correct"), "/", ev["summary"].get("n_samples"))
print("VALID" if bad == 0 else "NOT VALID: judge errors present; do not report this score")
