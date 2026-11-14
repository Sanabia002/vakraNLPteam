"""Fake-data tests for final_report.py. Every file is written to a fresh temp dir (the script uses relative paths and
runs with that dir as cwd), so nothing in results/ or tmp_gt_*/ is touched. Run: python -m pytest -q test_final_report.py
(or python test_final_report.py)."""
import json, os, subprocess, sys, tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_report.py")
S = "selftest"


def gold(uuid, hops):
    return {"uuid": uuid, "output": [{"sequence": {"tool_call": [{}] * hops}}]}


def dialogue(uuid, ok, judge_error=False):
    ex = {"answer": "Judge Error: 413 request too large"} if judge_error else {}
    return {"uuid": uuid, "score": 1.0 if ok else 0.0, "details": {"per_turn": [{"metadata": {"score_explanation": ex}}]}}


def ev(rows):
    """rows: list of (uuid, domain, ok[, judge_error])."""
    doms = {}
    for r in rows:
        doms.setdefault(r[1], []).append(dialogue(r[0], r[2], *r[3:]))
    return {"domains": {d: {"dialogues": v} for d, v in doms.items()}}


def report(baseline, f, gt, merged=None):
    """baseline/f: rows for ev(); gt: {domain: [(uuid, hops)]}; merged: {"baseline"|"F": rows}. Returns stdout."""
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(f"{tmp}/results"); os.makedirs(f"{tmp}/tmp_gt_{S}")
        for dom, items in gt.items():
            json.dump([gold(u, h) for u, h in items], open(f"{tmp}/tmp_gt_{S}/{dom}.json", "w"))
        json.dump(ev(baseline), open(f"{tmp}/results/eval_final_{S}_baseline.json", "w"))
        json.dump(ev(f), open(f"{tmp}/results/eval_final_{S}_F.json", "w"))
        for name, rows in (merged or {}).items():
            json.dump(ev(rows), open(f"{tmp}/results/eval_final_{S}_{name}_merged.json", "w"))
        p = subprocess.run([sys.executable, SCRIPT, S], cwd=tmp, capture_output=True, text=True)
    assert p.returncode == 0, f"final_report.py crashed:\n{p.stderr}"
    return " ".join(p.stdout.split())


def ten(ok, dom=lambda i: "dA" if i < 5 else "dB"):
    return [(f"u{i}", dom(i), i in ok) for i in range(10)]


GT10 = {"dA": [(f"u{i}", 2) for i in range(5)], "dB": [(f"u{i}", 2 if i < 6 else 3) for i in range(5, 10)]}


def test_original_selftest():
    out = report(ten({0, 1, 2, 3}), ten({0, 1, 2, 4, 5, 6}), GT10)   # both 3, only baseline 1, only F 3
    for want in ("baseline 4/10 = 40.0% (95% interval 17-69%)", "F 6/10 = 60.0% (95% interval 31-83%)",
                 "both 3 | only baseline 1 | only F 3", "gold calls = 2: n=6 baseline 4 F 5",
                 "gold calls = 3: n=4 baseline 0 F 1", "dA 4 / 4 / 5", "dB 0 / 2 / 5"):
        assert want in out, f"missing: {want!r}\n{out}"
    assert "WARNING" not in out and "NOT USABLE" not in out


def test_unjudged_items_flagged_and_counted_wrong():
    base = ten({0, 1, 2, 3}); base[0] = ("u0", "dA", False, True); base[5] = ("u5", "dB", False, True)
    out = report(base, ten({0, 1}), GT10)
    assert "baseline 3/10 = 30.0%" in out and "unjudged 2 <-- NOT USABLE: re-judge first" in out
    assert "F 2/10 = 20.0% (95% interval 6-51%) unjudged 0" in out and out.count("NOT USABLE") == 1


def test_merged_file_preferred_over_original():
    base = ten({0, 1, 2, 3}); base[0] = ("u0", "dA", False, True)
    out = report(base, ten({0}), GT10, merged={"baseline": ten({0, 1, 2, 3})})
    assert "baseline 4/10 = 40.0%" in out and "NOT USABLE" not in out


def test_question_in_only_one_run():
    base = ten({0, 1, 2, 3}) + [("u10", "dC", True)]            # u10 only in baseline (correct, own domain)
    f = [r for r in ten({0, 1, 2, 4, 5, 6}) if r[0] != "u9"]     # u9 only in baseline too
    gt = {**GT10, "dC": [("u10", 4)]}
    out = report(base, f, gt)
    assert "baseline 5/11" in out and "F 6/9" in out
    assert "do not cover the same questions: 2 differ; paired numbers below use the 9 questions in both" in out
    assert "both 3 | only baseline 1 | only F 3" in out          # u10 is not counted as "only baseline"
    assert "gold calls = 3: n=3 baseline 0 F 1" in out and "gold calls = 4: n=0" in out
    assert "dB 0 / 2 / 4" in out and "dC" not in out


def test_zero_correct_wilson_interval():
    out = report(ten(set()), ten(set(range(10))), GT10)
    assert "baseline 0/10 = 0.0% (95% interval 0-28%)" in out, out   # was "-0-28%" (float noise)
    assert "F 10/10 = 100.0% (95% interval 72-100%)" in out
    assert "both 0 | only baseline 0 | only F 10" in out


def test_empty_results():
    out = report([], [], {})
    assert "baseline 0/0: no questions in the results file" in out and "F 0/0: no questions" in out
    assert "both 0 | only baseline 0 | only F 0" in out


def test_one_run_empty():
    out = report(ten({0, 1}), [], GT10)
    assert "baseline 2/10 = 20.0%" in out and "F 0/0: no questions" in out
    assert "10 differ; paired numbers below use the 0 questions in both" in out
    assert "gold calls = 2: n=0" in out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"SELFTEST PASSED ({len(tests)} tests)")
