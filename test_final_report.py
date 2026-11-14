import json, os, shutil, subprocess, sys
S = "selftest"; os.makedirs(f"tmp_gt_{S}", exist_ok=True); os.makedirs("results", exist_ok=True)
base_ok = {0, 1, 2, 3}; f_ok = {0, 1, 2, 4, 5, 6}                      # both = 3, only baseline = 1, only F = 3
def gold(i): return {"uuid": f"u{i}", "output": [{"sequence": {"tool_call": [{}] * (2 if i < 6 else 3)}}]}
json.dump([gold(i) for i in range(5)], open(f"tmp_gt_{S}/dA.json", "w")); json.dump([gold(i) for i in range(5, 10)], open(f"tmp_gt_{S}/dB.json", "w"))
def ev(ok):
    dom = {"dA": [], "dB": []}
    for i in range(10):
        dom["dA" if i < 5 else "dB"].append({"uuid": f"u{i}", "score": 1.0 if i in ok else 0.0, "details": {"per_turn": [{"metadata": {"score_explanation": {}}}]}})
    return {"domains": {d: {"dialogues": v} for d, v in dom.items()}}
json.dump(ev(base_ok), open(f"results/eval_final_{S}_baseline.json", "w")); json.dump(ev(f_ok), open(f"results/eval_final_{S}_F.json", "w"))
try:
    out = subprocess.run([sys.executable, "final_report.py", S], capture_output=True, text=True, check=True).stdout; print(out)
    for want in ("baseline 4/10 = 40.0% (95% interval 17-69%)", "F 6/10 = 60.0% (95% interval 31-83%)", "both 3 | only baseline 1 | only F 3",
                 "gold calls = 2: n=6 baseline 4 F 5", "gold calls = 3: n=4 baseline 0 F 1", "dA 4 / 4 / 5", "dB 0 / 2 / 5"):
        assert want in " ".join(out.split()), f"missing in output: {want!r}"
    print("SELFTEST PASSED")
finally:
    shutil.rmtree(f"tmp_gt_{S}", ignore_errors=True)
    for n in ("baseline", "F"): os.remove(f"results/eval_final_{S}_{n}.json")
