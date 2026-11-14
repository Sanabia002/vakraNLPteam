import json, os
root = "data/train/capability_3_multihop_reasoning/output"
dev = json.load(open("splits/dev_split.json"))["capability_3"]["dev_domains"]
test = ["bike_share_1", "computer_student", "disney", "olympics", "restaurant", "student_loan"]
for name, doms in (("dev", dev), ("test", test)):
    os.makedirs(f"tmp_gt_{name}", exist_ok=True); uu, per = [], {}
    for d in doms:
        items = json.load(open(f"{root}/{d}.json"))
        json.dump(items, open(f"tmp_gt_{name}/{d}.json", "w")); uu += [it["uuid"] for it in items]; per[d] = len(items)
    json.dump(uu, open(f"results/{name}_uuids.json", "w")); print(name, len(uu), "questions", per)
