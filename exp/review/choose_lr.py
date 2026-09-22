"""Lock the learning rate per arm from the wave-1 pilot (seed 43, tracking-val MRR only; test is never consulted).

    python -m exp.review.choose_lr --glob "results/review_2026_09_22/matched/wiki_*_s43_lr*" --out results/review_2026_09_22/matched/wiki_lr_choice.json
"""
import argparse
import glob
import json
import os
import re

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    best = {}
    rows = []
    for d in sorted(glob.glob(args.glob)):
        res = os.path.join(d, "results.csv")
        if not os.path.exists(res):
            continue
        m = re.match(r".*/wiki_(\w+?)_s43_lr(.+)$", d)
        arm, lr = m.group(1), m.group(2)
        r = pd.read_csv(res).iloc[0]
        val = float(r["best_track_val_mrr"])
        rows.append(dict(arm=arm, lr=lr, best_track_val_mrr=val, best_epoch=int(r["best_track_epoch"]), final_val=r.get("validation_mrr")))
        if arm not in best or val > best[arm][1]:
            best[arm] = (lr, val)
    choice = {arm: lr for arm, (lr, _) in best.items()}
    table = pd.DataFrame(rows).sort_values(["arm", "lr"])
    print(table.to_string(index=False))
    print("choice:", choice)
    json.dump({"choice": choice, "rule": "highest tracking-validation MRR at seed 43; ties -> first", "table": rows},
              open(args.out, "w"), indent=1)
    json.dump(choice, open(args.out.replace(".json", "_flat.json"), "w"))


if __name__ == "__main__":
    main()
