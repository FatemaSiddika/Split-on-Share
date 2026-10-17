#!/usr/bin/env python3
"""Routing-ablation table (read-only).

Reads eval_task_<J>_<name><suffix>/evaluation_summary.json inside ONE checkpoint
folder and prints, for every routing variant found, the score per task and OP
(mean over the requested tasks). The column "existing" is the original
eval_task_<J>_<name>/ folder (learned gate, from the training run).

  python analysis/router_eval_report.py --ckpt <.../shared_moE_checkpoints_<model>_8_<name>> \
      --task_list "C-STANCE,FOMC,..." --tasks 1:2:3:4:5:6:7:8 --out analysis/router_eval/<name>

Writes router_eval_table.csv to --out. Never writes inside --ckpt.
"""
import argparse, csv, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--task_list", required=True, help="comma list, same order as training")
ap.add_argument("--tasks", required=True, help="task ids to report, colon or comma list")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

names = [t for t in a.task_list.split(",") if t]
ids = [int(x) for x in re.split(r"[:,]", a.tasks) if x.strip()]


def score(d):
    f = os.path.join(d, "evaluation_summary.json")
    if not os.path.isfile(f):
        return None
    res = json.load(open(f)).get("results", {})
    vals = [v.get("metric_value") for v in res.values()
            if isinstance(v, dict) and v.get("metric_value") is not None]
    return float(vals[0]) if vals else None


# suffix -> {task id: score}
table = {}
for j in ids:
    prefix = f"eval_task_{j}_{names[j - 1]}"
    for d in os.listdir(a.ckpt):
        if d == prefix or (d.startswith(prefix + "_") and os.path.isdir(os.path.join(a.ckpt, d))):
            sfx = d[len(prefix):]
            table.setdefault(sfx, {})[j] = score(os.path.join(a.ckpt, d))


def order(sfx):
    if sfx == "":
        return (0, 0, sfx)
    if sfx == "_learned":
        return (1, 0, sfx)
    if sfx.startswith("_uniform"):
        return (2, 0, sfx)
    if sfx == "_oracle":
        return (3, 0, sfx)
    m = re.fullmatch(r"_top(\d+)", sfx)
    if m:
        return (4, int(m.group(1)), sfx)
    return (5, 0, sfx)


cols = sorted(table, key=order)
label = {s: ("existing" if s == "" else s.lstrip("_")) for s in cols}
fmt = lambda x: "---" if x is None else f"{x:.4f}"

op = {}
for s in cols:
    v = [table[s].get(j) for j in ids]
    op[s] = sum(v) / len(v) if v and all(x is not None for x in v) else None

w = max(12, max(len(label[s]) for s in cols) + 2) if cols else 12
print(f"{'task':<18}" + "".join(f"{label[s]:>{w}}" for s in cols))
for j in ids:
    print(f"{str(j) + ' ' + names[j - 1]:<18}" + "".join(f"{fmt(table[s].get(j)):>{w}}" for s in cols))
print(f"{'OP':<18}" + "".join(f"{fmt(op[s]):>{w}}" for s in cols))
print("\nOP is shown only when every requested task has a score for that variant.")

with open(os.path.join(a.out, "router_eval_table.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["task_id", "task"] + [label[s] for s in cols])
    for j in ids:
        wr.writerow([j, names[j - 1]] + [table[s].get(j) for s in cols])
    wr.writerow(["OP", ""] + [op[s] for s in cols])
print(f"\nwritten to {os.path.join(a.out, 'router_eval_table.csv')}")
