#!/usr/bin/env python3
"""
rescore_sciqa.py — rescore every saved ScienceQA prediction of a run with the
fixed scorer (utility/TRACE_metrics.eval_SciQA: first option letter on the
first line, accuracy over all items). No GPU, no generation.

Every evaluation_summary.json under the run (checkpoint eval dirs AND the
fl_logs snapshots the collector reads) that holds a ScienceQA result is
recomputed from the predictions_test.json next to it.

    cd <...>/fed-con
    python3 utils/rescore_sciqa.py <run folder>           # dry run: print old -> new
    python3 utils/rescore_sciqa.py <run folder> --write   # update the summaries
    then re-run collect_results.sh (same settings as training) for LA / OP / BWT

--write keeps the original of each file once as evaluation_summary.json.pre_sciqa
and is safe to repeat. Other tasks are never modified; for C-STANCE, FOMC and
NumGLUE the script only REPORTS how many outputs the current scorer drops from
the accuracy denominator (unparsed outputs are skipped, not counted wrong).
"""
import argparse
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from TRACE_metrics import (eval_SciQA, postprocess_choice_acc,   # noqa: E402
                                   postprocess_choice_num)

SKIP = ("/fl_data/", "/fl_workdirs/")
BACKUP = "evaluation_summary.json.pre_sciqa"


def preds_for(summary_dir, task):
    hits = glob.glob(os.path.join(summary_dir, task, "predictions_test.json")) or \
        glob.glob(os.path.join(summary_dir, "**", task, "predictions_test.json"), recursive=True)
    return hits[0] if hits else None


def numeric(text):
    import re
    m = re.findall(r'-?\d+(?:\.\d+)?', (text or "").strip())
    return m[-1] if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    run = os.path.abspath(a.run.rstrip("/"))
    summaries = sorted(p for p in glob.glob(os.path.join(run, "**", "evaluation_summary.json"),
                                            recursive=True) if not any(s in p for s in SKIP))
    print(f"{len(summaries)} evaluation_summary.json files under {run}\n")

    n_sci = n_changed = 0
    other = []
    for sp in summaries:
        d = os.path.dirname(sp)
        rel = os.path.relpath(d, run)
        try:
            obj = json.load(open(sp))
        except Exception as e:
            print(f"  skip {rel}: {type(e).__name__}")
            continue
        res = obj.get("results") or {}
        dirty = False
        for task, r in res.items():
            if not isinstance(r, dict) or "error" in r:
                continue
            pf = preds_for(d, task)
            if not pf:
                continue
            items = json.load(open(pf))
            P = [x.get("generated_text", "") for x in items]
            G = [x.get("answer", "") for x in items]
            if task == "ScienceQA":
                new = eval_SciQA(P, G)
                old = r.get("metric_value")
                n_sci += 1
                mark = "" if old is not None and abs(old - new["accuracy"]) < 1e-6 else "  *"
                n_changed += bool(mark)
                print(f"  ScienceQA {rel:<95} {old!s:>9} -> {new['accuracy']:.6f}"
                      f"   (no letter {new['no_letter']}/{new['n']}){mark}")
                if a.write and mark:
                    bk = os.path.join(d, BACKUP)
                    if not os.path.exists(bk):
                        shutil.copy2(sp, bk)
                    r["metric_value"] = round(new["accuracy"], 6)
                    r["eval_result"] = new
                    r["scorer"] = "sciqa_firstline_alln_v1"
                    dirty = True
            elif task in ("C-STANCE", "FOMC", "NumGLUE-cm", "NumGLUE-ds"):
                if task.startswith("NumGLUE"):
                    pp = postprocess_choice_num([numeric(p) for p in P])
                else:
                    pp = postprocess_choice_acc(P)
                dropped = sum(1 for p in pp if p == "")
                if dropped:
                    other.append((task, rel, dropped, len(pp)))
        if a.write and dirty:
            with open(sp, "w") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

    print(f"\nScienceQA: {n_sci} cells, {n_changed} changed"
          + ("" if a.write else "   (dry run — add --write to update the summaries)"))
    if other:
        print(f"\nOther accuracy tasks — outputs the CURRENT scorer drops from the denominator "
              f"(not changed by this script): {len(other)} cells")
        for task, rel, k, n in other[:40]:
            print(f"  {task:<11} {rel:<95} {k}/{n}")
    else:
        print("\nOther accuracy tasks: no dropped outputs — their denominators are already all items.")


if __name__ == "__main__":
    main()
