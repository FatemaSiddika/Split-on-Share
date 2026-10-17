#!/usr/bin/env python3
"""
rescore_string_labels.py — score saved predictions of string-label tasks that were
evaluated before dev_trace_benchmark_eval.py had a scorer for them (metric -1).
No model is loaded; it reads predictions_<split>.json and rewrites the matching
evaluation_summary.json entry, so cl_metrics.py can then be rerun.

    python rescore_string_labels.py --base_dir <trained_models/.../ablation_<tag>> [--base_dir ...] [--dry_run]

Scoring = eval_StringLabel from dev_trace_benchmark_eval.py, the same function
dbpedia / amazon / yahoo / agnews / yelp / BoolQA / QQP are scored with, and the
same rounding as get_result_value (6 decimals). Only entries still at -1 are
changed unless --force. Each summary is backed up once to
evaluation_summary.json.bak_prerescore before the first change.
"""
import os, sys, json, glob, shutil, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dev_trace_benchmark_eval import eval_StringLabel

TASKS = ["MNLI", "CB", "WiC", "COPA", "RTE", "IMDB", "SST-2", "MultiRC"]


def predictions_path(summary_path, ds, res):
    p = res.get("predictions_file")
    if p and os.path.isfile(p):
        return p
    split = res.get("split", "test")
    p = os.path.join(os.path.dirname(summary_path), ds, f"predictions_{split}.json")
    return p if os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", action="append", required=True)
    ap.add_argument("--dry_run", action="store_true", help="print scores, write nothing")
    ap.add_argument("--force", action="store_true", help="also rescore entries that already have a score")
    a = ap.parse_args()

    labels_seen = {}
    n_changed = 0
    for base in a.base_dir:
        print(f"\n== {base}")
        summaries = sorted(glob.glob(os.path.join(base, "shared_moE_checkpoints_*", "eval_task_*", "evaluation_summary.json")))
        if not summaries:
            print("   no evaluation_summary.json found")
            continue
        for sp in summaries:
            with open(sp) as f:
                summ = json.load(f)
            results = summ.get("results", {})
            changed = False
            for ds, res in results.items():
                if ds not in TASKS:
                    continue
                if res.get("metric_value") not in (-1, None) and not a.force:
                    continue
                pp = predictions_path(sp, ds, res)
                rel = os.path.relpath(sp, base)
                if pp is None:
                    print(f"   [skip] {rel}: predictions file not found")
                    continue
                with open(pp) as f:
                    preds = json.load(f)
                gen = [r.get("generated_text", "") for r in preds]
                gold = [r.get("answer", "") for r in preds]
                ev = eval_StringLabel(gen, gold)
                val = round(ev["accuracy"], 6)
                labels_seen.setdefault(ds, set()).update(g.strip() for g in gold)
                print(f"   {ds:<8} {val:.4f}  (n={len(gold)})  {rel}")
                res["metric_value"] = val
                res["eval_result"] = ev
                res["rescored_by"] = "rescore_string_labels.py (eval_StringLabel)"
                changed = True
            if changed and not a.dry_run:
                bak = sp + ".bak_prerescore"
                if not os.path.exists(bak):
                    shutil.copy2(sp, bak)
                with open(sp, "w") as f:
                    json.dump(summ, f, indent=2, ensure_ascii=False)
                n_changed += 1

    print("\nGold labels per task (check they are short labels, as for dbpedia/QQP):")
    for ds in TASKS:
        if ds in labels_seen:
            labs = sorted(labels_seen[ds])
            more = f" ... ({len(labs)} distinct)" if len(labs) > 6 else ""
            print(f"   {ds:<8} {labs[:6]}{more}")
    print(f"\n{'(dry run) ' if a.dry_run else ''}summaries updated: {n_changed}")


if __name__ == "__main__":
    main()
