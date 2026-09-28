#!/usr/bin/env python3
"""
collect_results.py — collect the continual-learning results of Split-on-Share
(SETA_fix) runs into one place: <run>/results/  (matrix, LA / OP / BWT).
Read-only on the run folders. No GPU. Works for the plain and the *_order
launchers alike: both write the same layout, and the task order is read from
the folder names, not from a task list.

Layout it reads (run_setafix*.sh, setafix_all_tasks*.slurm, RL_Lamda_fsys_eval_only_arms*.sh):
    <run>/shared_moE_checkpoints_<model>_<T>_<task>/            model after task T  (row T)
          eval_task_<j>_<Task><EVAL_OUT_SUFFIX>/evaluation_summary.json   scored on task j (column j)

Usage (from Split-on-Share/):
    one run, found from the model + the SAME settings you trained with:
      python3 helpers/collect_results.py --model qwen3_4b --launcher order \
          BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical SETAFIX_ORTHO=on \
          SETAFIX_GATE_PROTECT=freeze SETAFIX_LAYER_GATE=budget SETAFIX_PHI=0.012
      (--launcher plain = scripts/run_setafix.sh, order = scripts/run_setafix_order.sh;
       settings are KEY=VALUE, exactly what you put before 'bash scripts/run_setafix*.sh')
    a run folder directly:
      python3 helpers/collect_results.py <run folder> [<run folder> ...]
    every run:
      python3 helpers/collect_results.py --all trained_models                 # every run
      python3 helpers/collect_results.py --all trained_models --match order   # names containing "order"
    --suffix _<mode> (or EVAL_OUT_SUFFIX=_<mode>) uses eval folders written with EVAL_OUT_SUFFIX.

Output per run: <run>/results/
    metrics.txt      matrix (x100) + LA / OP / BWT (+ BWT without ScienceQA)
    R_matrix.csv     rows = model after task T, columns = scored task j
    all_results.csv  every cell with its file and time
    metrics.json
With --all, also <dir>/results_summary.csv: one line per run.

    LA  = mean_j R[j][j]                    accuracy right after learning task j
    OP  = mean_j R[N][j]                    last model on every task
    BWT = mean_{j<N} (R[N][j] - R[j][j])    change on old tasks (negative = forgetting)
    N   = last task that has a checkpoint. Missing cells are listed, never filled.
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import subprocess
import sys

CKPT_RE = re.compile(r"^shared_moE_checkpoints_(?P<model>.+)_(?P<T>\d+)_(?P<name>[^_]+)$")
EVAL_RE = re.compile(r"^eval_task_(?P<j>\d+)_(?P<rest>.+)$")


def score_of(summary_path):
    """(dataset, metric_value) from an evaluation_summary.json, or (None, None)."""
    try:
        res = json.load(open(summary_path)).get("results") or {}
    except Exception:
        return None, None
    for ds, r in res.items():
        if isinstance(r, dict) and "error" not in r:
            v = r.get("metric_value")
            if isinstance(v, (int, float)) and v >= 0:
                return ds, float(v)
    return None, None


def pct(v):
    return "      -" if v is None else f"{100 * v:7.2f}"


def collect(run, suffix):
    run = os.path.abspath(run.rstrip("/"))
    ckpts = {}                                   # T -> (folder, task name lower)
    for d in sorted(glob.glob(os.path.join(run, "shared_moE_checkpoints_*"))):
        m = CKPT_RE.match(os.path.basename(d))
        if m and os.path.isdir(d):
            ckpts[int(m.group("T"))] = (d, m.group("name"))
    if not ckpts:
        return None

    cells, names, skipped = {}, {}, []
    for T, (d, tname) in ckpts.items():
        names.setdefault(T, tname)
        for ed in sorted(glob.glob(os.path.join(d, "eval_task_*"))):
            m = EVAL_RE.match(os.path.basename(ed))
            sp = os.path.join(ed, "evaluation_summary.json")
            if not m or not os.path.isfile(sp):
                continue
            j = int(m.group("j"))
            ds, v = score_of(sp)
            if ds is None:
                skipped.append((sp, "no usable metric_value (error or -1)"))
                continue
            rest = m.group("rest")
            if rest != ds + suffix:          # other EVAL_OUT_SUFFIX variant: not this table
                if rest.startswith(ds) and rest != ds:
                    continue
                skipped.append((sp, f"folder '{rest}' does not match dataset '{ds}{suffix}'"))
                continue
            names[j] = names.get(j) or ds.lower()
            if j in ckpts and ckpts[j][1] != ds.lower():
                skipped.append((sp, f"task {j} is '{ckpts[j][1]}' by its checkpoint, "
                                    f"but this eval is '{ds}'"))
                continue
            cells[(T, j)] = {"score": v, "dataset": ds, "file": sp,
                             "mtime": os.path.getmtime(sp)}
            names[j] = ds                    # proper-case name for the header

    N = max(ckpts)
    label = [names.get(j, f"task{j}") for j in range(1, N + 1)]
    R = [[cells[(T, j)]["score"] if (T, j) in cells else None for j in range(1, N + 1)]
         for T in range(1, N + 1)]
    diag = [R[j][j] for j in range(N)]
    last = R[N - 1]
    miss_diag = [f"T{j + 1}" for j in range(N) if diag[j] is None]
    miss_last = [f"T{j + 1}" for j in range(N) if last[j] is None]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    LA, OP = mean(diag), mean(last)
    pairs = [(label[j], last[j] - diag[j]) for j in range(N - 1)
             if last[j] is not None and diag[j] is not None]
    BWT = mean([d for _, d in pairs])
    BWT_noSQA = mean([d for n, d in pairs if n.lower() != "scienceqa"])

    # ── write ────────────────────────────────────────────────────────────
    out = os.path.join(run, "results")
    os.makedirs(out, exist_ok=True)
    L = [f"run   : {run}",
         f"tasks : " + ", ".join(f"T{j + 1}={label[j]}" for j in range(N)),
         f"checkpoints T{min(ckpts)}..T{N}: {len(ckpts)}   cells: {len(cells)}"
         + (f"   eval suffix: '{suffix}'" if suffix else ""), ""]
    L.append("after " + "".join(f"{('T' + str(j + 1)):>8}" for j in range(N)))
    for T in range(N):
        if any(v is not None for v in R[T]):
            L.append(f"T{T + 1:<4} " + "".join(f" {pct(v)}" for v in R[T]))
    L.append("")
    L.append(f"LA  = {pct(LA).strip()}" + (f"   (diagonal missing: {' '.join(miss_diag)})" if miss_diag else ""))
    L.append(f"OP  = {pct(OP).strip()}" + (f"   (last row T{N} missing: {' '.join(miss_last)})" if miss_last else ""))
    L.append(f"BWT = {pct(BWT).strip()}   over {len(pairs)} task(s)")
    if BWT_noSQA is not None and any(n.lower() == "scienceqa" for n, _ in pairs):
        L.append(f"BWT without ScienceQA = {pct(BWT_noSQA).strip()}")
    if miss_diag or miss_last:
        L.append("NOTE: averages use only the cells that exist; missing cells are listed above.")
    dup = [n for n in set(label) if label.count(n) > 1]
    if dup:
        L.append(f"NOTE: task(s) appear more than once in this order: {dup}")
    if skipped:
        L.append("")
        L.append(f"not used: {len(skipped)} file(s)")
        for f, why in skipped[:15]:
            L.append(f"  {os.path.relpath(f, run)} — {why}")
    text = "\n".join(L)

    open(os.path.join(out, "metrics.txt"), "w").write(text + "\n")
    with open(os.path.join(out, "R_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["after"] + [f"T{j + 1}_{label[j]}" for j in range(N)])
        for T in range(N):
            w.writerow([f"T{T + 1}"] + ["" if v is None else round(100 * v, 4) for v in R[T]])
    with open(os.path.join(out, "all_results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model_after_T", "scored_task_j", "task", "score_x100", "file", "modified"])
        for (T, j), c in sorted(cells.items()):
            w.writerow([T, j, c["dataset"], round(100 * c["score"], 4), os.path.relpath(c["file"], run),
                        datetime.datetime.fromtimestamp(c["mtime"]).isoformat(timespec="seconds")])
    json.dump({"run": run, "tasks": label, "N": N, "LA": LA, "OP": OP, "BWT": BWT,
               "BWT_without_ScienceQA": BWT_noSQA, "R": R, "eval_suffix": suffix,
               "missing_diagonal": miss_diag, "missing_last_row": miss_last},
              open(os.path.join(out, "metrics.json"), "w"), indent=2)
    return {"run": run, "N": N, "LA": LA, "OP": OP, "BWT": BWT, "BWT_noSQA": BWT_noSQA,
            "cells": len(cells), "missing_diag": len(miss_diag), "missing_last": len(miss_last),
            "text": text}


def resolve_run(model, launcher, settings):
    """Run folder for these settings, named by the launcher itself: its code up
    to the line that sets BASE_DIR is run (that part only reads and checks
    files; nothing is created or submitted) with the settings as environment."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sdir = os.path.join(root, "scripts")
    path = os.path.join(sdir, {"plain": "run_setafix.sh",
                               "order": "run_setafix_order.sh"}[launcher])
    if not os.path.isfile(path):
        sys.exit(f"ERROR: launcher not found: {path}")
    lines = open(path).read().splitlines()
    cut = next((i for i, l in enumerate(lines) if l.startswith("BASE_DIR=")), None)
    if cut is None:
        sys.exit(f"ERROR: no 'BASE_DIR=' line in {path}")
    tmp = os.path.join(sdir, f".collect_basedir_{os.getpid()}.sh")
    with open(tmp, "w") as fh:     # next to the launcher: its 'source sos_common*.sh' is relative
        fh.write("\n".join(lines[:cut + 1]) + '\necho "__BASE_DIR__=${BASE_DIR}"\n')
    try:
        p = subprocess.run(["bash", tmp, "1", "fft,split,train,eval", model],
                           env={**os.environ, **settings}, capture_output=True, text=True)
    finally:
        os.remove(tmp)
    base = next((l.split("=", 1)[1] for l in p.stdout.splitlines()
                 if l.startswith("__BASE_DIR__=")), "")
    if p.returncode != 0 or not base:
        sys.stderr.write(p.stderr)
        sys.exit(f"ERROR: {os.path.basename(path)} refused these settings (above).")
    print(f"launcher : {os.path.basename(path)}   model: {model}")
    print(f"run      : {base}")
    if not os.path.isdir(base):
        others = sorted(glob.glob(os.path.join(os.path.dirname(base), "ablation_*")))
        sys.exit("ERROR: that folder does not exist — the settings differ from the training run.\n"
                 "       Runs present for this model:\n"
                 + "\n".join("         " + os.path.basename(o) for o in others))
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run folder(s) .../ablation_<tag>, and/or KEY=VALUE "
                    "training settings (with --model)")
    ap.add_argument("--model", default="", help="qwen3_4b | llama2_7b | ... : find the run folder "
                    "from the model + KEY=VALUE settings")
    ap.add_argument("--launcher", choices=("plain", "order"), default="plain",
                    help="naming of scripts/run_setafix.sh (plain) or run_setafix_order.sh (order)")
    ap.add_argument("--all", default="", help="collect every ablation_* run below this folder "
                    "(e.g. trained_models)")
    ap.add_argument("--match", default="", help="with --all: only runs whose folder name contains this")
    ap.add_argument("--suffix", default="", help="EVAL_OUT_SUFFIX of the eval folders to use (default none)")
    a = ap.parse_args()

    settings = dict(x.split("=", 1) for x in a.runs if "=" in x and not os.path.exists(x))
    runs = [x for x in a.runs if x.split("=", 1)[0] not in settings]
    if settings and not a.model:
        ap.error("KEY=VALUE settings need --model")
    if not a.suffix:
        a.suffix = settings.get("EVAL_OUT_SUFFIX", "")
    if a.model:
        runs.append(resolve_run(a.model, a.launcher, settings))
    if a.all:
        runs += sorted(p for p in glob.glob(os.path.join(a.all, "**", "ablation_*"), recursive=True)
                       if os.path.isdir(p) and a.match in os.path.basename(p)
                       and "/results" not in p)
    if not runs:
        ap.error("give --model with settings, run folder(s), or --all <folder>")

    rows = []
    for run in runs:
        r = collect(run, a.suffix)
        print("=" * 78)
        if r is None:
            print(f"{run}\n  no shared_moE_checkpoints_* folders — not a run folder, skipped")
            continue
        print(r["text"])
        print(f"-> {os.path.join(r['run'], 'results')}/")
        rows.append(r)

    if a.all and rows:
        path = os.path.join(os.path.abspath(a.all), "results_summary.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["run", "N", "LA", "OP", "BWT", "BWT_without_ScienceQA",
                        "cells", "missing_diagonal", "missing_last_row"])
            for r in rows:
                f = lambda x: "" if x is None else round(100 * x, 2)
                w.writerow([os.path.relpath(r["run"], os.path.abspath(a.all)), r["N"], f(r["LA"]),
                            f(r["OP"]), f(r["BWT"]), f(r["BWT_noSQA"]), r["cells"],
                            r["missing_diag"], r["missing_last"]])
        print("=" * 78)
        print(f"summary of {len(rows)} run(s) -> {path}")


if __name__ == "__main__":
    main()