#!/usr/bin/env python3
"""
cl_metrics.py — LA, OP and BWT for one run tree, read from the eval outputs the
trainer already writes. No model is loaded.

    python utils/cl_metrics.py --base_dir <trained_models/.../ablation_<tag>> \
                         --model qwen3_4b --task_list "C-STANCE,FOMC,..." --last 8 [--grid]

Score of checkpoint m on task j (the trainer's own output location):
    <base_dir>/shared_moE_checkpoints_<model>_<m>_<task_m lower>/eval_task_<j>_<task_j>/evaluation_summary.json
    -> results[<task_j>].metric_value

With T = --last (tasks 1..T):
    diagonal  a_jj   task j evaluated right after training on it
    final row a_Tj   checkpoint T evaluated on task j   (a_TT is the diagonal entry)
    LA  = mean_j a_jj                         over j = 1..T
    OP  = mean_j a_Tj                         over j = 1..T
    BWT = mean_j (a_Tj - a_jj)                over j = 1..T-1   (negative = forgetting)
    Mean forgetting = -BWT
Cells that are missing (or scored -1 / error) are left out and listed.

--grid also prints every lower-triangular cell that exists (cumulative evals).
Writes <base_dir>/cl_metrics_T<T>.json.
"""
import os, json, argparse


def score(base_dir, model, tasks, m, j):
    """Metric of checkpoint m on task j, or None."""
    name_m, name_j = tasks[m - 1], tasks[j - 1]
    p = os.path.join(base_dir, f"shared_moE_checkpoints_{model}_{m}_{name_m.lower()}",
                     f"eval_task_{j}_{name_j}", "evaluation_summary.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            res = json.load(f).get("results", {}).get(name_j, {})
    except (OSError, ValueError):
        return None
    v = res.get("metric_value")
    if v is None or "error" in res:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v == -1 else v


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x, signed=False):
    if x is None:
        return "   ---"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task_list", required=True, help="comma separated, in training order")
    ap.add_argument("--last", type=int, required=True, help="T: last trained task id")
    ap.add_argument("--grid", action="store_true", help="also print every lower-triangular cell found")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tasks = [t.strip() for t in a.task_list.split(",") if t.strip()]
    T = a.last
    if not (1 <= T <= len(tasks)):
        raise SystemExit(f"--last {T} is outside the task list (1..{len(tasks)})")

    diag  = [score(a.base_dir, a.model, tasks, j, j) for j in range(1, T + 1)]
    final = [score(a.base_dir, a.model, tasks, T, j) for j in range(1, T + 1)]

    la  = mean([d for d in diag if d is not None])
    op  = mean([f for f in final if f is not None])
    dif = [final[j] - diag[j] for j in range(T - 1)
           if diag[j] is not None and final[j] is not None]
    bwt = mean(dif)
    fgt = -bwt if bwt is not None else None

    w = max(len(t) for t in tasks[:T]) + 2
    print("")
    print("=" * 60)
    print(f"  CL metrics  (T = {T}: checkpoint {T} = {tasks[T - 1]})")
    print("=" * 60)
    print(f"  {'Task':<{w}}  {'diag':>7}  {'final':>7}  {'final-diag':>10}")
    for j in range(T):
        d = (final[j] - diag[j]) if (j < T - 1 and diag[j] is not None and final[j] is not None) else None
        print(f"  {tasks[j]:<{w}}  {fmt(diag[j]):>7}  {fmt(final[j]):>7}  {fmt(d, True):>10}")
    print("-" * 60)
    print(f"  Learning Accuracy (LA)     {fmt(la)}")
    print(f"  Overall Performance (OP)   {fmt(op)}")
    print(f"  Backward Transfer (BWT)    {fmt(bwt, True)}   (over tasks 1..{T - 1})")
    print(f"  Mean Forgetting            {fmt(fgt, True)}")
    miss_d = [tasks[j] for j in range(T) if diag[j] is None]
    miss_f = [tasks[j] for j in range(T) if final[j] is None]
    if miss_d:
        print(f"  missing diagonal : {', '.join(miss_d)}")
    if miss_f:
        print(f"  missing final row: {', '.join(miss_f)}")
    print("=" * 60)

    grid = None
    if a.grid:
        grid = [[score(a.base_dir, a.model, tasks, m, j) if j <= m else None
                 for j in range(1, T + 1)] for m in range(1, T + 1)]
        cw = 11
        print("\n  Grid (row = checkpoint, column = task)")
        print("  " + " " * w + "".join(f"{t[:10]:>{cw}}" for t in tasks[:T]))
        for m in range(T):
            print(f"  {tasks[m]:<{w}}" + "".join(f"{fmt(x):>{cw}}" for x in grid[m]))

    out = a.out or os.path.join(a.base_dir, f"cl_metrics_T{T}.json")
    rec = {
        "tasks": tasks[:T],
        "diagonal": dict(zip(tasks[:T], diag)),
        "final_row": dict(zip(tasks[:T], final)),
        "learning_accuracy": la,
        "overall_performance": op,
        "backward_transfer": bwt,
        "mean_forgetting": fgt,
        "missing_diagonal": miss_d,
        "missing_final_row": miss_f,
    }
    if grid is not None:
        rec["grid"] = {tasks[m]: dict(zip(tasks[:T], grid[m])) for m in range(T)}
    try:
        with open(out, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"  -> {out}")
    except OSError as e:
        print(f"  (could not write {out}: {e})")


if __name__ == "__main__":
    main()
