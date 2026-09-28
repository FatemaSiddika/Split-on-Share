#!/usr/bin/env python3
"""
setafix_report.py — what did the split produce across the task stream?

Reads, per task dir under one or more BASE_DIRs:
    shared_moE_checkpoints_<model>_<t>_<task>/init_structure_checkpoint/expert_definitions.json
    shared_moE_checkpoints_<model>_<t>_<task>/setafix_split_report_T<t>.json   (seam runs)

Reports per task: number of experts, size of the task's UNIQUE expert
(exp_T<t>), total shared blocks, co-resident blocks (a position held by >1
expert = rejected overlap kept for both), share/reject decisions and mean
alignment score.  Pass two trees to compare, e.g. original split vs SEAM-SoS:

    python setafix_report.py <BASE_DIR_seam> [<BASE_DIR_off>] [--plot]

No torch needed.
"""
import os, re, sys, json, glob
from collections import defaultdict

def _digits(key):
    s = str(key)
    return [int(x) for x in re.findall(r"\d+", s)] if "-" in s else [int(c) for c in re.findall(r"\d", s)]

def _blocks(d):
    return sum(len(v) for g in ("mlp", "attn") for v in d.get(g, {}).values())

def scan(base):
    rows = []
    for p in sorted(glob.glob(os.path.join(base, "shared_moE_checkpoints_*_*_*"))):
        m = re.search(r"_(\d+)_([^/]+)$", os.path.basename(p))
        if not m:
            continue
        t, name = int(m.group(1)), m.group(2)
        dj = os.path.join(p, "init_structure_checkpoint", "expert_definitions.json")
        if not os.path.exists(dj):
            dj = os.path.join(p, "expert_definitions.json")
        if not os.path.exists(dj):
            continue
        defs = {k: v for k, v in json.load(open(dj)).items() if isinstance(v, dict)}
        occ = defaultdict(int)
        for k, d in defs.items():
            for g in ("mlp", "attn"):
                for lk, pos in d.get(g, {}).items():
                    for pp in pos:
                        occ[(lk, tuple(pp))] += 1
        cores = sum(1 for v in occ.values() if v > 1)
        uniq = _blocks(defs.get(f"exp_T{t}", {}))
        shared = sum(_blocks(d) for k, d in defs.items() if len(_digits(k)) > 1)
        rep_p = os.path.join(p, f"setafix_split_report_T{t}.json")
        rep = json.load(open(rep_p)) if os.path.exists(rep_p) else {}
        c = rep.get("counts", {})
        scores = [b["score"] for b in rep.get("blocks", []) if b.get("score") is not None]
        rows.append(dict(t=t, name=name, n_exp=len(defs), unique=uniq, shared=shared,
                         total=sum(_blocks(d) for d in defs.values()), coresident=cores,
                         share=c.get("share", 0), reject=c.get("reject", 0),
                         noev=c.get("no_evidence", 0), skipped=c.get("expert_split_skipped", 0),
                         mean_score=(sum(scores) / len(scores)) if scores else float("nan")))
    return sorted(rows, key=lambda r: r["t"])

def table(rows, title):
    hdr = (f"{'t':<3}{'task':<13}{'experts':>8}{'unique':>8}{'shared':>8}{'total':>7}"
           f"{'co-res':>8}{'share':>7}{'reject':>8}{'no-ev':>7}{'skip':>6}{'align':>8}")
    print(f"\n{title}\n{hdr}\n{'-'*len(hdr)}")
    for r in rows:
        print(f"{r['t']:<3}{r['name'][:12]:<13}{r['n_exp']:>8}{r['unique']:>8}{r['shared']:>8}"
              f"{r['total']:>7}{r['coresident']:>8}{r['share']:>7}{r['reject']:>8}{r['noev']:>7}"
              f"{r['skipped']:>6}{r['mean_score']:>8.3f}")

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    trees = {os.path.basename(a.rstrip("/")): scan(a) for a in args}
    for name, rows in trees.items():
        table(rows, name)
    if "--plot" in sys.argv:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for name, rows in trees.items():
            ts = [r["t"] for r in rows]
            ax[0].plot(ts, [r["unique"] for r in rows], marker="o", lw=2, label=name[:28])
            ax[1].plot(ts, [r["n_exp"] for r in rows], marker="s", lw=2, label=name[:28])
        ax[0].set_title("unique-expert blocks per task"); ax[1].set_title("experts alive")
        for a in ax: a.set_xlabel("task"); a.legend(fontsize=8); a.grid(False)
        out = os.path.join(args[0], "setafix_structure.png")
        fig.tight_layout(); fig.savefig(out, dpi=200); print(f"\nplot -> {out}")

if __name__ == "__main__":
    main()
