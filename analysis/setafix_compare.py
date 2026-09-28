#!/usr/bin/env python3
"""
setafix_compare.py — compare two (or more) Split-on-Share structure trees, e.g. the
SEAM split at two block budgets (15% vs 25%), with the same plots as the
SoS-vs-SEAM comparison plus budget-normalised views.

    python setafix_compare.py TREE_A TREE_B --labels "SEAM 15%","SEAM 25%" [--out DIR]

TREE = .../RL_Lamda_fsys_trained_models_SoS_<model>_<proj>/ablation_<tag>
(the run tag carries the budget, e.g. ..._bbf12_... vs ..._bbf25_...).

Outputs (default: TREE_A/setafix_compare/):
    compare_per_task.csv, compare_tables.tex, compare_summary.md
    fig_structure_over_tasks   unique blocks | experts alive | shared blocks
    fig_budget_normalised      unique as % of selected | shared as % of footprint
    fig_final_summary          grouped bars, final state
    fig_unique_delta           per-task unique difference (B - A)
No torch needed. Reuses the scanner from setafix_analyze.py (same folder).
"""
import os, sys, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from setafix_analyze import scan_tree, find_fft_dir, task_dirs  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 20
plt.rcParams.update({"font.size": FS, "axes.titlesize": FS + 1, "axes.labelsize": FS,
                     "xtick.labelsize": FS - 2, "ytick.labelsize": FS - 2,
                     "legend.fontsize": FS - 3, "axes.grid": False})
COLORS = ["#2A7F62", "#1F3A5F", "#D9A441", "#B4532A"]


def savefig(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=220, bbox_inches="tight")
    fig.savefig(os.path.join(out, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trees", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels = a.labels.split(",") if a.labels else [os.path.basename(t.rstrip("/"))[-20:] for t in a.trees]
    out = a.out or os.path.join(a.trees[0].rstrip("/"), "setafix_compare")
    os.makedirs(out, exist_ok=True)

    data = {}
    for lab, tree in zip(labels, a.trees):
        rows, _, _, sizes = scan_tree(tree.rstrip("/"), find_fft_dir(tree.rstrip("/"), []))
        if not rows:
            sys.exit(f"no task directories found under {tree}")
        data[lab] = (rows, sizes)
        print(f"[tree] {lab}: {len(rows)} tasks  ({tree})")

    # ── per-task CSV + markdown ──────────────────────────────────────────
    with open(os.path.join(out, "compare_per_task.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tree", "task", "name", "selected", "unique", "unique_pct", "shared", "shared_pct_footprint",
                    "footprint", "experts", "coresident", "median_expert"])
        for lab, (rows, _) in data.items():
            for r in rows:
                w.writerow([lab, r["task"], r["name"], r["selected"], r["unique"],
                            f"{pct(r['unique'], r['selected']):.1f}", r["shared_total"],
                            f"{pct(r['shared_total'], r['footprint']):.1f}", r["footprint"], r["experts"],
                            r["coresident"], r["median_expert"]])
    md = ["# budget comparison\n"]
    for lab, (rows, _) in data.items():
        md.append(f"\n## {lab}\n\n| task | selected | unique | unique % | shared | footprint | experts | co-res |\n|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['task']} {r['name'][:10]} | {r['selected']} | {r['unique']} | {pct(r['unique'], r['selected']):.0f}% | "
                      f"{r['shared_total']} | {r['footprint']} | {r['experts']} | {r['coresident']} |")
    open(os.path.join(out, "compare_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    # ── fig 1: structure over tasks (absolute) ──────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.8))
    for i, (lab, (rows, _)) in enumerate(data.items()):
        t = [r["task"] for r in rows]
        ax[0].plot(t, [r["unique"] for r in rows], marker="o", lw=3, ms=9, color=COLORS[i], label=lab)
        ax[1].plot(t, [r["experts"] for r in rows], marker="s", lw=3, ms=9, color=COLORS[i], label=lab)
        ax[2].plot(t, [r["shared_total"] for r in rows], marker="^", lw=3, ms=9, color=COLORS[i], label=lab)
    for a_, title, yl in zip(ax, ["unique-expert blocks per task", "experts alive", "blocks held by shared experts"],
                             ["blocks", "experts", "blocks"]):
        a_.set_title(title); a_.set_ylabel(yl); a_.set_xlabel("task"); a_.legend(frameon=False)
        a_.set_xticks([r["task"] for r in next(iter(data.values()))[0]])
    savefig(fig, out, "fig_structure_over_tasks")

    # ── fig 2: budget-normalised ────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.8))
    for i, (lab, (rows, _)) in enumerate(data.items()):
        t = [r["task"] for r in rows]
        ax[0].plot(t, [pct(r["unique"], r["selected"]) for r in rows], marker="o", lw=3, ms=9, color=COLORS[i], label=lab)
        ax[1].plot(t, [pct(r["shared_total"], r["footprint"]) for r in rows], marker="^", lw=3, ms=9, color=COLORS[i], label=lab)
        ax[2].plot(t, [r["footprint"] for r in rows], marker="s", lw=3, ms=9, color=COLORS[i], label=lab)
    ax[0].set_title("share of its own blocks a task keeps"); ax[0].set_ylabel("% of selected"); ax[0].set_ylim(0, 105)
    ax[1].set_title("shared blocks as % of footprint"); ax[1].set_ylabel("%")
    ax[2].set_title("total blocks held (footprint)"); ax[2].set_ylabel("blocks")
    for a_ in ax:
        a_.set_xlabel("task"); a_.legend(frameon=False)
        a_.set_xticks([r["task"] for r in next(iter(data.values()))[0]])
    savefig(fig, out, "fig_budget_normalised")

    # ── fig 3: final-state grouped bars ─────────────────────────────────
    metrics = [("mean unique\nblocks (t≥2)", lambda r: sum(x["unique"] for x in r[1:]) / max(len(r) - 1, 1)),
               ("mean unique %\n(t≥2)", lambda r: sum(pct(x["unique"], x["selected"]) for x in r[1:]) / max(len(r) - 1, 1)),
               ("median expert\nsize (final)", lambda r: r[-1]["median_expert"]),
               ("experts\n(final)", lambda r: r[-1]["experts"]),
               ("footprint\n(final) / 10", lambda r: r[-1]["footprint"] / 10)]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = range(len(metrics)); n = len(data); w = 0.8 / n
    for i, (lab, (rows, _)) in enumerate(data.items()):
        vals = [f(rows) for _, f in metrics]
        bars = ax.bar([xx + (i - (n - 1) / 2) * w for xx in x], vals, w, color=COLORS[i], label=lab)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=FS - 5)
    ax.set_xticks(list(x)); ax.set_xticklabels([m for m, _ in metrics], fontsize=FS - 4)
    ax.set_title("final-state comparison"); ax.legend(frameon=False)
    savefig(fig, out, "fig_final_summary")

    # ── fig 4: per-task unique delta (B − A) ────────────────────────────
    if n >= 2:
        (la, (ra, _)), (lb, (rb, _)) = list(data.items())[:2]
        t = [r["task"] for r in ra[:len(rb)]]
        d_abs = [b["unique"] - a_["unique"] for a_, b in zip(ra, rb)]
        d_pct = [pct(b["unique"], b["selected"]) - pct(a_["unique"], a_["selected"]) for a_, b in zip(ra, rb)]
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
        for a_, dd, title, yl in ((ax[0], d_abs, f"unique blocks: {lb} − {la}", "blocks"),
                                  (ax[1], d_pct, f"unique % of selected: {lb} − {la}", "percentage points")):
            a_.bar(t, dd, color=[COLORS[0] if v >= 0 else COLORS[3] for v in dd])
            for tt, v in zip(t, dd):
                a_.annotate(f"{v:+.0f}", (tt, v), xytext=(0, 4 if v >= 0 else -18), textcoords="offset points",
                            ha="center", fontsize=FS - 4)
            a_.axhline(0, color="black", lw=1); a_.set_xticks(t); a_.set_xlabel("task"); a_.set_ylabel(yl); a_.set_title(title)
        savefig(fig, out, "fig_unique_delta")

    # ── LaTeX ───────────────────────────────────────────────────────────
    labs = list(data)
    L = ["% auto-generated by setafix_compare.py (\\usepackage{booktabs,multirow})",
         "\\begin{table}[t]\\centering",
         "\\caption{SEAM-SoS at two block budgets (structure-only pass). Unique\\% = unique-expert blocks as a share of the task's selected blocks.}",
         "\\label{tab:seam_budget}\\small",
         "\\begin{tabular}{ll" + "rrrr" * len(labs) + "}", "\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{l}}}" for l in labs) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{3 + 4 * i}-{6 + 4 * i}}}" for i in range(len(labs))),
         "$t$ & task & " + " & ".join("Sel. & Unique & Unique\\% & Experts" for _ in labs) + " \\\\", "\\midrule"]
    nrows = min(len(r) for r, _ in data.values())
    for i in range(nrows):
        r0 = data[labs[0]][0][i]
        cells = [f"{data[l][0][i]['selected']} & {data[l][0][i]['unique']} & "
                 f"{pct(data[l][0][i]['unique'], data[l][0][i]['selected']):.0f} & {data[l][0][i]['experts']}" for l in labs]
        L.append(f"{r0['task']} & {r0['name'][:11]} & " + " & ".join(cells) + " \\\\")
    L += ["\\midrule"]
    cells = []
    for l in labs:
        rows = data[l][0]
        cells.append(f"-- & {sum(x['unique'] for x in rows[1:]) / max(len(rows) - 1, 1):.0f} & "
                     f"{sum(pct(x['unique'], x['selected']) for x in rows[1:]) / max(len(rows) - 1, 1):.0f} & {rows[-1]['experts']}")
    L.append("\\multicolumn{2}{l}{mean ($t\\ge2$) / final} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(out, "compare_tables.tex"), "w").write("\n".join(L) + "\n")
    print(f"\n-> outputs in {out}")


if __name__ == "__main__":
    main()



# python setafix_compare.py \
#   ../trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_*_bbf12_*_setafix_seam_svd_structonly \
#   ../trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_*_bbf25_*_setafix_seam_svd_structonly \
#   --labels "SEAM 12%","SEAM 25%"



# cd /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis
# python setafix_compare.py \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/SETAV2/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf25_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000* \
#   ../trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_*bbf25*setafix_seam_svd_structonly \
#   --labels "SoS 25%","SEAM 25%" \
#   --out /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis/compare_sos_vs_seam_25


# python setafix_compare.py \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/SETAV2/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_gatefz_sept18 \
#   --labels "SoS 12%","SEAM 12%" \
#   --out /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis/compare_sos_vs_seam_12



# python setafix_compare.py \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_gatefz_sept18 \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn0025_gatefz \
#   --labels "seam_18Ssept 12%","seam_fn0025 12%" \
#   --out /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis/compare_seam_18Ssept_vs_seam_fn0025



# python setafix_compare.py \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn0015_lgbud0015_gatefz \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/SETAV2/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix \
#   --labels "seam_015 12%","SETAV2 12%" \
#   --out /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis/compare_seam_22Sept_015_vs_SETAV2_qwen


# python setafix_compare.py \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn002_lgbud002_gatefz \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/SETAV2/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix \
#   --labels "seam_020 12%","SETAV2 12%" \
#   --out /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis/compare_seam_22Sept_020_vs_SETAV2_qwen




# python setafix_compare.py \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn0015_lgbud0015_gatefz \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn002_lgbud002_gatefz \
#   --labels "seam_015 12%","seam_020 12%" \
#   --out /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis/compare_seam_22Sept_015_vs_seam_22Sept_020


# python setafix_compare.py \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn0012_lgbud0012_gatefz \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn002_lgbud002_gatefz \
#   --labels "seam_012 12%","seam_020 12%" \
#   --out /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis/compare_seam_22Sept_012_vs_seam_22Sept_020


# python setafix_compare.py \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn0012_lgbud0012_gatefz \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn0015_lgbud0015_gatefz \
#  --labels "seam_012 12%","seam_015 12%" \
#   --out /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis/compare_seam_22Sept_012_vs_seam_22Sept_015


# python setafix_compare.py \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix_setafix_seam_svd_empirical_ortho_fn0012_lgbud0012_gatefz \
#   /lustre/hdd/LAS/jannesar-lab/fatemask/SETAV2/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_full_system_bbf12_ord1_v_proj_only_seed100_RL_Lamda_dynamic_grlFalse_gatesigmoid_rrnone_d5000_100_fix \
#   --labels "seam_012 12%","SETAV2 12%" \
#   --out /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share/analysis/compare_seam_22Sept_012_vs_SETAV2_qwen

