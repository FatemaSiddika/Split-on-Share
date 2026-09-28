#!/usr/bin/env python3
"""
setafix_analyze.py — analysis & diagnosis of the expert structure produced by
the split (original vs SEAM-SoS), from structure-only or full runs.

    python setafix_analyze.py TREE [TREE ...] [--fft DIR ...] [--out DIR] [--labels a,b]

TREE = a run directory  .../RL_Lamda_fsys_trained_models_SoS_<model>_<proj>/ablation_<tag>
       containing shared_moE_checkpoints_<model>_<t>_<task>/init_structure_checkpoint/
FFT  = matching FFT_steps_<model>_<tag>_Step*_bgt* directory (auto-discovered next to
       the trees' parent if not given) — needed for the selected budget per task,
       the gradient tiles (task-similarity heatmap) and the split reports.

Outputs (in --out, default TREE1/setafix_analysis):
    summary.md, per_task.csv, blocks_provenance.csv, and figures:
    fig1_unique_vs_budget, fig2_experts_footprint, fig3_provenance,
    fig4_alignment_scores, fig5_expert_sizes_final, fig6_task_similarity (torch),
    fig7_per_layer_reject, fig8_layer_profile

Everything degrades gracefully: missing files -> that figure/column is skipped.
"""
import os, re, sys, json, glob, csv, math, argparse
from collections import defaultdict, OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 18
plt.rcParams.update({"font.size": FS, "axes.titlesize": FS + 1, "axes.labelsize": FS,
                     "xtick.labelsize": FS - 2, "ytick.labelsize": FS - 2,
                     "legend.fontsize": FS - 3, "axes.grid": False})
COLORS = ["#2A7F62", "#8A8880", "#D9A441", "#1F3A5F", "#B4532A"]
PROV_COL = {"novel": "#1F3A5F", "shared": "#2A7F62", "own-copy": "#D9A441"}


# ───────────────────────── helpers ──────────────────────────────────────
def digits(key):
    s = str(key)
    return [int(x) for x in re.findall(r"\d+", s)] if "-" in s else [int(c) for c in re.findall(r"\d", s)]


def n_blocks(d):
    return sum(len(v) for g in ("mlp", "attn") for v in d.get(g, {}).values())


def blocks_of(d):
    out = set()
    for g in ("mlp", "attn"):
        for lk, pos in d.get(g, {}).items():
            lk = tuple(lk) if isinstance(lk, list) else lk
            for p in pos:
                out.add((str(lk), tuple(int(x) for x in p)))
    return out


def parse_layer_key(lk):
    """expert_definitions layer key -> (proj, layer). Keys are saved as strings
    like "('v_proj', 3)" or "v_proj_3" or lists ['v_proj', 3]."""
    if isinstance(lk, (list, tuple)) and len(lk) == 2:
        return str(lk[0]), int(lk[1])
    s = str(lk)
    m = re.search(r"'?([a-z_]+_proj)'?\D+(\d+)", s)
    return (m.group(1), int(m.group(2))) if m else (s, -1)


def load_defs(path):
    if not os.path.exists(path):
        return None
    raw = json.load(open(path))
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def task_dirs(tree):
    out = []
    for p in sorted(glob.glob(os.path.join(tree, "shared_moE_checkpoints_*_*_*"))):
        m = re.search(r"shared_moE_checkpoints_(.+)_(\d+)_([^/]+)$", os.path.basename(p))
        if m:
            out.append((int(m.group(2)), m.group(3), m.group(1), p))
    return sorted(out)


def find_fft_dir(tree, given):
    tag = os.path.basename(tree.rstrip("/")).replace("ablation_", "", 1)
    cands = list(given or [])
    root = os.path.dirname(os.path.dirname(tree.rstrip("/")))
    cands += glob.glob(os.path.join(os.path.dirname(root), "FFT_steps", f"FFT_steps_*{tag}*"))  # Split-on-Share layout
    cands += glob.glob(os.path.join(root, f"FFT_steps_*{tag}*"))
    cands += glob.glob(os.path.join(os.getcwd(), f"FFT_steps_*{tag}*"))
    for c in cands:
        if os.path.isdir(c) and tag in os.path.basename(c):
            return c
    return None


def _norm(x):
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def selected_positions(fft_dir, task_name, model):
    """Blocks selected by FFT for this task: {(proj,layer): [(r,c),...]}.
    Matching is fuzzy (case/punctuation-insensitive) because task directories
    are lower-cased while FFT files use the original dataset names."""
    if not fft_dir:
        return {}
    want = _norm(f"{task_name}_{model}")
    best = None
    for f in glob.glob(os.path.join(fft_dir, "selected_submatrix_attention_*.json")):
        b = os.path.basename(f)[len("selected_submatrix_attention_"):-len(".json")]
        ranks = b.endswith("_with_ranks")
        if _norm(b.replace("_with_ranks", "")) == want:
            if best is None or (not ranks and best[1]):      # prefer the plain file
                best = (f, ranks)
    if best is None:
        return {}
    data = json.load(open(best[0]))
    out = {}
    for k, v in data.items():
        parts = k.rsplit("_layer_", 1)
        if len(parts) == 2:
            out[(parts[0], int(parts[1]))] = [tuple(int(x) for x in it) for it in v if len(it) == 2]
    return out


# ───────────────────────── per-tree scan ─────────────────────────────────
def scan_tree(tree, fft_dir):
    rows, prov_rows, reports = [], [], {}
    prev_all = set()  # every block owned by any expert before task t
    for t, name, model, tdir in task_dirs(tree):
        defs = load_defs(os.path.join(tdir, "init_structure_checkpoint", "expert_definitions.json")) \
            or load_defs(os.path.join(tdir, "expert_definitions.json"))
        if defs is None:
            continue
        # FFT-selected budget (cross-check only; may be missing)
        sel = selected_positions(fft_dir, name, model)
        fft_selected = sum(len(v) for v in sel.values())
        occ = defaultdict(int)
        exp_blocks = {}
        for k, d in defs.items():
            b = set()
            for g in ("mlp", "attn"):
                for lk, pos in d.get(g, {}).items():
                    pl = parse_layer_key(lk)
                    for pp in pos:
                        key = (str(pl), tuple(int(x) for x in pp))
                        b.add(key); occ[key] += 1
            exp_blocks[k] = b
        unique_key = f"exp_T{t}"
        uniq = exp_blocks.get(unique_key, set())
        # EXACT selection of task t = every block owned by an expert stamped task_id == t
        # (its unique expert + the shared children carved for it this task)
        sel_blocks = set().union(*[exp_blocks[k] for k, d in defs.items()
                                   if d.get("task_id") == t]) if defs else set()
        shared_keys = [k for k in defs if len(digits(k)) > 1]
        shared_now = set().union(*[exp_blocks[k] for k in shared_keys]) if shared_keys else set()
        coresident = {k for k, v in occ.items() if v > 1}
        all_now = set().union(*exp_blocks.values()) if exp_blocks else set()
        # provenance of this task's selected blocks
        if sel_blocks:
            novel = len(sel_blocks - prev_all)
            own_copy = len((sel_blocks & prev_all) & uniq)
            shared_abs = len((sel_blocks & prev_all) - uniq)
        else:
            novel = own_copy = shared_abs = 0
        rep_p = os.path.join(tdir, f"setafix_split_report_T{t}.json")
        rep = json.load(open(rep_p)) if os.path.exists(rep_p) else {}
        reports[t] = rep
        c = rep.get("counts", {})
        scores = [b["score"] for b in rep.get("blocks", []) if b.get("score") is not None]
        sizes = sorted((len(v) for v in exp_blocks.values()), reverse=True)
        per_layer = defaultdict(set)          # layer -> experts present in it
        for k, b in exp_blocks.items():
            for lk, _ in b:
                per_layer[lk].add(k)
        rows.append(OrderedDict(
            task=t, name=name, selected=len(sel_blocks), fft_selected=fft_selected, unique=len(uniq),
            unique_frac=(len(uniq) / len(sel_blocks)) if sel_blocks else float("nan"),
            shared_total=len(shared_now), footprint=len(all_now), experts=len(defs),
            shared_experts=len(shared_keys), coresident=len(coresident),
            novel=novel, shared_absorbed=shared_abs, own_copy=own_copy,
            share=c.get("share", 0), reject=c.get("reject", 0), no_evidence=c.get("no_evidence", 0),
            split_skipped=c.get("expert_split_skipped", 0),
            align_mean=(sum(scores) / len(scores)) if scores else float("nan"),
            median_expert=(sizes[len(sizes) // 2] if sizes else 0), smallest_expert=(sizes[-1] if sizes else 0),
            largest_expert=(sizes[0] if sizes else 0),
            max_experts_per_layer=max(len(v) for v in per_layer.values()) if per_layer else 0,
        ))
        prov_rows.append((t, name, novel, shared_abs, own_copy))
        prev_all = all_now
    final_sizes = sizes if rows else []
    return rows, prov_rows, reports, final_sizes


# ───────────────────────── figures ──────────────────────────────────────
def savefig(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_headline(trees, out):
    """One-look comparison: unique blocks per task, experts alive per task, and
    the final-state summary bars."""
    labs = list(trees)
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.6))
    for i, lab in enumerate(labs):
        rows = trees[lab]; ts = [r["task"] for r in rows]
        ax[0].plot(ts, [r["unique"] for r in rows], marker="o", lw=2.8, ms=8, color=COLORS[i], label=lab)
        ax[1].plot(ts, [r["experts"] for r in rows], marker="s", lw=2.8, ms=8, color=COLORS[i], label=lab)
    ax[0].set_title("unique-expert blocks per task"); ax[0].set_xlabel("task"); ax[0].set_ylabel("blocks")
    ax[1].set_title("experts alive"); ax[1].set_xlabel("task"); ax[1].set_ylabel("experts")
    metrics = [("mean unique\n(t\u22652)", lambda r: sum(x["unique"] for x in r[1:]) / max(len(r) - 1, 1)),
               ("median expert\nsize (final)", lambda r: r[-1]["median_expert"]),
               ("experts\n(final)", lambda r: r[-1]["experts"]),
               ("shared blocks\n(final) /10", lambda r: r[-1]["shared_total"] / 10)]
    x = range(len(metrics)); w = 0.8 / len(labs)
    for i, lab in enumerate(labs):
        vals = [f(trees[lab]) for _, f in metrics]
        bars = ax[2].bar([xx + (i - (len(labs) - 1) / 2) * w for xx in x], vals, w, color=COLORS[i], label=lab)
        for b, v in zip(bars, vals):
            ax[2].annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                           textcoords="offset points", ha="center", fontsize=FS - 5)
    ax[2].set_xticks(list(x)); ax[2].set_xticklabels([m for m, _ in metrics], fontsize=FS - 5)
    ax[2].set_title("final-state summary")
    for a in ax: a.legend(frameon=False)
    savefig(fig, out, "fig0_headline")


def fig_unique_vs_budget(trees, out):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    for i, (lab, rows) in enumerate(trees.items()):
        ts = [r["task"] for r in rows]
        if i == 0 and any(r["selected"] for r in rows):
            ax[0].bar(ts, [r["selected"] for r in rows], color="#DDDDDD", label="selected by FFT")
        ax[0].plot(ts, [r["unique"] for r in rows], marker="o", lw=2.6, color=COLORS[i], label=f"unique expert ({lab})")
        ax[1].plot(ts, [100 * r["unique_frac"] for r in rows], marker="o", lw=2.6, color=COLORS[i], label=lab)
    ax[0].set_title("unique-expert blocks per task"); ax[0].set_xlabel("task"); ax[0].set_ylabel("blocks")
    ax[1].set_title("share of the task's own blocks it keeps"); ax[1].set_xlabel("task"); ax[1].set_ylabel("% of selected")
    ax[1].set_ylim(0, 105)
    for a in ax: a.legend(frameon=False)
    savefig(fig, out, "fig1_unique_vs_budget")


def fig_experts(trees, out):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
    for i, (lab, rows) in enumerate(trees.items()):
        ts = [r["task"] for r in rows]
        ax[0].plot(ts, [r["experts"] for r in rows], marker="s", lw=2.6, color=COLORS[i], label=lab)
        ax[1].plot(ts, [r["footprint"] for r in rows], marker="o", lw=2.6, color=COLORS[i], label=lab)
        ax[2].plot(ts, [r["max_experts_per_layer"] for r in rows], marker="^", lw=2.6, color=COLORS[i], label=lab)
    for a, t, y in zip(ax, ["experts alive", "total blocks held (footprint)", "max experts in one layer"],
                       ["experts", "blocks", "experts"]):
        a.set_title(t); a.set_xlabel("task"); a.set_ylabel(y); a.legend(frameon=False)
    savefig(fig, out, "fig2_experts_footprint")


def fig_provenance(trees_prov, out):
    n = len(trees_prov)
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.6), sharey=True)
    axes = [axes] if n == 1 else list(axes)
    for a, (lab, prov) in zip(axes, trees_prov.items()):
        ts = [p[0] for p in prov]
        nov = [p[2] for p in prov]; sh = [p[3] for p in prov]; own = [p[4] for p in prov]
        a.bar(ts, nov, color=PROV_COL["novel"], label="novel (never touched)")
        a.bar(ts, sh, bottom=nov, color=PROV_COL["shared"], label="overlap → shared")
        a.bar(ts, own, bottom=[x + y for x, y in zip(nov, sh)], color=PROV_COL["own-copy"], label="overlap → own copy")
        a.set_title(f"where each task's blocks went — {lab}"); a.set_xlabel("task")
        tot = max([x + y + z for x, y, z in zip(nov, sh, own)] or [1])
        a.set_ylim(0, tot * 1.38)
        a.legend(frameon=False, loc="upper center", ncol=1)
    axes[0].set_ylabel("selected blocks")
    savefig(fig, out, "fig3_provenance")


def fig_alignment(reports_by_tree, out):
    made = False
    for lab, reports in reports_by_tree.items():
        share, reject, thr = [], [], []
        for t, rep in reports.items():
            for b in rep.get("blocks", []):
                if b.get("score") is None:
                    continue
                (share if b["decision"].startswith("share") else reject).append(b["score"])
                if b.get("thr") is not None:
                    thr.append(b["thr"])
        if not share and not reject:
            continue
        made = True
        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        bins = [x / 40 for x in range(41)]
        ax.hist(reject, bins=bins, color="#D9A441", alpha=0.85, label=f"rejected (n={len(reject)})")
        ax.hist(share, bins=bins, color="#2A7F62", alpha=0.85, label=f"shared (n={len(share)})")
        if thr:
            ax.axvline(sum(thr) / len(thr), color="#C0392B", lw=2.4, ls="--", label="mean chance floor")
        ax.set_xlabel("direction agreement (0 = unrelated, 1 = identical)"); ax.set_ylabel("blocks")
        ax.set_title(f"alignment of overlapping blocks — {lab}"); ax.legend(frameon=False)
        savefig(fig, out, f"fig4_alignment_scores_{lab}")
    return made


def fig_sizes(final_sizes, out):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for i, (lab, sizes) in enumerate(final_sizes.items()):
        if sizes:
            ax.plot(range(1, len(sizes) + 1), sizes, marker="o", lw=2.2, color=COLORS[i], label=f"{lab} ({len(sizes)} experts)")
    ax.set_yscale("log"); ax.set_xlabel("expert rank (largest → smallest)"); ax.set_ylabel("blocks (log)")
    ax.set_title("expert sizes after the last task"); ax.legend(frameon=False)
    savefig(fig, out, "fig5_expert_sizes_final")


def fig_per_layer_reject(reports_by_tree, out):
    made = False
    for lab, reports in reports_by_tree.items():
        if not reports:
            continue
        t_last = max(reports)
        per = defaultdict(lambda: [0, 0])
        for b in reports[t_last].get("blocks", []):
            lay = int(b["layer"].split("|")[1]) if "|" in b["layer"] else -1
            per[lay][0 if b["decision"].startswith("share") else 1] += 1
        if not per:
            continue
        made = True
        lays = sorted(per)
        fig, ax = plt.subplots(figsize=(13, 5.2))
        ax.bar(lays, [per[l][0] for l in lays], color="#2A7F62", label="shared")
        ax.bar(lays, [per[l][1] for l in lays], bottom=[per[l][0] for l in lays], color="#D9A441", label="own copy")
        ax.set_xlabel("layer"); ax.set_ylabel("overlapping blocks"); ax.set_title(f"task {t_last}: split decision per layer — {lab}")
        ax.legend(frameon=False)
        savefig(fig, out, f"fig7_per_layer_reject_{lab}")
    return made


def fig_layer_profile(trees_dirs, out):
    """blocks per layer of the last task's unique expert vs. all experts."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, (lab, tdirs) in enumerate(trees_dirs.items()):
        if not tdirs:
            continue
        t, name, model, tdir = tdirs[-1]
        defs = load_defs(os.path.join(tdir, "init_structure_checkpoint", "expert_definitions.json")) \
            or load_defs(os.path.join(tdir, "expert_definitions.json"))
        if not defs:
            continue
        per = defaultdict(int)
        for lk, pos in {**defs.get(f"exp_T{t}", {}).get("attn", {}), **defs.get(f"exp_T{t}", {}).get("mlp", {})}.items():
            per[parse_layer_key(lk)[1]] += len(pos)
        lays = sorted(per)
        ax.plot(lays, [per[l] for l in lays], marker="o", lw=2.2, color=COLORS[i], label=f"{lab}: exp_T{t}")
    ax.set_xlabel("layer"); ax.set_ylabel("blocks"); ax.set_title("last task's unique expert — blocks per layer")
    ax.legend(frameon=False)
    savefig(fig, out, "fig8_layer_profile")


def fig_task_similarity(fft_dir, task_names, model, out, label):
    """T x T mean direction agreement of FFT gradient tiles on commonly selected blocks."""
    try:
        import torch
    except ImportError:
        print("  [skip] torch not available -> no task-similarity heatmap")
        return False
    tiles = {}
    for t, name in task_names:
        p = os.path.join(fft_dir, f"setafix_grad_tiles_{name}_{model}.pt")
        if not os.path.exists(p):
            cands = [f for f in glob.glob(os.path.join(fft_dir, "setafix_grad_tiles_*.pt"))
                     if _norm(os.path.basename(f)[len("setafix_grad_tiles_"):-3]) == _norm(f"{name}_{model}")]
            p = cands[0] if cands else None
        if p:
            tiles[t] = torch.load(p, map_location="cpu", weights_only=True)
    if len(tiles) < 2:
        print("  [skip] fewer than 2 gradient-tile files -> no heatmap")
        return False
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def sub(t):
        t = t.float().to(dev)
        if t.norm() < 1e-12:
            return None
        _, S, Vh = torch.linalg.svd(t, full_matrices=False)
        e = (S ** 2).cumsum(0) / (S ** 2).sum()
        r = max(1, min(int((e < 0.9).sum().item()) + 1, 8))
        return Vh[:r].T

    cache = {t: {} for t in tiles}
    ts = sorted(tiles)
    M = [[float("nan")] * len(ts) for _ in ts]
    for i, a in enumerate(ts):
        for j, b in enumerate(ts):
            if j < i:
                M[i][j] = M[j][i]; continue
            common = set(tiles[a]) & set(tiles[b])
            vals = []
            for k in common:
                Qa = cache[a].setdefault(k, sub(tiles[a][k])); Qb = cache[b].setdefault(k, sub(tiles[b][k]))
                if Qa is None or Qb is None:
                    continue
                vals.append(float(((Qa.T @ Qb) ** 2).sum()) / min(Qa.shape[1], Qb.shape[1]))
            M[i][j] = (sum(vals) / len(vals)) if vals else float("nan")
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    names = dict(task_names)
    ax.set_xticks(range(len(ts))); ax.set_yticks(range(len(ts)))
    ax.set_xticklabels([f"T{t}" for t in ts]); ax.set_yticklabels([f"T{t} {names[t][:10]}" for t in ts])
    for i in range(len(ts)):
        for j in range(len(ts)):
            if M[i][j] == M[i][j]:
                ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center", fontsize=FS - 5,
                        color="white" if M[i][j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="mean direction agreement on common blocks")
    ax.set_title("how similar are tasks on the blocks they both selected?")
    savefig(fig, out, f"fig6_task_similarity_{label}")
    return True


# ───────────────────────── LaTeX ───────────────────────────────────────
def write_latex(trees, out):
    labs = list(trees)
    L = ["% auto-generated by setafix_analyze.py  (\\usepackage{booktabs,multirow})",
         "\\begin{table}[t]\\centering",
         "\\caption{Expert structure produced by the split across the task stream "
         "(structure-only pass: FFT + split, no training). Unique = blocks in the task's own "
         "expert $\\mathrm{exp\\_T}t$; Sel.\\ = all blocks the task received; Experts = experts alive; "
         "Shared = blocks held by shared experts; Med.\\ = median expert size.}",
         "\\label{tab:setafix_structure}\\small",
         "\\begin{tabular}{ll" + "rrrrr" * len(labs) + "}", "\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{lab}}}" for lab in labs) + " \\\\",
         " ".join(f"\\cmidrule(lr){{{3 + 5 * i}-{7 + 5 * i}}}" for i in range(len(labs))),
         "$t$ & task & " + " & ".join("Sel. & Unique & Unique\\% & Experts & Shared" for _ in labs) + " \\\\",
         "\\midrule"]
    n = min(len(r) for r in trees.values())
    for i in range(n):
        cells = []
        for lab in labs:
            r = trees[lab][i]
            uf = f"{100 * r['unique'] / r['selected']:.0f}" if r["selected"] else "--"
            cells.append(f"{r['selected']} & {r['unique']} & {uf} & {r['experts']} & {r['shared_total']}")
        r0 = trees[labs[0]][i]
        L.append(f"{r0['task']} & {r0['name'][:11]} & " + " & ".join(cells) + " \\\\")
    L += ["\\midrule"]
    cells = []
    for lab in labs:
        rows = trees[lab]; last = rows[-1]
        mean_u = sum(r["unique"] for r in rows[1:]) / max(len(rows) - 1, 1)
        cells.append(f"-- & {mean_u:.0f} & -- & {last['experts']} & {last['shared_total']}")
    L.append("\\multicolumn{2}{l}{mean unique ($t\\ge2$) / final} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    # compact summary table
    S = ["", "\\begin{table}[t]\\centering",
         "\\caption{Summary after the last task.}\\label{tab:setafix_summary}\\small",
         "\\begin{tabular}{l" + "r" * len(labs) + "}", "\\toprule",
         "metric & " + " & ".join(labs) + " \\\\", "\\midrule"]
    def row(name, f):
        S.append(f"{name} & " + " & ".join(f(trees[l]) for l in labs) + " \\\\")
    row("mean unique-expert blocks ($t\\ge2$)", lambda r: f"{sum(x['unique'] for x in r[1:]) / max(len(r) - 1, 1):.0f}")
    row("min unique-expert blocks", lambda r: f"{min(x['unique'] for x in r)}")
    row("experts alive (final)", lambda r: f"{r[-1]['experts']}")
    row("shared experts (final)", lambda r: f"{r[-1]['shared_experts']}")
    row("median expert size (final)", lambda r: f"{r[-1]['median_expert']}")
    row("largest expert (final)", lambda r: f"{r[-1]['largest_expert']}")
    row("max experts in one layer", lambda r: f"{r[-1]['max_experts_per_layer']}")
    row("shared blocks (final)", lambda r: f"{r[-1]['shared_total']}")
    row("total blocks held (final)", lambda r: f"{r[-1]['footprint']}")
    S += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(out, "tables.tex"), "w").write("\n".join(L + S) + "\n")
    print(f"LaTeX tables -> {os.path.join(out, 'tables.tex')}")


# ───────────────────────── main ─────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trees", nargs="+")
    ap.add_argument("--fft", nargs="*", default=[], help="FFT dirs (auto-discovered if omitted)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--labels", default=None, help="comma list, one per tree")
    a = ap.parse_args()

    labels = a.labels.split(",") if a.labels else []
    out = a.out or os.path.join(a.trees[0], "setafix_analysis")
    os.makedirs(out, exist_ok=True)

    trees, prov, reports, final_sizes, tdirs, ffts = {}, {}, {}, {}, {}, {}
    for i, tree in enumerate(a.trees):
        tree = tree.rstrip("/")
        lab = labels[i] if i < len(labels) else \
            ("seam" if "_seam_" in tree else "original" if "_off_" in tree else os.path.basename(tree)[-24:])
        fft = find_fft_dir(tree, a.fft)
        print(f"[tree] {lab}: {tree}\n       FFT: {fft or '(not found — budget/provenance columns will be 0)'}")
        rows, pr, rep, fs = scan_tree(tree, fft)
        if not rows:
            print("       no task directories with expert_definitions.json — skipped"); continue
        trees[lab], prov[lab], reports[lab], final_sizes[lab], tdirs[lab], ffts[lab] = rows, pr, rep, fs, task_dirs(tree), fft

    if not trees:
        sys.exit("nothing to analyse")

    # tables
    cols = list(next(iter(trees.values()))[0].keys())
    with open(os.path.join(out, "per_task.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tree"] + cols)
        for lab, rows in trees.items():
            for r in rows:
                w.writerow([lab] + [r[c] for c in cols])
    with open(os.path.join(out, "blocks_provenance.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tree", "task", "name", "novel", "shared_absorbed", "own_copy"])
        for lab, pr in prov.items():
            for row in pr:
                w.writerow([lab] + list(row))

    md = ["# Split-on-Share structure analysis\n"]
    for lab, rows in trees.items():
        md.append(f"\n## {lab}\n")
        md.append("| task | name | selected | unique | unique % | shared blocks | footprint | experts | co-resident | share | reject | no-ev | align |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            uf = f"{100 * r['unique_frac']:.0f}%" if r["unique_frac"] == r["unique_frac"] else "-"
            al = f"{r['align_mean']:.2f}" if r["align_mean"] == r["align_mean"] else "-"
            md.append(f"| {r['task']} | {r['name'][:12]} | {r['selected']} | {r['unique']} | {uf} | {r['shared_total']} | "
                      f"{r['footprint']} | {r['experts']} | {r['coresident']} | {r['share']} | {r['reject']} | {r['no_evidence']} | {al} |")
        last = rows[-1]
        md.append(f"\nAfter task {last['task']}: {last['experts']} experts, largest {last['largest_expert']} blocks, "
                  f"median {last['median_expert']}, smallest {last['smallest_expert']}; "
                  f"max {last['max_experts_per_layer']} experts in a single layer.")
    if len(trees) >= 2:
        labs = list(trees)
        A, B = trees[labs[0]], trees[labs[1]]
        md.append(f"\n## {labs[0]} vs {labs[1]} (same task index)\n")
        md.append("| task | unique Δ | experts Δ | footprint Δ |")
        md.append("|---|---|---|---|")
        for ra, rb in zip(A, B):
            md.append(f"| {ra['task']} | {ra['unique'] - rb['unique']:+d} | {ra['experts'] - rb['experts']:+d} | {ra['footprint'] - rb['footprint']:+d} |")
        md.append("\nPositive unique Δ = the first tree keeps more of each task's own capacity; "
                  "negative experts Δ = fewer expert fragments.")

    # diagnosis lines
    md.append("\n## Diagnosis\n")
    for lab, rows in trees.items():
        ufs = [r["unique_frac"] for r in rows if r["unique_frac"] == r["unique_frac"]]
        if len(ufs) >= 3:
            trend = ufs[-1] - ufs[0]
            md.append(f"- **{lab}**: unique share goes {100 * ufs[0]:.0f}% → {100 * ufs[-1]:.0f}% across the stream "
                      f"({'shrinking' if trend < -0.15 else 'stable' if abs(trend) <= 0.15 else 'growing'}).")
        rej = sum(r["reject"] for r in rows); sh = sum(r["share"] for r in rows)
        if rej + sh:
            md.append(f"- **{lab}**: of {rej + sh} overlapping blocks judged, {100 * sh / (rej + sh):.0f}% were direction-aligned "
                      f"(shared) and {100 * rej / (rej + sh):.0f}% were not (own copy). "
                      + ("Low alignment rate means positional overlap rarely reflects shared meaning." if rej > sh else
                         "Most overlaps are genuinely aligned; sharing was mostly justified."))
        noev = sum(r["no_evidence"] for r in rows)
        if noev:
            md.append(f"- **{lab}**: {noev} overlapping blocks had no direction evidence (missing gradient tiles) and used the default policy — check phase-0 logs for `[SETAFIX] saved N gradient tiles`.")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(md) + "\n")
    write_latex(trees, out)
    print("\n".join(md))

    # figures
    fig_headline(trees, out)
    fig_unique_vs_budget(trees, out)
    fig_experts(trees, out)
    fig_provenance(prov, out)
    fig_alignment(reports, out)
    fig_sizes(final_sizes, out)
    fig_per_layer_reject(reports, out)
    fig_layer_profile(tdirs, out)
    for lab in trees:
        if ffts[lab]:
            names = [(t, n) for t, n, _, _ in tdirs[lab]]
            model = tdirs[lab][0][2] if tdirs[lab] else ""
            # dir names are lower-case; the FFT files use the original task names
            orig = {}
            for f in glob.glob(os.path.join(ffts[lab], "selected_submatrix_attention_*.json")):
                b = os.path.basename(f)[len("selected_submatrix_attention_"):-len(".json")].replace("_with_ranks", "")
                orig[b.lower()] = b
            names = [(t, orig.get(f"{n}_{model}".lower(), f"{n}_{model}")[:-(len(model) + 1)]) for t, n in names]
            fig_task_similarity(ffts[lab], names, model, out, lab)
            break  # tiles are identical across trees (FFT is split-independent)
    print(f"\n-> outputs in {out}")


if __name__ == "__main__":
    main()


# (/lustre/hdd/LAS/jannesar-lab/fatemask/seta_env) bash-5.1$ bash run_analysis.sh --local \
#   ../trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_*_setafix_seam_svd_structonly \
#   ../trained_models/RL_Lamda_fsys_trained_models_SoS_qwen3_4b_v_proj_only/ablation_*_setafix_off_svd_structonly