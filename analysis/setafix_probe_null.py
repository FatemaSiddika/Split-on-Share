#!/usr/bin/env python3
"""
setafix_probe_null.py — is SEAM's agreement score measuring task similarity,
or just the frozen backbone's input structure?

For each pair of tasks (A = earlier, C = later) it computes two distributions
from the gradient tiles saved in phase 0:

  MATCHED  score( A's tile on block B , C's tile on the SAME block B )
           — exactly what the split test uses today
  NULL     score( A's tile on block B , C's tile on OTHER blocks of the
           same layer )
           — pairs that have no reason to agree except through the shared
             backbone; this is what SETAFIX_NULL=empirical would use as the bar

Reading the result
  matched ≈ null            the score carries no task information; the Gaussian
                            bar is far too low and the empirical null will
                            raise it a lot (expect many more rejects)
  matched >> null           the signal is genuine; the current bar is
                            defensible and dense sharing at large budgets is an
                            overlap-density problem, not a threshold problem

Usage
  python setafix_probe_null.py FFT_DIR [FFT_DIR2 ...] \
         [--labels "12%","25%"] [--layers 6] [--blocks 40] [--null-k 32]
         [--side right|left|both] [--out DIR]

FFT_DIR is the FFT_steps_* directory of a run (it holds
setafix_grad_tiles_<task>_<model>.pt). Needs torch; CPU is fine, GPU is used
if available.
"""
import os, re, glob, json, argparse, random

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 19
plt.rcParams.update({"font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
                     "xtick.labelsize": FS - 3, "ytick.labelsize": FS - 3,
                     "legend.fontsize": FS - 4, "axes.grid": False})
TEAL, GRAY, RED = "#2A7F62", "#8A8880", "#C0392B"


def subspaces(tile, energy=0.9, rmax=8, dev="cpu"):
    t = tile.float().to(dev)
    if not torch.isfinite(t).all() or float(t.norm()) < 1e-12:
        return None, None
    try:
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
    except Exception:
        return None, None
    e = (S ** 2).cumsum(0) / max(float((S ** 2).sum()), 1e-12)
    r = max(1, min(int((e < energy).sum().item()) + 1, rmax, Vh.shape[0]))
    return Vh[:r].T.contiguous(), U[:, :r].contiguous()


def agree(sa, sb, side):
    vals = []
    if side in ("right", "both"):
        M = sa[0].T @ sb[0]
        vals.append(float((M ** 2).sum()) / min(sa[0].shape[1], sb[0].shape[1]))
    if side in ("left", "both"):
        M = sa[1].T @ sb[1]
        vals.append(float((M ** 2).sum()) / min(sa[1].shape[1], sb[1].shape[1]))
    return sum(vals) / len(vals)


def load_tiles(fft_dir):
    """task_name -> {key: tile}. Handles both storage modes (per-block tiles
    'proj|layer|r,c' and full projections 'proj|layer')."""
    out = {}
    for f in sorted(glob.glob(os.path.join(fft_dir, "setafix_grad_tiles_*.pt"))):
        name = os.path.basename(f)[len("setafix_grad_tiles_"):-3]
        try:
            out[name] = torch.load(f, map_location="cpu", weights_only=True)
        except TypeError:
            out[name] = torch.load(f, map_location="cpu")
    return out


def block_keys(tiles, bd):
    """(layer_key -> list of (r,c)) available in this tile dict."""
    per = {}
    for k in tiles:
        parts = k.split("|")
        if len(parts) == 3:
            per.setdefault(f"{parts[0]}|{parts[1]}", []).append(
                tuple(int(x) for x in parts[2].split(",")))
        elif len(parts) == 2:
            t = tiles[k]
            per.setdefault(k, []).extend(
                [(r, c) for r in range(t.shape[0] // bd) for c in range(t.shape[1] // bd)])
    return per


def get_tile(tiles, lk, pos, bd):
    k = f"{lk}|{pos[0]},{pos[1]}"
    if k in tiles:
        return tiles[k]
    full = tiles.get(lk)
    if full is None:
        return None
    r, c = pos
    if (r + 1) * bd > full.shape[0] or (c + 1) * bd > full.shape[1]:
        return None
    return full[r * bd:(r + 1) * bd, c * bd:(c + 1) * bd]


def probe(fft_dir, label, args, dev):
    tiles = load_tiles(fft_dir)
    if len(tiles) < 2:
        print(f"[{label}] fewer than 2 gradient-tile files in {fft_dir} — skipped")
        return None
    names = sorted(tiles)
    bd = args.block_dim
    rng = random.Random(0)
    matched, null = [], []
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    rng.shuffle(pairs)
    per_pair = {}
    for a, b in pairs[:args.pairs]:
        ka, kb = block_keys(tiles[a], bd), block_keys(tiles[b], bd)
        layers = [l for l in ka if l in kb]
        rng.shuffle(layers)
        m_here, n_here = [], []
        for lk in layers[:args.layers]:
            common = [p for p in ka[lk] if p in set(kb[lk])]
            others = [p for p in kb[lk] if p not in set(ka[lk])] or kb[lk]
            rng.shuffle(common); rng.shuffle(others)
            for pos in common[:args.blocks]:
                ta = get_tile(tiles[a], lk, pos, bd)
                tb = get_tile(tiles[b], lk, pos, bd)
                if ta is None or tb is None:
                    continue
                sa, sb = subspaces(ta, dev=dev), subspaces(tb, dev=dev)
                if sa[0] is None or sb[0] is None:
                    continue
                m_here.append(agree(sa, sb, args.side))
                for opos in others[:args.null_k]:
                    to = get_tile(tiles[b], lk, opos, bd)
                    if to is None:
                        continue
                    so = subspaces(to, dev=dev)
                    if so[0] is not None:
                        n_here.append(agree(sa, so, args.side))
        if m_here:
            per_pair[f"{a.split('_')[0]} vs {b.split('_')[0]}"] = (
                sum(m_here) / len(m_here), sum(n_here) / len(n_here) if n_here else float("nan"))
        matched += m_here; null += n_here
    if not matched:
        print(f"[{label}] no comparable blocks found — skipped")
        return None
    mm = sum(matched) / len(matched)
    nn = sum(null) / len(null) if null else float("nan")
    sd = (sum((x - nn) ** 2 for x in null) / len(null)) ** 0.5 if null else float("nan")
    print(f"\n[{label}]  matched n={len(matched)}  mean={mm:.3f}   |   "
          f"null n={len(null)}  mean={nn:.3f}  sd={sd:.3f}")
    print(f"[{label}]  gaussian bar (mean+3sd of RANDOM tiles) is typically ~0.06-0.11")
    print(f"[{label}]  empirical bar (mean+3sd of the NULL above) = {nn + 3 * sd:.3f}")
    frac_above = sum(1 for x in matched if x >= nn + 3 * sd) / len(matched)
    print(f"[{label}]  matched blocks that would still pass the empirical bar: "
          f"{100 * frac_above:.0f}%   (today ~100% pass the gaussian bar at 25%)")
    print(f"[{label}]  per task-pair (matched / null):")
    for k, (m, n) in sorted(per_pair.items(), key=lambda kv: -kv[1][0])[:12]:
        print(f"[{label}]     {k:<28s} {m:.3f} / {n:.3f}")
    return dict(label=label, matched=matched, null=null, mean=mm, null_mean=nn,
                null_sd=sd, bar=nn + 3 * sd, frac_above=frac_above, per_pair=per_pair)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fft_dirs", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--layers", type=int, default=6, help="layers sampled per task pair")
    ap.add_argument("--blocks", type=int, default=40, help="matched blocks per layer")
    ap.add_argument("--null-k", type=int, default=16, help="other-blocks per matched block")
    ap.add_argument("--pairs", type=int, default=10, help="task pairs sampled")
    ap.add_argument("--side", default="right", choices=["right", "left", "both"])
    ap.add_argument("--block-dim", type=int, default=128)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    labels = args.labels.split(",") if args.labels else \
        [os.path.basename(d.rstrip("/"))[-18:] for d in args.fft_dirs]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  side={args.side}  layers={args.layers} blocks={args.blocks} "
          f"null_k={args.null_k} pairs={args.pairs}")

    res = [r for r in (probe(d.rstrip("/"), l, args, dev)
                       for d, l in zip(args.fft_dirs, labels)) if r]
    if not res:
        raise SystemExit("nothing to plot")
    os.makedirs(args.out, exist_ok=True)

    fig, axes = plt.subplots(1, len(res), figsize=(8.2 * len(res), 5.6), squeeze=False)
    bins = [i / 40 for i in range(41)]
    for ax, r in zip(axes[0], res):
        ax.hist(r["null"], bins=bins, color=GRAY, alpha=0.85, density=True,
                label=f"other blocks (null)\nmean {r['null_mean']:.2f}")
        ax.hist(r["matched"], bins=bins, color=TEAL, alpha=0.8, density=True,
                label=f"same block (matched)\nmean {r['mean']:.2f}")
        ax.axvline(r["bar"], color=RED, lw=2.4, ls="--",
                   label=f"empirical bar {r['bar']:.2f}")
        ax.set_title(f"{r['label']}  —  {100 * r['frac_above']:.0f}% of matched pass")
        ax.set_xlabel("subspace agreement"); ax.legend(frameon=False)
    axes[0][0].set_ylabel("density")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"probe_matched_vs_null.{ext}"),
                    dpi=220, bbox_inches="tight")
    json.dump([{k: v for k, v in r.items() if k not in ("matched", "null")} for r in res],
              open(os.path.join(args.out, "probe_summary.json"), "w"), indent=2)
    print(f"\n-> {os.path.join(args.out, 'probe_matched_vs_null.png')} + probe_summary.json")
    print("\nREAD: matched ≈ null  -> the score is backbone structure; SETAFIX_NULL=empirical "
          "will reject far more, and the 86% share rate is not trustworthy.\n"
          "      matched >> null -> the signal is real; dense sharing at 25% is an "
          "overlap-density problem, not a threshold problem.")


if __name__ == "__main__":
    main()
