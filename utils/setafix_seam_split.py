"""
setafix_seam_split.py — direction-aware Split-on-Share ("SEAM-SoS").

Plugs into RL_Lamda_dyn_SETA_SoS_Aug20.SETAModel._perform_split_on_share.
The original split shares a block whenever the new task's support merely
OVERLAPS an old expert's positions. This module adds one question per
overlapping block:

    "Does the new task's gradient act on the same input subspace the old
     expert's update occupies?"  (sign-blind: A and -A score the same;
     SETAFIX_SIGN=on adds a signed agreement check, see validate)

  * yes  -> block becomes shared (weight inheritance, as before)
  * no   -> old expert KEEPS the block (frozen, untouched) AND the new task
            gets its own fresh copy of the block in its unique expert
            (co-residency: the MoE forward sums expert outputs, so both
            deltas live on the same block without interfering in weight
            space). This is what stops later tasks from losing their unique
            capacity to "blind" sharing.

Direction evidence
  new task : gradient tile of the block from this task's FFT warm-up,
             saved by save_grad_tiles() in Phase 0 (no FFT logic changed).
  old expert: its trained delta tile (selected_weight - base). If the delta
             is ~0 (e.g. structure-only runs where experts never trained),
             fall back to the old task's OWN saved gradient tile.

Score  (SETAFIX_SCORE)
  svd     : mean squared principal cosine between the top-r input
            subspaces of the two tiles (SEAM's agreement statistic)   [default]
  fisher  : cosine between |G_new| and |D_old| magnitude patterns
            (diagonal-Fisher style, direction-blind but cheaper)

Threshold — no fixed number. For each (rank_a, rank_b, d) the chance level
of the score for UNRELATED tiles is measured by Monte Carlo once and cached;
a block is "aligned" only if its score clears mean + SETAFIX_Z * std of that
null (default z = 3).

Relative size rule (avoids absolute block counts): a shared child is only
carved out of an old expert if the aligned overlap is at least
SETAFIX_MIN_SHARED_FRAC of that expert's blocks (default 0.15) — otherwise the
whole overlap is treated as rejected (co-resident), which also keeps the
expert count from growing with tiny shared fragments.

Everything is env-gated; SETAFIX_SPLIT=off reproduces the original split.
"""
import os
import re
import json
import math
from collections import defaultdict

import torch

PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")
EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────
# Phase-0 helper: persist gradient tiles of the SELECTED blocks
# ─────────────────────────────────────────────────────────────────────────
def _short_proj(name):
    s = str(name)
    for p in PROJ_NAMES:
        if p in s:
            return p
    return None


def _load_positions_json(save_path_FFT, config_name):
    for suffix in ("", "_with_ranks"):
        p = os.path.join(save_path_FFT,
                         f"selected_submatrix_attention_{config_name}{suffix}.json")
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            out = {}
            for k, v in data.items():
                parts = k.rsplit("_layer_", 1)
                if len(parts) != 2:
                    continue
                try:
                    out[(_short_proj(parts[0]), int(parts[1]))] = \
                        [tuple(int(x) for x in it) for it in v if len(it) == 2]
                except Exception:
                    continue
            return out
    return {}


def grad_tiles_path(save_path_FFT, config_name):
    return os.path.join(save_path_FFT, f"setafix_grad_tiles_{config_name}.pt")


def save_grad_tiles(raw_grads, save_path_FFT, config_name, block_dim):
    """Called in Phase 0 right after block selection. Saves fp16 gradient
    tiles for the selected blocks only:  {"proj|layer|r,c": tensor[bd,bd]}."""
    positions = _load_positions_json(save_path_FFT, config_name)
    if not positions:
        print("[SETAFIX] no selected-position JSON found; grad tiles not saved")
        return None
    bd = int(block_dim)
    tiles, n_layers = {}, 0
    for key, g in raw_grads.items():
        if not isinstance(key, (tuple, list)) or len(key) < 2:
            continue
        proj = _short_proj(key[0])
        try:
            layer = int(key[1])
        except Exception:
            continue
        pos = positions.get((proj, layer))
        if not pos or proj is None:
            continue
        g = g.detach()
        if g.dim() != 2:
            continue
        n_layers += 1
        for (r, c) in pos:
            if (r + 1) * bd <= g.shape[0] and (c + 1) * bd <= g.shape[1]:
                tiles[f"{proj}|{layer}|{r},{c}"] = \
                    g[r * bd:(r + 1) * bd, c * bd:(c + 1) * bd].to(torch.float16).cpu()
    out = grad_tiles_path(save_path_FFT, config_name)
    torch.save(tiles, out)
    print(f"[SETAFIX] saved {len(tiles)} gradient tiles over {n_layers} layers -> {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────
# Alignment scores + chance calibration
# ─────────────────────────────────────────────────────────────────────────
def _subspaces(tile, energy, rmax):
    """(Q_right d_in x r, Q_left d_out x r): top-r input and output directions."""
    t = tile.float()
    n = float(t.norm())
    if not math.isfinite(n) or n < EPS:
        return None, None
    try:
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
    except Exception:
        return None, None
    e = (S ** 2).cumsum(0) / max(float((S ** 2).sum()), EPS)
    r = int((e < energy).sum().item()) + 1
    r = max(1, min(r, rmax, Vh.shape[0]))
    return Vh[:r].T.contiguous(), U[:, :r].contiguous()


def _input_subspace(tile, energy, rmax):
    return _subspaces(tile, energy, rmax)[0]


def _svd_score(Qa, Qb):
    M = Qa.T @ Qb
    return float((M ** 2).sum()) / max(1, min(Qa.shape[1], Qb.shape[1]))


def _fisher_score(A, B):
    a = A.float().abs().reshape(-1)
    b = B.float().abs().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + EPS))


class ChanceFloor:
    """Monte-Carlo null distribution of a score for unrelated tiles."""

    def __init__(self, n_draws=48, seed=0):
        self.n = n_draws
        self.gen = torch.Generator().manual_seed(seed)
        self.cache = {}

    def svd(self, ra, rb, d):
        key = ("svd", ra, rb, d)
        if key not in self.cache:
            vals = []
            for _ in range(self.n):
                Qa, _ = torch.linalg.qr(torch.randn(d, ra, generator=self.gen))
                Qb, _ = torch.linalg.qr(torch.randn(d, rb, generator=self.gen))
                vals.append(_svd_score(Qa, Qb))
            v = torch.tensor(vals)
            self.cache[key] = (float(v.mean()), float(v.std(unbiased=False)))
        return self.cache[key]

    def fisher(self, d):
        key = ("fisher", d)
        if key not in self.cache:
            vals = []
            for _ in range(self.n):
                A = torch.randn(d, d, generator=self.gen)
                B = torch.randn(d, d, generator=self.gen)
                vals.append(_fisher_score(A, B))
            v = torch.tensor(vals)
            self.cache[key] = (float(v.mean()), float(v.std(unbiased=False)))
        return self.cache[key]


# ─────────────────────────────────────────────────────────────────────────
# The validator used inside _perform_split_on_share
# ─────────────────────────────────────────────────────────────────────────
class SeamSplitValidator:
    def __init__(self, seta_model, new_task_id):
        env = os.environ.get
        self.mode = env("SETAFIX_SPLIT", "seam").lower()          # seam | off
        self.score = env("SETAFIX_SCORE", "svd").lower()          # svd | fisher
        self.z = float(env("SETAFIX_Z", "3.0"))
        self.min_align = float(env("SETAFIX_MIN_ALIGN", "0.0"))   # optional hard floor
        self.energy = float(env("SETAFIX_ENERGY", "0.9"))
        self.rmax = int(env("SETAFIX_RMAX", "8"))
        self.min_shared_frac = float(env("SETAFIX_MIN_SHARED_FRAC", "0.15"))
        self.no_evidence = env("SETAFIX_NOEVIDENCE", "share").lower()  # share | reject
        self.side = env("SETAFIX_SIDE", "right").lower()          # right | left | both
        self.null = env("SETAFIX_NULL", "gauss").lower()          # gauss | empirical
        self.null_k = int(env("SETAFIX_NULL_K", "32"))            # other-block samples per test
        # density controls — these bind regardless of what the direction test
        # scores, which is what makes SEAM work at large block budgets where
        # almost every overlap clears any chance bar.
        self.share_q = float(env("SETAFIX_SHARE_Q", "1.0"))        # keep top-q of each overlap group
        # [PHI] one knob: min share of the NEW task's selection a shared child
        # must carry. Defaults SETAFIX_MIN_SHARED_FRAC_NEW here and
        # SETAFIX_LAYER_FRAC in _perform_split_on_share.
        self.phi = float(env("SETAFIX_PHI", "0.025"))
        self.min_frac_new = float(env("SETAFIX_MIN_SHARED_FRAC_NEW", str(self.phi)))
        self.max_shared = int(env("SETAFIX_MAX_SHARED_PER_TASK", "0"))  # 0 = unlimited
        # [SIGN] the subspace score cannot tell A from -A. SETAFIX_SIGN=on also
        # requires signed agreement c > SETAFIX_SIGN_TAU before a block is shared,
        # c = <D_old, -G_new>/(|D_old||G_new|)  (old delta vs new descent direction),
        # c = <G_old,  G_new>/(|G_old||G_new|)  (when the old tile is a grad fallback).
        # Default off; c is logged per block either way ("cos" in the report).
        self.sign_on = env("SETAFIX_SIGN", "off").lower() == "on"
        self.sign_tau = float(env("SETAFIX_SIGN_TAU", "0.0"))
        self.n_shared_made = 0
        self._n_new_selected = None
        self.ortho_bases = {}                                     # saved for the training-time projection
        self._sub_cache = {}
        self.fft_dir = env("SETAFIX_FFT_DIR", "")
        self.model_name = env("SETAFIX_MODEL_NAME", "")
        self.task_list = [t for t in env("SETAFIX_TASK_LIST", "").split(",") if t]
        self.new_cfg = env("SETAFIX_CONFIG_NAME", "")
        self.report_dir = env("SETAFIX_REPORT_DIR", self.fft_dir or ".")
        self.task_id = int(new_task_id)
        self.bd = int(seta_model.block_dimension) if hasattr(seta_model, "block_dimension") \
            else int(getattr(seta_model, "block_dim", 128))
        self.floor = ChanceFloor()
        self._grad_cache = {}
        self.records = []
        self.counts = defaultdict(int)

        # layer_key -> SequentialMoELayer of the PREVIOUS model
        self.layers = {}
        for name, module in seta_model.model.named_modules():
            if hasattr(module, "experts") and hasattr(module, "base_weight_ref"):
                lk = seta_model._get_layer_key_from_name(name)
                if lk is not None:
                    self.layers[lk] = module
        self.new_tiles = self._load_grad_tiles(self.new_cfg)
        self.tiles_signed = None
        if self.new_tiles:
            _smp = list(self.new_tiles.values())[:64]
            self.tiles_signed = any(float(t.float().min()) < 0 for t in _smp)
            if not self.tiles_signed:
                print("[SETAFIX] WARNING: saved gradient tiles are all non-negative "
                      "(|grad| or grad^2 accumulation?) — no sign to test; "
                      "SETAFIX_SIGN is ignored and 'cos' is not meaningful")
                self.sign_on = False
        print(f"[SETAFIX] SEAM-SoS validator: mode={self.mode} score={self.score} "
              f"z={self.z} energy={self.energy} rmax={self.rmax} "
              f"phi={self.phi} min_shared_frac={self.min_shared_frac}/{self.min_frac_new} share_q={self.share_q} "
              f"max_shared={self.max_shared or 'inf'} side={self.side} null={self.null} "
              f"sign={'on' if self.sign_on else 'off'}(tau={self.sign_tau}) | new-task grad tiles: "
              f"{len(self.new_tiles)} | layers mapped: {len(self.layers)}")

    # ---- evidence -------------------------------------------------------
    def _load_grad_tiles(self, cfg):
        if not cfg or not self.fft_dir:
            return {}
        if cfg in self._grad_cache:
            return self._grad_cache[cfg]
        p = grad_tiles_path(self.fft_dir, cfg)
        tiles = torch.load(p, map_location="cpu") if os.path.exists(p) else {}
        if not tiles:
            print(f"[SETAFIX] no gradient tiles for '{cfg}' at {p}")
        self._grad_cache[cfg] = tiles
        return tiles

    def _cfg_for_task(self, tid):
        if 1 <= tid <= len(self.task_list) and self.model_name:
            return f"{self.task_list[tid - 1]}_{self.model_name}"
        return ""

    def _old_tile(self, old_key, old_defn, layer_key, pos):
        """Old expert's delta on block `pos`; falls back to the old task's
        saved gradient tile when the delta is ~0. Returns (tile, source)."""
        module = self.layers.get(layer_key)
        bd = self.bd
        if module is not None and old_key in module.experts:
            exp = module.experts[old_key]
            try:
                i = [tuple(p) for p in exp.index_list].index(tuple(pos))
                r, c = int(pos[0]), int(pos[1])
                W = module.base_weight_ref
                delta = exp.selected_weight[i * bd:(i + 1) * bd, :].detach().float().cpu() \
                    - W[r * bd:(r + 1) * bd, c * bd:(c + 1) * bd].detach().float().cpu()
                base_n = float(W[r * bd:(r + 1) * bd, c * bd:(c + 1) * bd].float().norm())
                if float(delta.norm()) > 1e-6 * max(base_n, 1.0):
                    return delta, "delta"
            except (ValueError, IndexError):
                pass
        # fallback: gradient tile of the task that last owned this expert
        tid = old_defn.get("task_id", None)
        if isinstance(tid, int):
            tiles = self._load_grad_tiles(self._cfg_for_task(tid))
            k = f"{layer_key[0]}|{layer_key[1]}|{pos[0]},{pos[1]}"
            if k in tiles:
                return tiles[k], "grad"
        return None, "none"

    def _new_tile(self, layer_key, pos):
        k = f"{layer_key[0]}|{layer_key[1]}|{pos[0]},{pos[1]}"
        return self.new_tiles.get(k)

    # ---- per-block decision ---------------------------------------------
    def _sub(self, key, tile):
        if key not in self._sub_cache:
            self._sub_cache[key] = _subspaces(tile, self.energy, self.rmax)
        return self._sub_cache[key]

    def _pair_score(self, sa, sb):
        """Agreement between two (Q_right, Q_left) pairs according to SETAFIX_SIDE."""
        vals = []
        if self.side in ("right", "both"):
            vals.append(_svd_score(sa[0], sb[0]))
        if self.side in ("left", "both"):
            vals.append(_svd_score(sa[1], sb[1]))
        return sum(vals) / len(vals)

    def _empirical_floor(self, old_sub, layer_key):
        """Null = this old tile scored against the NEW task's tiles on OTHER blocks
        of the same layer. Backbone-wide activation structure raises every pair
        alike, so the bar rises with it; only task-specific agreement clears it."""
        prefix = f"{layer_key[0]}|{layer_key[1]}|"
        keys = [k for k in self.new_tiles if k.startswith(prefix)]
        if len(keys) < 4:
            return None
        step = max(1, len(keys) // self.null_k)
        vals = []
        for k in keys[::step][:self.null_k]:
            sb = self._sub(("new", k), self.new_tiles[k])
            if sb[0] is not None:
                vals.append(self._pair_score(old_sub, sb))
        if len(vals) < 4:
            return None
        v = torch.tensor(vals)
        return float(v.mean() + self.z * v.std(unbiased=False))

    def _signed_cos(self, old_tile, new_tile, src):
        """Signed agreement of the two tiles (see SETAFIX_SIGN in __init__)."""
        a = old_tile.float().reshape(-1)
        b = new_tile.float().reshape(-1)
        c = float((a @ b) / (a.norm() * b.norm() + EPS))
        return -c if src == "delta" else c

    def _aligned(self, old_tile, new_tile, layer_key=None, pos=None):
        d = old_tile.shape[1]
        if self.score == "fisher":
            s = _fisher_score(old_tile, new_tile)
            mu, sd = self.floor.fisher(d)
            return s >= max(mu + self.z * sd, self.min_align), s, mu + self.z * sd, None
        sa = self._sub(("old", layer_key, pos), old_tile)
        sb = self._sub(("new", f"{layer_key[0]}|{layer_key[1]}|{pos[0]},{pos[1]}"), new_tile)
        if sa[0] is None or sb[0] is None:
            return None, float("nan"), float("nan"), None
        s = self._pair_score(sa, sb)
        thr = None
        if self.null == "empirical":
            thr = self._empirical_floor(sa, layer_key)
        if thr is None:                                   # gauss, or empirical fallback
            mu, sd = self.floor.svd(sa[0].shape[1], sb[0].shape[1], d)
            thr = mu + self.z * sd
        thr = max(thr, self.min_align)
        return s >= thr, s, thr, sa

    def n_new_selected(self):
        """Total blocks the new task selected (from its FFT selection JSON).
        Used by the relative-size rule; 0 if the file is unavailable."""
        if self._n_new_selected is None:
            pos = _load_positions_json(self.fft_dir, self.new_cfg) if self.fft_dir else {}
            self._n_new_selected = sum(len(v) for v in pos.values())
        return self._n_new_selected

    def validate(self, old_key, old_defn, layer_key, intersection):
        """Split an overlap into (aligned, rejected) block lists.

        Two gates, in order:
          1. chance bar      — a block below it is never shared (quality floor)
          2. quantile cap    — of the blocks above the bar, only the best
                               SETAFIX_SHARE_Q fraction is shared; the rest go
                               the own-copy route (density control)
        The bar alone is not enough: at large block budgets nearly every
        overlapping block clears it, so without (2) the split degenerates to
        the positional rule."""
        if self.mode == "off" or not intersection:
            return list(intersection), []

        scored, no_ev, degenerate = [], [], []
        cos_of = {}
        for pos in intersection:
            new_t = self._new_tile(layer_key, pos)
            old_t, src = self._old_tile(old_key, old_defn, layer_key, pos)
            if new_t is None or old_t is None:
                no_ev.append((pos, src))
                continue
            ok, sc, thr, sa = self._aligned(old_t, new_t, layer_key, pos)
            if ok is None:
                degenerate.append(pos)
                continue
            cos_of[tuple(pos)] = self._signed_cos(old_t, new_t, src)
            scored.append((pos, sc, thr, ok, sa))

        aligned, rejected = list(degenerate), []
        # 1. chance bar
        passed = [x for x in scored if x[3]]
        failed = [x for x in scored if not x[3]]
        # 1b. sign gate (SETAFIX_SIGN=on): same subspace but opposing update -> own copy
        sign_rej = []
        if self.sign_on:
            sign_rej = [x for x in passed if cos_of[tuple(x[0])] <= self.sign_tau]
            passed = [x for x in passed if cos_of[tuple(x[0])] > self.sign_tau]
        # 2. quantile cap over the blocks that passed the bar
        n_keep = len(passed) if self.share_q >= 1.0 else \
            int(math.floor(self.share_q * len(passed) + 1e-9))
        passed.sort(key=lambda x: x[1], reverse=True)
        keep, capped = passed[:n_keep], passed[n_keep:]

        for pos, sc, thr, _, sa in keep:
            aligned.append(pos)
            self.counts["share"] += 1
            self.records.append({"expert": old_key, "layer": f"{layer_key[0]}|{layer_key[1]}",
                                 "block": list(pos), "score": round(sc, 4),
                                 "thr": None if thr != thr else round(thr, 4),
                                 "cos": round(cos_of[tuple(pos)], 4),
                                 "source": "tile", "decision": "share"})
        for pos, sc, thr, _, sa in capped:
            rejected.append(pos)
            self.counts["reject_quantile"] += 1
            if sa is not None:
                self.ortho_bases[f"{layer_key[0]}|{layer_key[1]}|{pos[0]},{pos[1]}"] = \
                    {"Q": sa[0].cpu(), "U": sa[1].cpu(), "old_expert": old_key}
            self.records.append({"expert": old_key, "layer": f"{layer_key[0]}|{layer_key[1]}",
                                 "block": list(pos), "score": round(sc, 4),
                                 "thr": None if thr != thr else round(thr, 4),
                                 "cos": round(cos_of[tuple(pos)], 4),
                                 "source": "tile", "decision": "reject(quantile)"})
        for pos, sc, thr, _, sa in failed:
            rejected.append(pos)
            self.counts["reject"] += 1
            if sa is not None:
                self.ortho_bases[f"{layer_key[0]}|{layer_key[1]}|{pos[0]},{pos[1]}"] = \
                    {"Q": sa[0].cpu(), "U": sa[1].cpu(), "old_expert": old_key}
            self.records.append({"expert": old_key, "layer": f"{layer_key[0]}|{layer_key[1]}",
                                 "block": list(pos), "score": round(sc, 4),
                                 "thr": None if thr != thr else round(thr, 4),
                                 "cos": round(cos_of[tuple(pos)], 4),
                                 "source": "tile", "decision": "reject"})
        for pos, sc, thr, _, sa in sign_rej:
            rejected.append(pos)
            self.counts["reject_sign"] += 1
            if sa is not None:
                self.ortho_bases[f"{layer_key[0]}|{layer_key[1]}|{pos[0]},{pos[1]}"] = \
                    {"Q": sa[0].cpu(), "U": sa[1].cpu(), "old_expert": old_key}
            self.records.append({"expert": old_key, "layer": f"{layer_key[0]}|{layer_key[1]}",
                                 "block": list(pos), "score": round(sc, 4),
                                 "thr": None if thr != thr else round(thr, 4),
                                 "cos": round(cos_of[tuple(pos)], 4),
                                 "source": "tile", "decision": "reject(sign)"})
        for pos, src in no_ev:
            dec = "share" if self.no_evidence == "share" else "reject"
            (aligned if dec == "share" else rejected).append(pos)
            self.counts["no_evidence"] += 1
            self.records.append({"expert": old_key, "layer": f"{layer_key[0]}|{layer_key[1]}",
                                 "block": list(pos), "score": None, "thr": None,
                                 "source": src, "decision": dec + "(no-evidence)"})
        if degenerate:
            self.counts["share_degenerate"] += len(degenerate)
        return aligned, rejected

    # ---- per-expert decision --------------------------------------------
    def keep_split(self, old_key, old_defn, n_aligned):
        """Whether this old expert earns a shared child. Three relative rules,
        none of them an absolute block count:
          a) aligned overlap >= SETAFIX_MIN_SHARED_FRAC of the OLD expert
          b) aligned overlap >= SETAFIX_MIN_SHARED_FRAC_NEW of the NEW task's
             selection  (at large budgets old experts are huge, so (a) alone
             stops binding and every signature spawns a child)
          c) at most SETAFIX_MAX_SHARED_PER_TASK shared children per task
        A refused split is not lost capacity: the overlap stays co-resident
        (old expert keeps it frozen, new task keeps its own copy)."""
        if self.mode == "off":
            return True
        n_old = sum(len(v) for g in ("mlp", "attn") for v in old_defn.get(g, {}).values())
        n_new = self.n_new_selected()
        why = None
        if n_aligned < self.min_shared_frac * max(n_old, 1):
            why = (f"{n_aligned} < {self.min_shared_frac:.2f} x {n_old} (old expert)")
            self.counts["skip_small_vs_old"] += 1
        elif self.min_frac_new > 0 and n_new and n_aligned < self.min_frac_new * n_new:
            why = (f"{n_aligned} < {self.min_frac_new:.3f} x {n_new} (new task's selection)")
            self.counts["skip_small_vs_new"] += 1
        elif self.max_shared and self.n_shared_made >= self.max_shared:
            why = f"already made {self.n_shared_made} shared children this task (cap {self.max_shared})"
            self.counts["skip_cap"] += 1
        if why is not None:
            self.counts["expert_split_skipped"] += 1
            print(f"    - [SETAFIX] '{old_key}': no shared child — {why} "
                  f"(overlap stays co-resident)")
            return False
        self.n_shared_made += 1
        return True

    # ---- report ------------------------------------------------------------
    def write_report(self, expert_definitions):
        sizes = {k: sum(len(v) for g in ("mlp", "attn") for v in d.get(g, {}).values())
                 for k, d in expert_definitions.items() if isinstance(d, dict)}
        rep = {"task_id": self.task_id, "mode": self.mode, "score": self.score,
               "z": self.z, "min_shared_frac": self.min_shared_frac,
               "min_shared_frac_new": self.min_frac_new, "share_q": self.share_q,
               "max_shared_per_task": self.max_shared, "side": self.side, "null": self.null,
               "sign": self.sign_on, "sign_tau": self.sign_tau, "tiles_signed": self.tiles_signed,
               "n_shared_children": self.n_shared_made,
               "n_new_selected": self.n_new_selected(),
               "counts": dict(self.counts), "expert_sizes_after_split": sizes,
               "n_experts_after_split": len(sizes), "blocks": self.records}
        os.makedirs(self.report_dir, exist_ok=True)
        if self.ortho_bases:
            bp = os.path.join(self.report_dir, f"setafix_ortho_bases_T{self.task_id}.pt")
            torch.save(self.ortho_bases, bp)
            print(f"[SETAFIX] old-expert bases for {len(self.ortho_bases)} own-copy blocks -> {bp}")
        p = os.path.join(self.report_dir, f"setafix_split_report_T{self.task_id}.json")
        with open(p, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"[SETAFIX] split report -> {p} | counts={dict(self.counts)} | "
              f"experts now={len(sizes)}")
        return p


# ─────────────────────────────────────────────────────────────────────────
# Training-time protections (Phase 2). Both default OFF.
# ─────────────────────────────────────────────────────────────────────────
def attach_ortho_hooks(seta_model, task_id, bases_dir):
    """SETAFIX_ORTHO=on: every own-copy block of exp_T{task_id} trains with its
    gradient projected off the old expert's directions on that block
    (g <- g (I - Q Q^T), and (I - U U^T) g when SETAFIX_SIDE includes 'left').
    Trainable-parameter count is unchanged; the forward pass is untouched."""
    if os.environ.get("SETAFIX_ORTHO", "off").lower() != "on":
        return 0
    bp = os.path.join(bases_dir, f"setafix_ortho_bases_T{task_id}.pt")
    if not os.path.exists(bp):
        print(f"[SETAFIX][ortho] no bases file at {bp} (no own-copy blocks?) -> nothing to project")
        return 0
    bases = torch.load(bp, map_location="cpu")
    side = os.environ.get("SETAFIX_SIDE", "right").lower()
    new_key = f"exp_T{task_id}"
    bd = int(getattr(seta_model, "block_dim", 128))
    n_blocks, handles = 0, []
    for name, module in seta_model.model.named_modules():
        if not (hasattr(module, "experts") and hasattr(module, "base_weight_ref")):
            continue
        if new_key not in module.experts:
            continue
        lk = seta_model._get_layer_key_from_name(name)
        if lk is None:
            continue
        exp = module.experts[new_key]
        w = exp.selected_weight
        if not w.requires_grad:
            continue
        proj = []
        for i, pos in enumerate([tuple(p) for p in exp.index_list]):
            b = bases.get(f"{lk[0]}|{lk[1]}|{pos[0]},{pos[1]}")
            if b is None:
                continue
            PR = None if side == "left" else torch.eye(bd) - b["Q"] @ b["Q"].T
            PL = None if side == "right" else torch.eye(bd) - b["U"] @ b["U"].T
            proj.append((i, PR, PL))
        if not proj:
            continue
        n_blocks += len(proj)

        def _mk(proj_list, bd=bd):
            moved = {}
            def _hook(grad):
                g = grad.clone()
                for i, PR, PL in proj_list:
                    key = (i, g.device, g.dtype)
                    if key not in moved:
                        moved[key] = (None if PR is None else PR.to(g.device, g.dtype),
                                      None if PL is None else PL.to(g.device, g.dtype))
                    pr, pl = moved[key]
                    blk = g[i * bd:(i + 1) * bd, :]
                    if pr is not None:
                        blk = blk @ pr
                    if pl is not None:
                        blk = pl @ blk
                    g[i * bd:(i + 1) * bd, :] = blk
                return g
            return _hook
        handles.append(w.register_hook(_mk(proj)))
    print(f"[SETAFIX][ortho] projecting {n_blocks} own-copy blocks of {new_key} "
          f"off their old expert's directions (side={side})")
    return handles


def freeze_prior_gate_rows(seta_model, task_id):
    """SETAFIX_GATE_PROTECT=freeze: gate rows of experts created before task_id
    receive zero gradient (routing-drift protection). Sigmoid gates make rows
    independent, so current-era rows train exactly as before."""
    if os.environ.get("SETAFIX_GATE_PROTECT", "off").lower() != "freeze":
        return 0
    n_rows, handles = 0, []
    for name, module in seta_model.model.named_modules():
        if not (hasattr(module, "experts") and hasattr(module, "gating_network")):
            continue
        keys = list(module.experts.keys())
        gw = module.gating_network.weight
        if gw.shape[0] != len(keys):
            print(f"[SETAFIX][gate][WARN] {name}: {gw.shape[0]} rows != {len(keys)} experts — skipped")
            continue
        old = torch.tensor([max(_task_ids_of(k)) < int(task_id) for k in keys])
        if not bool(old.any()):
            continue
        def _mk(mask):
            def _hook(grad):
                g = grad.clone(); g[mask.to(g.device)] = 0.0; return g
            return _hook
        if gw.requires_grad:
            handles.append(gw.register_hook(_mk(old)))
        gb = getattr(module.gating_network, "bias", None)
        if gb is not None and gb.requires_grad:
            handles.append(gb.register_hook(_mk(old)))
        n_rows += int(old.sum())
    print(f"[SETAFIX][gate] froze {n_rows} prior-era gate rows (task {task_id})")
    return handles


def _task_ids_of(key):
    s = str(key)
    return [int(x) for x in re.findall(r"\d+", s)] if "-" in s else [int(c) for c in re.findall(r"\d", s)] or [0]