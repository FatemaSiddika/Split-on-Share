# Split-on-Share (formerly SETA_fix) — direction-aware Split-on-Share (SEAM-SoS) for centralized SETA

Self-contained copy of the centralized SETA pipeline living at
`fed-con/Split-on-Share/` (see README.md for the folder layout). Nothing in `fed-con/` or `SETAV2/` is modified; data and
base models are still read from their SETAV2 / models locations.

## What changed (and only this)
| file | change |
|---|---|
| `setafix_seam_split.py` | NEW. Gradient-tile saver (phase 0), `SeamSplitValidator` (direction test, chance-calibrated threshold, relative-size rule, report). |
| `SETA_SoS` | `_perform_split_on_share` only: 5 small `[SETAFIX]` insertions — validator built at the top, `validate()` on the MLP and attention intersections, `keep_split()` at the final decision, report at the end. `SETAFIX_SPLIT=off` reproduces the original byte-for-byte behaviour. |
| `main.py` | Phase 0: one call saving the selected blocks' gradient tiles (after selection, FFT untouched). Phase 1: a task-1 branch for structure-only mode, plus five `os.environ.setdefault` lines telling the validator where its evidence is. |
| `scripts/run_setafix.sh` / `scripts/setafix_pipeline.slurm` | Copies of the fused pipeline with paths pointed at `Split-on-Share/` (code, checkpoints, logs) and the `STRUCTURE_ONLY` chain. |
| `analysis/setafix_report.py` | NEW. Per-task structure summary / tree comparison. |

## The idea in one paragraph
The original split shares a block whenever the new task's support merely
overlaps an old expert. SEAM-SoS asks, per overlapping block, whether the new
task's FFT gradient tile and the old expert's delta tile move the block in the
same input directions (top-r subspace agreement; `svd`), or — cheaper — have the
same |grad| pattern (`fisher`). The score is compared with a Monte-Carlo chance
level for unrelated tiles of the same ranks (no fixed threshold; `SETAFIX_Z`
z-score). Aligned → shared as before. Not aligned → the old expert keeps the
block frozen AND the new task keeps its own fresh copy in `exp_T<t>` (the MoE
forward sums experts, so both live on the block). A shared child is only carved
if the aligned overlap is a meaningful fraction of the old expert
(`SETAFIX_MIN_SHARED_FRAC`, relative, not a block count) — this is what keeps
the expert count from growing with tiny fragments.

## Fast structure pass (no training, ~hours for 8 tasks)
```bash
cd /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/Split-on-Share
STRUCTURE_ONLY=1 bash scripts/run_setafix.sh 1-8 fft,split qwen3_4b            # SEAM split
STRUCTURE_ONLY=1 SETAFIX_SPLIT=off bash scripts/run_setafix.sh 1-8 fft,split qwen3_4b   # original, for comparison
```
Each task runs FFT → split and saves `init_structure_checkpoint`; the next task
resumes from it. Trees are tagged `..._setafix_<split>_<score>_structonly`.
Then:
```bash
python analysis/setafix_report.py <BASE_DIR_seam> <BASE_DIR_off> --plot
```
Columns: experts alive, unique-expert blocks, shared blocks, co-resident
blocks, share/reject decisions, mean alignment. Per-block decisions are in
`setafix_split_report_T<t>.json` inside each task directory.

In structure-only runs experts are untrained (delta = 0), so the validator
compares gradient tile vs gradient tile (old task's FFT vs new task's FFT) —
still a real direction signal. In full runs it uses the trained delta.

## Full run (training) once the structure looks right
```bash
bash scripts/run_setafix.sh 1-8 fft,split,train,eval qwen3_4b
```
Knobs: `SETAFIX_SPLIT=seam|off`, `SETAFIX_SCORE=svd|fisher`, `SETAFIX_Z`,
`SETAFIX_MIN_SHARED_FRAC`, `SETAFIX_NOEVIDENCE=share|reject`,
`SETAFIX_ENERGY`, `SETAFIX_RMAX`.

## Log lines to look for
`[SETAFIX] saved N gradient tiles ...` (phase 0), `[SETAFIX] SEAM-SoS validator:
...` (phase 1), `[SETAFIX] '<expert>': aligned overlap ... -> no shared child`,
`[SETAFIX] split report -> ... counts={...} experts now=K`.
