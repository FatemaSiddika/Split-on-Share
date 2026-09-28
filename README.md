# Split-on-Share — how to run

The method is described in `README_SETAFIX.md`. This file covers the folder
layout and the commands. Run everything from the project root.

## Folder layout
```
Split-on-Share/
  *.py, helpers/     trainer, model, SEAM validator, eval code, cl_metrics.py
  scripts/           launchers and slurm scripts
  analysis/          analysis-only code; its logs go to analysis/logs/
  FFT_steps/         every FFT_steps_* folder              (created by runs)
  trained_models/    every RL_Lamda_fsys_trained_models_*  (created by runs)
  logs/              training and eval job logs            (created by runs)
```
`scripts/sos_common.sh` holds the shared settings: paths, dataset folders, task
orders, learning-rate tag. The data folders are set at its top (`DATA_ROOT`,
`DATA_PATH_5000`, `DATA_PATH_500`).

## Main settings (environment variables)
| variable | values | default |
|---|---|---|
| `DATASET` | `5000` (8 TRACE tasks) · `500` (15 tasks, includes the 8) | `5000` |
| `TASK_ORDER` | `1`-`3` (15-task lists) · `4` TRACE-paper order (8) · `5` order 1 reversed (8) · or `TASK_LIST_STR=a,b,...` | `1` |
| `PROJECTION_MODE` | `v_proj` · `q_proj` · `k_proj` · `qkv_proj` (alias `vqk_proj`) · `mlp_proj` | `v_proj_only` |
| `SEED` | integer | `100` |
| `LEARNING_RATE` | `--learning_rate` (Phase-2 training, FFT on non-generation tasks) | `9.65e-6` |
| `SUBMIT_MODE` | `chain` (one job per task) · `sequential` · `onejob` (one job for the whole stream) | `chain` |

All other knobs (`ABLATION`, `BLOCK_BUDGET_FRACTION`, `SETAFIX_*`, ...) are
unchanged. With default values the run folder names are the same as before, so
existing runs are found after they are moved (see Migration).

## Train all tasks in ONE job, then LA / OP / BWT
```bash
SUBMIT_MODE=onejob BLOCK_BUDGET_FRACTION=0.12 ABLATION=1 SETAFIX_NULL=empirical \
SETAFIX_ORTHO=on SETAFIX_GATE_PROTECT=freeze SETAFIX_LAYER_GATE=budget SETAFIX_PHI=0.02 \
bash scripts/run_setafix.sh 1-8 fft,split,train,eval qwen3_4b
```
Per task: fft, split, train, eval on its own task (the diagonal). Then the last
checkpoint on tasks 1..N-1 (the final row). Then `cl_metrics.py` prints LA, OP,
BWT and writes `<run folder>/cl_metrics_T<N>.json`. No full grid.

* 15 tasks: `DATASET=500 ... 1-15`. Another 8-task order: `TASK_ORDER=4 ... 1-8`.
* Walltime: `ONEJOB_TIME` (default `7-00:00:00`; the partition limit applies).
* After a timeout or failure: rerun the same command with `FORCE_OVERWRITE=True`;
  finished tasks are skipped (`SKIP_DONE=True`, the default).
* `FINAL_ROW=False` skips the final row and metrics.
* Each task still writes its own log `..._T<id>_<task>_fused_<jobid>.txt`; the
  stream itself logs to `..._ALL_T1-8_<model>_<jobid>_driver.txt`.

One job per task (as before): the same command without `SUBMIT_MODE=onejob`.

## Eval only (same knobs as the training run)
```bash
bash scripts/RL_Lamda_fsys_eval_only_arms.sh 8 qwen3_4b           # final row: T8 on tasks 1..8
bash scripts/RL_Lamda_fsys_eval_only_arms.sh 8 qwen3_4b 3,5       # T8 on tasks 3 and 5
bash scripts/RL_Lamda_fsys_eval_only_arms.sh 1-8 qwen3_4b         # full grid: TN on tasks 1..N
EVAL_SCHEME=diag bash scripts/RL_Lamda_fsys_eval_only_arms.sh 1-8 qwen3_4b   # diagonal only
```
Then, when the jobs are done (the launcher prints this command):
```bash
python cl_metrics.py --base_dir <run folder> --model qwen3_4b \
       --task_list "<task list>" --last 8 [--grid]
```
LA = mean of the diagonal; OP = mean of the final row; BWT = mean over tasks
1..N-1 of (final − diagonal), negative = forgetting. `--grid` also prints every
grid cell that exists.

## Analysis
```bash
bash analysis/run_analysis.sh [--local] [TREE ...]
bash analysis/run_probe.sh [FFT_DIR ...]
python analysis/setafix_report.py <run folder> [<run folder>] --plot
python analysis/setafix_compare.py TREE_A TREE_B --labels "A","B"
```

## Migration of existing SETA_fix runs
```bash
bash scripts/migrate_layout.sh /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/SETA_fix          # dry run
APPLY=1 LOGS=1 bash scripts/migrate_layout.sh /lustre/hdd/LAS/jannesar-lab/fatemask/fed-con/SETA_fix
```
Moves FFT_steps_* into `FFT_steps/`, each run folder into
`trained_models/<same parent name>/`, and (with `LOGS=1`) the logs. It never
overwrites. Do not run it while jobs of those runs are queued or running.

`scripts/legacy/run_setafix_sept20.sh` is the old launcher, kept unmodified (it
still points at SETA_fix).
