"""
run_defaults.py — default run settings for Split-on-Share (paper configuration).

main.py calls apply(sys.argv) before parsing its arguments. It fills in every
argument and environment setting that is not given, so a run only needs:

    python main.py --use_model_name qwen3_4b --task_list "A,B,C" \
                   --current_task_id 2 --run_phase phase2_train \
                   --run_dir trained_models/qwen3_4b/my_run

Precedence: a command-line flag > an environment variable > the default here.

Settings that can be changed through the environment:
    SEED (100)  BLOCK_BUDGET_FRACTION (0.25)  SETAFIX_PHI (0.025)  SETAFIX_Z (3.0)
    NO_EXPERT_SHARING (False)  LEARNING_RATE (9.65e-6)  LAMDA (0.1)
    LAMBDA_MODE (dynamic)  GATING_RL (False)  PROJECTION_MODE (v_proj_only)
    MAX_SAMPLES (500 eval samples per task)  DATASET (5000 | 500)
    DATA_ROOT (<repo>/data_files/TRACE-Benchmark)  MODEL_ROOT (<repo>/models)
    and every SETAFIX_* / GATE_* variable in ENV_DEFAULTS below.

`python utils/run_defaults.py --check-data <task,task,...>` lists tasks whose
data folder is missing (used by the launchers before submitting).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Environment read by the model / split code (os.environ.get) ──────────────
# Set only when not already set. MIN_SHARED_FRAC_NEW is left unset on purpose:
# the split code then uses SETAFIX_PHI for it. PRUNING_MIN_BLOCKS is left unset
# on purpose as well (the code falls back to 8, as in all paper runs).
ENV_DEFAULTS = {
    "BLOCK_BUDGET_FRACTION": "0.25",
    "PROJECTION_MODE": "v_proj_only",
    "GATE_MODE": "sigmoid",
    "GATE_OPEN_BIAS": "1.0",
    "USE_REENTRANT": "False",
    "BUDGET_EXPANSION_CAP": "1.0",
    "LABEL_AUDIT_STRICT": "True",
    "LABEL_AUDIT_SAMPLES": "512",
    "LOSS_SANITY_OFF": "0",
    "SETAFIX_SPLIT": "seam",
    "SETAFIX_SCORE": "svd",
    "SETAFIX_Z": "3.0",
    "SETAFIX_SIGN": "on",
    "SETAFIX_NULL": "empirical",
    "SETAFIX_SIDE": "right",
    "SETAFIX_ORTHO": "on",
    "SETAFIX_GATE_PROTECT": "freeze",
    "SETAFIX_LAYER_GATE": "budget",
    "SETAFIX_PHI": "0.025",
    "SETAFIX_LAYER_FRAC": "0.25",
    "SETAFIX_LAYER_MIN": "3",
    "SETAFIX_MIN_SHARED_FRAC": "0.15",
    "SETAFIX_NOEVIDENCE": "share",
    "SETAFIX_SHARE_Q": "1.0",
    "SETAFIX_MAX_SHARED_PER_TASK": "0",
    "SETAFIX_STRUCTURE_ONLY": "0",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "SOS_FFT_ROOT": os.path.join(ROOT, "FFT_steps"),
}

# Generation tasks always read the 500-sample set.
GEN_TASKS = ("MeetingBank", "20Minuten")
MODELS = ("llama3_8b", "llama2_7b", "qwen3_4b", "qwen3_8b", "qwen3_14b")

PROJECTION_ALIASES = {"v_proj": "v_proj_only", "vqk_proj": "qkv_proj", "qvk_proj": "qkv_proj",
                      "qkv": "qkv_proj", "vqk": "qkv_proj", "mlp": "mlp_proj"}
TARGET_MODULES = {"v_proj_only": '["v_proj"]', "q_proj": '["q_proj"]', "k_proj": '["k_proj"]',
                  "qkv_proj": '["q_proj", "k_proj", "v_proj"]',
                  "mlp_proj": '["gate_proj", "up_proj", "down_proj"]'}

# Fixed training / trainer arguments of every paper run.
CONSTANT_ARGS = [
    ("--chat_template_format", "None"), ("--use_peft_SETA", "True"),
    ("--SETA_dropout", "0.0"), ("--SETA_offload", "False"),
    ("--SETA_learning_rate", "1e-4"), ("--SETA_w_decay", "0.0"),
    ("--selection_strategy", "no_restriction"), ("--calculation_strategy", "mean_abs"),
    ("--eval_set_size", "4"), ("--append_concat_token", "False"), ("--packing", "False"),
    ("--num_train_epochs", "3"), ("--logging_steps", "10"), ("--log_level", "info"),
    ("--logging_strategy", "steps"), ("--eval_steps", "999999"), ("--eval_strategy", "no"),
    ("--save_strategy", "no"), ("--hub_private_repo", "False"), ("--hub_strategy", "every_save"),
    ("--lr_scheduler_type", "linear"), ("--weight_decay", "0.0"), ("--warmup_steps", "2"),
    ("--max_grad_norm", "1.0"), ("--per_device_eval_batch_size", "1"),
    ("--gradient_checkpointing", "True"), ("--use_reentrant", "False"),
    ("--full_ft_steps", "100"), ("--load_best_model_at_end", "False"),
    ("--metric_for_best_model", "None"), ("--save_steps", "999999"), ("--save_total_limit", "0"),
    ("--train_sample_ratio", "1"), ("--skip_eval_task_list", ""),
    ("--expert_creation_threshold", "8"),
    ("--fft_stopping_mode", "dynamic"), ("--layer_norm_scoring", "True"),
    ("--divergence_scoring", "True"), ("--budget_mode", "adaptive"),
    ("--report_to", "none"),
]


def _env(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _model_args(model):
    block = "128" if model in ("llama3_8b", "qwen3_4b", "qwen3_8b", "qwen3_14b") else "256"
    small = "8" if model == "qwen3_4b" else "10"
    return [("--block_dimension", block), ("--pruning_min_blocks", small),
            ("--tiny_remainder_threshold", small)]


def _task_args(task, model):
    """(max_seq_length, train batch, grad accum). MeetingBank needs a long
    window or its response span is truncated away; the effective batch stays 8."""
    if task == "MeetingBank":
        seq = "2048" if model in ("llama2_7b", "qwen3_4b", "qwen3_8b") else "1536"
        return seq, "1", "8"
    if task == "20Minuten":
        return ("512", "1", "8") if model == "llama3_8b" else ("768", "2", "4")
    return "512", "2", "4"


def data_path(task):
    root = _env("DATA_ROOT", os.path.join(ROOT, "data_files", "TRACE-Benchmark"))
    dataset = "500" if task in GEN_TASKS else _env("DATASET", "5000")
    if dataset not in ("500", "5000"):
        sys.exit(f"ERROR: DATASET must be 500 or 5000 (got '{dataset}')")
    return os.path.join(root, f"LLM-CL-Benchmark_{dataset}")


def _get(argv, flag):
    if flag in argv:
        i = argv.index(flag)
        return argv[i + 1] if i + 1 < len(argv) else ""
    return None


def _pop(argv, flag):
    v = _get(argv, flag)
    if v is not None:
        i = argv.index(flag)
        del argv[i:i + 2]
    return v


def apply(argv):
    """Return argv with every missing argument filled in; set env defaults."""
    for k, v in ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    proj = PROJECTION_ALIASES.get(os.environ["PROJECTION_MODE"], os.environ["PROJECTION_MODE"])
    if proj not in TARGET_MODULES:
        sys.exit(f"ERROR: PROJECTION_MODE must be one of {'|'.join(TARGET_MODULES)} (got '{proj}')")
    os.environ["PROJECTION_MODE"] = proj

    argv = list(argv)
    run_dir = _pop(argv, "--run_dir")
    model = _get(argv, "--use_model_name")
    tasks = _get(argv, "--task_list")
    if model is None or tasks is None or _get(argv, "--current_task_id") is None:
        return argv                     # manual / json use: leave untouched
    if model not in MODELS:
        sys.exit(f"ERROR: unsupported model '{model}' (one of {', '.join(MODELS)})")
    task_list = [t.strip() for t in tasks.split(",")]
    tid = int(_get(argv, "--current_task_id"))
    task = task_list[tid - 1]
    phase = _get(argv, "--run_phase") or "auto"
    eval_ids = [x for x in os.environ.get("EVAL_TASK_IDS", "").replace(":", ",").split(",") if x.strip()]
    is_eval = phase == "phase3_eval"
    seq, batch, accum = _task_args(task, model)

    fill = [("--seed", _env("SEED", "100")),
            ("--model_name_or_path", os.path.join(_env("MODEL_ROOT", os.path.join(ROOT, "models")), model)),
            ("--num_submatrix_mlp", "1" if proj == "mlp_proj" else "0"),
            ("--num_submatrix_attn", "0" if proj == "mlp_proj" else "1"),
            ("--target_modules", TARGET_MODULES[proj]),
            ("--base_dataset_path", data_path(task)),
            ("--learning_rate", _env("LEARNING_RATE", "9.65e-6")),
            ("--per_device_train_batch_size", batch), ("--gradient_accumulation_steps", accum),
            ("--max_seq_length", seq),
            ("--lamda", _env("LAMDA", "0.1")), ("--lambda_mode", _env("LAMBDA_MODE", "dynamic")),
            ("--gating_rl", _env("GATING_RL", "False")),
            ("--max_samples", _env("MAX_SAMPLES", "500")),
            ("--do_train", "False" if is_eval else "True"),
            ("--do_eval", "True" if is_eval else "False"),
            ("--eval_current_task_only", "True" if is_eval else "False"),
            ("--eval_task_id", eval_ids[0] if eval_ids else str(tid)),
            ("--block_budget_fraction", os.environ["BLOCK_BUDGET_FRACTION"]),
            ("--no_expert_sharing", _env("NO_EXPERT_SHARING", "False")),
            ] + _model_args(model) + CONSTANT_ARGS
    if run_dir is not None:
        run_dir = os.path.abspath(run_dir)
        ckpt = lambda i: os.path.join(run_dir, f"shared_moE_checkpoints_{model}_{i}_{task_list[i - 1].lower()}")
        # the final row (EVAL_TASK_IDS set) loads the last checkpoint itself, no resume
        resume = ckpt(tid - 1) if tid > 1 and not eval_ids else "None"
        fill += [("--manual_output_path", ckpt(tid)), ("--output_dir", ckpt(tid)),
                 ("--resume_checkpoint_path", resume),
                 ("--ablation_tag", os.path.basename(run_dir))]
    for flag, value in fill:
        if flag not in argv:
            argv += [flag, value]
    return argv


def run_tag():
    """Folder-name part for the current settings: the budget, PHI and Z always,
    plus any other setting that differs from its default."""
    env = {k: _env(k, v) for k, v in ENV_DEFAULTS.items()}
    tag = f"bbf{env['BLOCK_BUDGET_FRACTION']}_phi{env['SETAFIX_PHI']}_z{env['SETAFIX_Z']}"
    if _env("NO_EXPERT_SHARING", "False") == "True":
        tag += "_noshare"
    skip = ("BLOCK_BUDGET_FRACTION", "SETAFIX_PHI", "SETAFIX_Z", "PYTORCH_CUDA_ALLOC_CONF", "SOS_FFT_ROOT")
    for k, v in ENV_DEFAULTS.items():
        if k not in skip and env[k] != v:
            tag += f"_{k.replace('SETAFIX_', '').lower()}{env[k]}"
    for k, v in (("LEARNING_RATE", "9.65e-6"), ("LAMDA", "0.1"), ("LAMBDA_MODE", "dynamic"),
                 ("GATING_RL", "False"), ("MAX_SAMPLES", "500")):
        if _env(k, v) != v:
            tag += f"_{k.lower()}{_env(k, v)}"
    return tag


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--tag":
        print(run_tag())
        sys.exit(0)
    if len(sys.argv) == 3 and sys.argv[1] == "--check-data":
        missing = [t for t in sys.argv[2].split(",") if t and not (
            os.path.isfile(os.path.join(data_path(t), t, "train.json"))
            or os.path.isfile(os.path.join(data_path(t), t, "test.json")))]
        for t in missing:
            print(f"ERROR: no data for task '{t}' under {os.path.join(data_path(t), t)}", file=sys.stderr)
        sys.exit(1 if missing else 0)
    print(__doc__)
