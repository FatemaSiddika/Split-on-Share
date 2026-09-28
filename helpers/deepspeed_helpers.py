# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed.comm as dist
import os
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import os
import math
import torch
from transformers import (
    AutoConfig,
    GenerationConfig,
)
from huggingface_hub import snapshot_download
from transformers.integrations import HfDeepSpeedConfig # transformers.integrations under certain transformers version.
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.nn as nn
import logging
import torch.nn.functional as F
import warnings
from helpers.model_names import LLAMA_8B_MODEL_NAMES

GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

# Deepspeed examples official websites: https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-VisualChat/utils/ds_utils.py#L9
# Explains refer to PDF: https://intro-llm.github.io/chapter/LLM-TAP.pdf
# get train_ds_config and get_eval_ds_config


def get_train_ds_config(offload,
                        dtype,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision=False,
                        tb_path="step1_tensorboard",
                        tb_name="SMT",
                        profiler_path=""):

    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    elif dtype == "fp32":
        data_type = "fp16"
        dtype_config = {"enabled": False}
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "reduce_bucket_size": 1e6,
        "memory_efficient_linear": False
    }
    if enable_mixed_precision:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator(
            ).device_count()
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        },
        "flops_profiler": {
            "enabled": False,
            "profile_step": 10,
            "module_depth": -1,
            "top_modules": 1,
            "detailed": False,
            "output_file": f"{profiler_path}/flops_profiler.log",
        }
    }


# Deepspeed examples official websites: https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-VisualChat/utils/ds_utils.py#L9
# Explains refer to PDF: https://intro-llm.github.io/chapter/LLM-TAP.pdf
# get train_ds_config and get_eval_ds_config


def get_eval_ds_config(offload, dtype, stage=0):
    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {
            "enabled": True,
        }
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/module/lora.py#L144
def make_model_gradient_checkpointing_compatible(model):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def print_rank_0(msg, rank=None):
    if rank is not None and rank <= 0:
        print(msg)
    else:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(msg)
        else:
            print(msg)


def synchronize_index_list(args, index_list, rank=None):
    if rank is not None and rank <= 0:
        torch.save(index_list,
                   f'./index_channel_{args.num_attention_channel}.txt')
        torch.distributed.barrier()
        # print(f"saved tmp, test!!:", index_list)
        return index_list
    else:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                torch.save(
                    index_list,
                    f'./index_channel_{args.num_attention_channel}.txt')
                torch.distributed.barrier()
                # print(f"saved tmp, test!!:", index_list)
                return index_list
            else:
                torch.distributed.barrier()
                tmp = torch.load(
                    f'./index_channel_{args.num_attention_channel}.txt')
                # print(f"loaded tmp, test!!:", tmp)
                return tmp
        else:
            return index_list


def load_save_grad_info(args, index_list=None, load=True):
    sub_folder = 'grad_info'
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    load_subfile = f'group_num_{args.cur_group_num - 1}.txt'
    save_subfile = f'group_num_{args.cur_group_num}.txt'
    load_file = os.path.join(output_dir, load_subfile)
    save_file = os.path.join(output_dir, save_subfile)

    if load:
        if torch.distributed.is_initialized():
            # to revise
            tmp = torch.load(load_file)
            # print(f"loaded tmp, test!!:", tmp)
            torch.distributed.barrier()
            return tmp
        else:
            tmp = torch.load(load_file)
            # print(f"loaded tmp, test!!:", tmp)
            return tmp
    else:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                torch.save(index_list, save_file)
                torch.distributed.barrier()
                # print(f"saved tmp, test!!:", index_list)
            else:
                torch.distributed.barrier()

        else:
            torch.save(index_list, save_file)


def load_save_group_param(args, grouped_param=None, load=True):

    sub_folder = 'grad_info'
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    param_subfile = f'grouped_param.txt'
    file = os.path.join(output_dir, param_subfile)

    if load:
        if torch.distributed.is_initialized():
            # to revise
            torch.distributed.barrier()
            grouped_param = torch.load(file)
            # print(f"loaded tmp, test!!:", grouped_param)
            torch.distributed.barrier()

            return grouped_param
        else:
            grouped_param = torch.load(file)
            # print(f"loaded tmp, test!!:", tmp)
            return grouped_param
    else:
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                torch.save(grouped_param, file)
                torch.distributed.barrier()
                torch.distributed.barrier()
                return grouped_param
                # print(f"saved tmp, test!!:", grouped_param)
        else:
            torch.save(grouped_param, file)


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
# TODO(Hector, Sida): fix bug for llama3-8B, Llama3-8B use AutoTokenizer instead of LlamaTokenizer.
# TODO(Sida): look into QWen family tokenizers.
# rewrite the code for llama
def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if model_name_or_path in ["yahma/llama-13b-hf", "NousResearch/Llama-2-13b-hf",
                              "yahma/llama-7b-hf", "meta-llama/Llama-2-7b-chat-hf"]:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path,
            fast_tokenizer=fast_tokenizer,
            add_bos_token=False)  # not adding start token
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            fast_tokenizer=fast_tokenizer,
            add_bos_token=False)  # not adding start token
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.padding_side = 'right'
        if "DeepSeek-R1-Distill" not in model_name_or_path:
            tokenizer.padding_side = 'left'
            tokenizer.pad_token_id = 0
    return tokenizer


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def load_hf_tokenizer(model_name_or_path,
                      max_seq_len,
                      fast_tokenizer=True,
                      add_special_tokens=None):

    if os.path.exists(model_name_or_path):
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file.get("_name_or_path",
                                             model_name_or_path)
            tokenizer = get_tokenizer(model_name,
                                      fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = get_tokenizer(model_name_or_path,
                                  fast_tokenizer=fast_tokenizer)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    tokenizer.model_max_length = max_seq_len
    return tokenizer


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def save_hf_format(model, tokenizer, args, sub_folder=""):
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    # need one more step to convert SMT into normal llama model.
    # Refer to LoRA code to see the implementation

    torch.save(save_dict, output_model_file)

    # free GPU memory
    del save_dict

    model_to_save.config.to_json_file(output_config_file)

    '''if "DeepSeek-R1-Distill" in tokenizer.name_or_path:
        tokenizer.save_pretrained(output_dir)
    else:
        tokenizer.save_vocabulary(output_dir)'''
    tokenizer.save_pretrained(output_dir)


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_json_file(path):
    def parse_json_lines(f):
        return [json.loads(line) for line in f]

    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_json(content):
        return json.loads(content)

    try:
        content = read_file(path)
        return load_json(content)
    except json.JSONDecodeError:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return parse_json_lines(f)
        except json.JSONDecodeError as error:
            logging.error(f"Failed to parse JSON: {error}")
            return None


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups


# Souce code from deepspeed official example websites:
# Code can refer to: https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/perf.py#L10
def print_throughput(hf_model, args, e2e_time, rank=0):
    if rank <= 0:
        hf_config = hf_model.config
        num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_seq_len
        batch_size = args.per_device_ft_batch_size
        samples_per_second = batch_size / e2e_time
        checkpoint_activations_factor = 4

        hf_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in hf_model.parameters()
        ])
        params_in_billions = hf_model._num_params / (1e9)

        train_flops_per_iteration = calculate_flops(
            checkpoint_activations_factor, batch_size, seq_length, hf_config)

        train_tflops = train_flops_per_iteration / (e2e_time * gpus_per_model *
                                                    (10**12))

        param_string = f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        print(
            f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, Sequence Length: {seq_length}"
        )


# Souce code from deepspeed official example websites:
# Code can refer to: https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/perf.py#L10
# Helper function to calculate FLOPs using the Megatron-LM paper's formula
def calculate_flops(checkpoint_activations_factor, batch_size, seq_length,
                    hf_config):
    num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)
    flops_per_iteration = (24 * checkpoint_activations_factor * batch_size *
                           seq_length * num_layers * (hidden_size**2)) * (
                               1.0 + (seq_length / (6.0 * hidden_size)) +
                               (vocab_size /
                                (16.0 * num_layers * hidden_size)))
    return flops_per_iteration


# Souce code from deepspeed official example websites:
# Code can refer to: https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/perf.py#L10
def get_hf_configs(hf_config):
    num_layers = getattr(hf_config, "num_hidden_layers",
                         getattr(hf_config, "n_layer", None))
    hidden_size = getattr(hf_config, "hidden_size",
                          getattr(hf_config, "n_embd", None))
    vocab_size = getattr(hf_config, "vocab_size", None)
    assert all(
        (num_layers, hidden_size, vocab_size)
    ), "Could not determine number of layers, hidden size, and vocab size of the model"

    return num_layers, hidden_size, vocab_size


# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
# This function is a modified version of code available in the from_pretrained API of HuggingFace Transformers
# The code is copied and modified from: https://github.com/huggingface/transformers/blob/5ee9693a1c77c617ebc43ef20194b6d3b674318e/src/transformers/modeling_utils.py#L498
# This function helps load a HF format checkpoint into a DeepSpeed wrapped model that has been sharded using ZeRO Stage 3
def load_state_dict_into_model(model_to_load=None,
                               state_dict=None,
                               start_prefix="",
                               zero_stage=0):
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)

    del state_dict

    return error_msgs


# Source Code from: DeepSpeed example official website model_utils
# Refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py#L19
def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


# Source Code from: DeepSpeed example official website model_utils
# Refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py#L19


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    trained=False,
                    dropout=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    configure_dropout(model_config, dropout)

    # specially define llama3 model to fix the tokenizer issue in current hf version
    if model_name_or_path in LLAMA_8B_MODEL_NAMES or model_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
        or model_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":
        tokenizer.pad_token_id = 0

    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if trained:
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    return model


# Source Code from: DeepSpeed example official website model_utils
# Refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py#L19
def create_hf_trained_model(model_class,
                            model_name_or_path,
                            tokenizer,
                            ds_config=None,
                            trained=False,
                            dropout=None):
    # Get generation config from llama
    if os.path.isdir(model_name_or_path):
        with open(os.path.join(model_name_or_path, 'config.json')) as f:
            model_config = json.load(f)
        print(f'--------parent model: {model_config["_name_or_path"]}--------')
    generation_config = GenerationConfig.from_pretrained(
        model_config["_name_or_path"])

    model = create_hf_model(model_class, model_name_or_path, tokenizer,
                            ds_config, trained, dropout)
    if trained:
        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        print(f"Loading model checkpoint from {model_ckpt_path}")
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
        err_msg = load_state_dict_into_model(model,
                                             model_ckpt_state_dict,
                                             "",
                                             zero_stage=0)
        if len(err_msg) > 0:
            print_rank_0(err_msg)

        model.generation_config = generation_config
    return model
