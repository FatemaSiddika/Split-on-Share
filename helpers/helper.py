from typing import Dict, Sequence
import copy
import logging
from dataclasses import dataclass
from typing import Dict, Sequence
import numpy as np
from transformers import AutoModelForCausalLM, SchedulerType, get_scheduler
import torch
from torch.utils.data import random_split
import transformers
import gc
import deepspeed
from torch.utils.data import Dataset
import argparse
from helpers.deepspeed_helpers import (
    print_rank_0,
    to_device,
    get_all_reduce_mean,
    read_json_file,
    save_hf_format,
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


# Prompt generate class. Source code from LLM-Apdaptor
# Source code link: https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/finetune.py
def generate_prompt(instruction=None, input=None, output=None):
    # sorry about the formatting disaster gotta move fast
    if instruction and input and output:
        return f"""<s> Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""  # noqa: E501

    elif instruction and input:
        return f"""<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
{output}"""  # noqa: E501

    else:
        return f"""<s> Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""  # noqa: E501


def get_output_or_chosen(example):
    if 'output' in example:
        return example['output']
    elif 'answer' in example:
        return example['answer']
    else:
        raise ValueError(
            'wrong fine-tuning data json format, must include output or answer key in the data dict'
        )


def get_instruction_or_prompt(example):
    if 'input' in example and example['input'] != '':
        return example['input']
    elif 'instruction' in example:
        return example['instruction']
    else:
        raise ValueError(
            'wrong fine-tuning data json format, must include input or instruction key in the data dict'
        )

def get_question_solution_answer_for_limo(example):
    if 'question' in example and 'solution' in example and 'answer' in example:
        return example['question'], example['solution'], example['answer']
    else:
        raise ValueError(
            'wrong LIMO dataset format.'
        )

# Data process, tokenizer function, and dataset are followed instruction from: Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions
# Open Source webstie refer to: https://github.com/sahil280114/codealpaca/blob/master/train.py
def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    ids_list = tokenizer(strings,
                         max_length=tokenizer.model_max_length,
                         truncation=True,
                         return_attention_mask=False)['input_ids']

    input_ids = []
    input_ids_lens = []

    for ids in ids_list:
        input_ids.append(torch.tensor(ids))
        input_ids_lens.append(len(ids))

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


# Data process, tokenizer function, and dataset are followed instruction from: Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions
# Open Source webstie refer to: https://github.com/sahil280114/codealpaca/blob/master/train.py
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    print_rank_0('-----------------')
    print_rank_0(examples[0])
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


# Data process, tokenizer function, and dataset are followed instruction from: Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions
# Open Source webstie refer to: https://github.com/sahil280114/codealpaca/blob/master/train.py
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = read_json_file(data_path)
        logging.warning("Formatting inputs...")
        
        if 'limo' in data_path.lower():
            sources, targets = [], []
            for example in list_data_dict:
                question, solution, _ = get_question_solution_answer_for_limo(example)
                sources.append(question)
                targets.append(solution + tokenizer.eos_token)
        else:
            sources = [
                generate_prompt(instruction=get_instruction_or_prompt(example))
                for example in list_data_dict
            ]
            print(1111111111111)
            print(sources[0])
            print(222222222222222222)
            targets = [
                f"{get_output_or_chosen(example).replace('</s>', '')}{tokenizer.eos_token}"
                for example in list_data_dict
            ]
            print(targets[0])
            print(3333333333333)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# Data process, tokenizer function, and dataset are followed instruction from: Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions
# Open Source webstie refer to: https://github.com/sahil280114/codealpaca/blob/master/train.py
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# Source code rewriten from DeepSpeed Examples official website, modified for better memory efficiency
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py#L308
def evaluation(model, dataloader, device, eval_iters=0):
    model.eval()
    # Ensure use_cache is False during evaluation to reduce memory usage
    # when use_cache = False, unable KV cache, reduce memory but increase inference/eval time
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    total_loss = 0.0
    num_steps = len(dataloader)

    # Use torch.inference_mode for better memory efficiency
    with torch.inference_mode():
        for i, data in enumerate(dataloader):
            data = to_device(data, device)
            with torch.no_grad():
                # Ensure use_cache is False during evaluation to reduce memory usage
                # when use_cache = False, unable KV cache, reduce memory but increase inference/eval time
                result = model(**data, use_cache=False)
            total_loss += result.loss.float()

            # Explicitly delete data variable
            # Explicitly delete result variable
            del data
            del result
            gc.collect()
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_steps
    try:
        avg_loss = get_all_reduce_mean(avg_loss)
        perplexity = torch.exp(avg_loss).item()
    except Exception as e:
        perplexity = float('inf')

    model.train()
    return perplexity, avg_loss.item()


# Source code rewriten from codealpaca
# https://github.com/sahil280114/codealpaca/blob/master/train.py
def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    try:
        print_rank_0(f"Data full path: {data_args.data_path[0]}",
                     data_args.global_rank)

        training_and_eval_data = SupervisedDataset(data_path=data_args.data_path[0],
                                          tokenizer=tokenizer)

        train_size = int(len(training_and_eval_data) * (1 - data_args.eval_set_ratio))
        eval_size = len(training_and_eval_data) - train_size

        print("Training data size", train_size, "validation data set", eval_size)

        train_dataset, eval_dataset = random_split(training_and_eval_data,
                                                   [train_size, eval_size])

        collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    except Exception as error:
        raise ValueError(
            'Failed to load data. Please verify the data path and format.'
        ) from error

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator
    }


def final_eval_save_model(model, eval_dataloader, device, best_eval_loss,
                          final_saved_model_index, tokenizer, best_model,
                          args):
    current_ppl, current_eval_loss = evaluation(model, eval_dataloader, device)

    if current_eval_loss < best_eval_loss:
        best_eval_loss = current_eval_loss
        if args.global_rank == 0:
            best_model = copy.deepcopy(model.module).cpu()

    print_rank_0(
        f"Validation perplexity: {current_ppl}, Validation loss: {current_eval_loss}, Best validation loss: {best_eval_loss}",
        args.global_rank)
    print_rank_0(f"Saving the best model at step {final_saved_model_index}",
                 args.global_rank)

    model_to_save = best_model if best_model is not None else model
    print_rank_0('Saving the final model...', args.global_rank)

    if args.global_rank == 0:
        try:
            save_hf_format(model_to_save, tokenizer, args)
        except Exception as e:
            raise ValueError(
                "Save model error! This may be caused by an incorrect output directory. Double-check your model structure and output directory."
            ) from e


def epoch_save_model(model, eval_dataloader, device, best_eval_loss, epoch_num,
                     tokenizer, args):
    current_ppl, current_eval_loss = evaluation(model, eval_dataloader, device)
    print_rank_0(
        f"Validation perplexity: {current_ppl}, Validation loss: {current_eval_loss}, Best validation loss: {best_eval_loss}",
        args.global_rank)
    print_rank_0(f"Saving the model at the last iteration at epoch {epoch_num}",
                 args.global_rank)

    # print_rank_0('Saving the final model...', args.global_rank)

    if args.global_rank == 0:
        try:
            save_hf_format(model,
                           tokenizer,
                           args,
                           sub_folder=f'epoch_{epoch_num}')
        except Exception as e:
            raise ValueError(
                "Save model error! This may be caused by an incorrect output directory. Double-check your model structure and output directory."
            ) from e


def group_save_model(model, eval_dataloader, device, cur_group_num, tokenizer,
                     args):
    current_ppl, current_eval_loss = evaluation(model, eval_dataloader, device)
    print_rank_0(
        f"Validation perplexity: {current_ppl}, Validation loss: {current_eval_loss}",
        args.global_rank)
    print_rank_0(f"Saving the best model after group {cur_group_num}...",
                 args.global_rank)

    if args.global_rank == 0:
        try:
            save_hf_format(model,
                           tokenizer,
                           args,
                           sub_folder=f'group_{cur_group_num}')
        except Exception as e:
            raise ValueError(
                "Save model error! This may be caused by an incorrect output directory. Double-check your model structure and output directory."
            ) from e


def iteration_save_model(model, eval_dataloader, device, best_eval_loss,
                         current_step_count, tokenizer, args):
    current_ppl, current_eval_loss = evaluation(model, eval_dataloader, device)
    print_rank_0(
        f"Validation perplexity: {current_ppl}, Validation loss: {current_eval_loss}, Best validation loss: {best_eval_loss}",
        args.global_rank)
    print_rank_0(f"Saving the best model at iteraion {current_step_count}",
                 args.global_rank)
    if args.global_rank == 0:
        try:
            save_hf_format(model,
                           tokenizer,
                           args,
                           sub_folder=f'iteration_{current_step_count}')
        except Exception as e:
            raise ValueError(
                "Save model error! This may be caused by an incorrect output directory. Double-check your model structure and output directory."
            ) from e


def save_pretrained_model(model, tokenizer, args):
    # current_ppl, current_eval_loss = evaluation(model, eval_dataloader, device)
    print_rank_0(
        f"Saving the initial pretrained model {args.model_name_or_path}",
        args.global_rank)

    if args.global_rank == 0:
        try:
            save_hf_format(model,
                           tokenizer,
                           args,
                           sub_folder=f'{args.model_name_or_path}')
        except Exception as e:
            raise ValueError(
                "Save model error! This may be caused by an incorrect output directory. Double-check your model structure and output directory."
            ) from e
    exit()
