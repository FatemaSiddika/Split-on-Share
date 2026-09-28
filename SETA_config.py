import os
from enum import Enum
from typing import Dict, Sequence
import torch
import transformers
from torch.utils.data import Dataset
import copy
import re
import logging
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig,
    AutoConfig,
    # GenerationConfig,
)
from torch.utils.data import random_split
import math
from typing import Optional, List, Union
import json
from dataclasses import dataclass, field
from config import PeftConfig, PeftType
# from peft import LoraConfig

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

@dataclass
class SETAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    num_submatrix_mlp: int = field(default=0, metadata={"help": "number of submatrices applied to MLP layers"})
    num_submatrix_attn: int = field(default=0, metadata={"help": "number of submatrices applied to Attention layers"})
    SETA_dropout: Optional[float] = field(default=0.0)
    # to fix, need to be more flexible
    model_name: str = field(
        default=None,
        metadata={
            "help" : "name of the llama model"
            "choose from ['yahma/llama-13b-hf', 'NousResearch/Llama-2-13b-hf', 'NousResearch/Llama-2-7b-hf', 'yahma/llama-7b-hf', 'meta-llama/Llama-2-7b-chat-hf', 'NousResearch/Meta-Llama-3-8B', 'meta-llama/Meta-Llama-3-8B']"
        },
    )
    full_ft_steps: int = field(default=100, metadata={"help": "number of iterations for full fine-tuning before SMT"})
    selection_strategy: str = field(default="no_restriction", metadata={"help": "sub-matrices distribution selection strategy"})
    calculation_strategy: str = field(default="mean_abs", metadata={"help": "gradient calculation within submatrices"})


    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v', 'k'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.SMT

def SETA_create_and_prepare_model(args, fft_model_args, data_args):

    peft_config = None
    peft_config = SETAConfig(
        num_submatrix_mlp = args.num_submatrix_mlp,
        num_submatrix_attn = args.num_submatrix_attn,
        SETA_dropout = args.SETA_dropout,
        model_name = args.model_name_or_path,
        full_ft_steps = fft_model_args.full_ft_steps,
        selection_strategy = args.selection_strategy,
        calculation_strategy = args.calculation_strategy,
        target_modules = args.target_modules,
        merge_weights = False,
    )

    add_special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        add_special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        add_special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    # Always load the tokenizer from the actual local model directory.
    # Previously this read _name_or_path from config.json and passed that to
    # get_tokenizer, which caused LLaMA-3 to fail because the local path
    # doesn't contain "Meta-Llama-3" and fell through to LlamaTokenizer.
    tokenizer = get_tokenizer(args.model_name_or_path, fast_tokenizer=True)

    if add_special_tokens is not None:
        add_special_tokens = [add_special_tokens] if isinstance(add_special_tokens, str) \
            else add_special_tokens
        tokenizer.add_special_tokens(
            {'additional_special_tokens': add_special_tokens})

    tokenizer.model_max_length = data_args.max_seq_length

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            dropout=args.SETA_dropout)

    return model, peft_config, tokenizer

# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    """
    Architecture-agnostic tokenizer loader.
    Supports LLaMA-2/3, Qwen3, Gemma, Mistral and any HuggingFace model.
    """
    model_lower = model_name_or_path.lower()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=fast_tokenizer,
        trust_remote_code=True,    # required for Qwen3
        add_bos_token=False,
    )

    # Qwen3: disable thinking mode — SETA uses standard instruction format
    if "qwen" in model_lower and hasattr(tokenizer, 'enable_thinking'):
        tokenizer.enable_thinking = False
        print(f"  [Tokenizer] Qwen3 thinking mode disabled.")

    # Ensure pad token is set — LLaMA-2, some Gemma variants don't set one
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"  [Tokenizer] pad_token set to eos_token ({tokenizer.eos_token!r})")

    tokenizer.padding_side = 'right'
    return tokenizer


# Source Code from: DeepSpeed example official website model_utils
# Refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py#L19
def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    dropout = None
                    ):
    model_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True)
    configure_dropout(model_config, dropout)

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=True,   # required for Qwen3
    )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # actually i do not know why we need this, but commenting it causes cuda error
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


# Source Code from: DeepSpeed example official website model_utils
# Refer to https://github.com/microsoft/DeepSpeedExamples/blob/75df1d7250452bcc7c065797a95c982bc8caab0b/applications/DeepSpeed-Chat/dschat/utils/model/model_utils.py#L19
def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]


# Prompt generate class. Source code from LLM-Apdaptor
# Source code link: https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/finetune.py
def generate_prompt(instruction = None, input = None, output = None):
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



# load from local
# the dataloader function refer to DeepSpeed Examples:
# Source Code please refer to:
# https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/step1_supervised_finetuning/main.py
# DataLoaders creation:
def create_local_datasets(tokenizer, data_args, training_args, apply_chat_template=False, use_second_db=False):
    # try:
    logging.warning(f"Data full path: {data_args.dataset_name_or_path}")

    if use_second_db:
        training_data = SupervisedDataset(data_path=data_args.dataset_name_or_path2, tokenizer=tokenizer)
    else:
        training_data = SupervisedDataset(data_path=data_args.dataset_name_or_path, tokenizer=tokenizer)
    
    train_size = len(training_data) - data_args.eval_set_size
    eval_size = data_args.eval_set_size

    train_dataset, eval_dataset = random_split(training_data, [train_size, eval_size])

    # NEW: Sample only a percentage of training data
    print(f'train_sample_ratio: "{data_args.train_sample_ratio}"')
    if data_args.train_sample_ratio < 1.0:
        original_train_size = len(train_dataset)
        reduced_train_size = int(original_train_size * data_args.train_sample_ratio)
        
        print(f"Reducing training data from {original_train_size} to {reduced_train_size} samples ({data_args.train_sample_ratio*100:.0f}%)")
        
        # Randomly sample the reduced training set
        train_indices = torch.randperm(len(train_dataset))[:reduced_train_size]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    print(f'training data from {len(train_dataset)}')
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # except Exception as error:
    #     raise ValueError('Failed to load data. Please verify the data path and format.') from error

    return train_dataset, eval_dataset, collator

def create_local_datasets_task(tokenizer, data_args, training_args, apply_chat_template=False, use_second_db=False):

    # train_path = os.path.join(data_args.dataset_name_or_path, "train.json")
    # train_dataset = SupervisedDataset(data_path=train_path, tokenizer=tokenizer)
    
    if use_second_db:
        print(f"Data full path: {data_args.dataset_name_or_path2}")
        logging.warning(f"data_args.dataset_name_or_path2: {data_args.dataset_name_or_path2}")
        train_dataset = SupervisedDataset(data_path=os.path.join(data_args.dataset_name_or_path2, "train.json"), tokenizer=tokenizer)
        test_path = os.path.join(data_args.dataset_name_or_path2, "test.json")
        eval_dataset = SupervisedDataset(data_path=test_path, tokenizer=tokenizer)
    else:
        print(f"Data full path: {data_args.dataset_name_or_path}")
        logging.warning(f"Data full path: {data_args.dataset_name_or_path}")
        train_dataset = SupervisedDataset(data_path=os.path.join(data_args.dataset_name_or_path, "train.json"), tokenizer=tokenizer)
        test_path = os.path.join(data_args.dataset_name_or_path, "test.json")
        eval_dataset = SupervisedDataset(data_path=test_path, tokenizer=tokenizer)

    # NEW: Sample only a percentage of training data
    if data_args.train_sample_ratio < 1.0:
        original_train_size = len(train_dataset)
        reduced_train_size = int(original_train_size * data_args.train_sample_ratio)

        print(f"Reducing training data from {original_train_size} to {reduced_train_size} samples ({data_args.train_sample_ratio*100:.0f}%)")
        
        # Randomly sample reduced training set
        train_indices = torch.randperm(original_train_size)[:reduced_train_size]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return train_dataset, eval_dataset, collator

# load from huggingface
def create_hf_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        for conversation in samples["messages"]:
            batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
        return {"content": batch}

    raw_datasets = DatasetDict()
    for split in data_args.splits.split(","):
        try:
            # Try first if dataset on a Hub repo
            # dataset = load_dataset(data_args.dataset_name, split=split)
            dataset = load_dataset(data_args.dataset_name_or_path, split=split)
        except DatasetGenerationError:
            # If not, check local dataset
            dataset = load_from_disk(os.path.join(data_args.dataset_name, split))

        if "train" in split:
            raw_datasets["train"] = dataset
        elif "test" in split:
            raw_datasets["test"] = dataset
        else:
            raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if apply_chat_template:
        raw_datasets = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data, None

# def get_output_or_chosen(example):
#     if 'output' in example:
#         return example['output']
#     elif 'answer' in example:
#         return example['answer']
#     else:
#         raise ValueError('wrong fine-tuning data json format, must include output or answer key in the data dict')


# def get_instruction_or_prompt(example):
#     if 'input' in example and example['input'] != '':
#         return example['input']
#     elif 'instruction' in example:
#         return example['instruction']
#     else:
#         raise ValueError('wrong fine-tuning data json format, must include input or instruction key in the data dict')


# Modify existing field getter functions to handle FLAN format: previously i used the prev 2 methods
def get_instruction_or_prompt(example):
    # Handle FLAN format first
    if 'inputs' in example:  # FLAN format
        return example['inputs']
    # Your existing format
    elif 'input' in example and example['input'] != '':
        return example['input']
    elif 'instruction' in example:
        return example['instruction']
    else:
        raise ValueError('wrong fine-tuning data json format, must include inputs, input or instruction key')

def get_output_or_chosen(example):
    # Handle FLAN format first  
    if 'targets' in example:  # FLAN format
        return example['targets']
    # Your existing format
    elif 'output' in example:
        return example['output']
    elif 'answer' in example:
        return example['answer']
    else:
        raise ValueError('wrong fine-tuning data json format, must include targets, output or answer key')



# Data process, tokenizer function, and dataset are followed instruction from: Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions
# Open Source webstie refer to: https://github.com/sahil280114/codealpaca/blob/master/train.py
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = read_json_file(data_path)
        logging.warning("Formatting inputs...")

        sources = [
            generate_prompt(instruction= get_instruction_or_prompt(example))
            for example in list_data_dict
        ]

        # targets = [f"{get_output_or_chosen(example).replace('</s>', '')}{tokenizer.eos_token}" for example in list_data_dict]
        targets = [f"{get_output_or_chosen_v2(example).replace('</s>', '').replace(tokenizer.eos_token, '')}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()
#         logging.warning("Loading data...")
#         list_data_dict = read_json_file(data_path)
#         logging.warning("Formatting inputs...")

#         sources = [
#             generate_prompt(instruction=get_instruction_or_prompt(example))
#             for example in list_data_dict
#         ]
#         targets = [
#             f"{get_output_or_chosen(example).replace('</s>', '')}{tokenizer.eos_token}"
#             for example in list_data_dict
#         ]

#         logging.warning("Tokenizing inputs... This may take some time...")
#         data_dict = preprocess(sources, targets, tokenizer)

#         # Ensure these are lists of torch.Tensor
#         self.input_ids = [torch.tensor(x, dtype=torch.long) for x in data_dict["input_ids"]]
#         self.labels = [torch.tensor(x, dtype=torch.long) for x in data_dict["labels"]]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
#         return {
#             "input_ids": self.input_ids[idx],
#             "labels": self.labels[idx]
#         }


# Data process, tokenizer function, and dataset are followed instruction from: Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions
# Open Source webstie refer to: https://github.com/sahil280114/codealpaca/blob/master/train.py
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# Source code rewriten from DeepSpeed Examples official website
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/main.py#L308
def evaluation(model, dataloader, device, eval_iters = 0):
    model.eval()
    total_loss = 0.0
    num_steps = len(dataloader)

    for i, data in enumerate(dataloader):
        data = to_device(data, device)
        with torch.no_grad():
            result = model(**data)
        total_loss += result.loss.float()

        # Explicitly delete data variable
        del data

    # Explicitly delete result variable
    del result

    avg_loss = total_loss / num_steps
    try:
        avg_loss = get_all_reduce_mean(avg_loss)
        perplexity = torch.exp(avg_loss).item()
    except Exception as e:
        perplexity = float('inf')

    model.train()
    return perplexity, avg_loss.item()




# source code from: Deepspeed example website.
# https://github.com/microsoft/DeepSpeedExamples/blob/cce62236a2c8f52d5548f310e64ee09ed2785416/applications/DeepSpeed-Chat/dschat/utils/utils.py
def read_json_file(path):
    def parse_json_lines(f):
        return [json.loads(line) for line in f]

    def read_file(file_path):
        print(f'Loading data from {file_path}')
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


# Data process, tokenizer function, and dataset are followed instruction from: Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions
# Open Source webstie refer to: https://github.com/sahil280114/codealpaca/blob/master/train.py
def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing.

    [FIX-2c RESPONSE-PRESERVING TRUNCATION]
    ---------------------------------------
    The original implementation tokenised ``s + t`` as ONE string and tokenised
    ``s`` separately, both with right-truncation at ``model_max_length``:

        examples_tokenized, sources_tokenized = [_tokenize_fn(x, tokenizer) ...]
        label[:source_len] = IGNORE_INDEX

    When the prompt alone is longer than the window (MeetingBank transcripts,
    long 20Minuten articles) BOTH get clipped to exactly ``model_max_length``.
    ``source_len`` then equals the full sequence length, so the mask loop sets
    EVERY position to IGNORE_INDEX and the sample has zero supervised tokens.

    Measured on MeetingBank / qwen3_4b at max_seq_length=2048:
        seq_len mean=1333 max=2048 | at truncation cap  : 198/500 (39.6%)
                                   | ZERO valid labels  : 192/500 (38.4%)

    Downstream, a batch of such samples produced sum(CE)/tiny_denominator,
    which is how Phase-2 train_loss reached 5.69e4 on llama2_7b and 6.24e4 on
    qwen3_4b — two models with a 4.7x vocabulary difference landing within 10%
    of each other, i.e. a normalisation artefact rather than a numerical
    blow-up.  Raising max_seq_length only shifts the threshold; it cannot fix
    the ordering, because the response always sits at the END of the sequence.

    The fix tokenises the TARGET first, reserves room for it, and keeps only the
    TAIL of the prompt.  The supervised span is therefore always retained; what
    gets dropped is transcript prefix, which is the correct trade for
    summarisation.  Sequences that already fit are byte-identical to before.
    """
    print_rank_0('-----------------')
    print_rank_0(examples_preview(sources, targets))

    max_len = int(getattr(tokenizer, "model_max_length", 2048) or 2048)

    # Preserve the original BOS behaviour: _tokenize_fn called tokenizer() with
    # default add_special_tokens=True, so a BOS is prepended where the tokenizer
    # defines one (LLaMA-2 yes, Qwen3 no).  Detect it once and re-add manually.
    _probe = tokenizer("x")["input_ids"]
    _bos = (tokenizer.bos_token_id
            if (getattr(tokenizer, "bos_token_id", None) is not None
                and len(_probe) > 0 and _probe[0] == tokenizer.bos_token_id)
            else None)

    input_ids, labels = [], []
    n_prompt_trimmed = 0
    for s, t in zip(sources, targets):
        tgt_ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        src_ids = tokenizer(s, add_special_tokens=False)["input_ids"]

        reserved = len(tgt_ids) + (1 if _bos is not None else 0)
        if reserved >= max_len:
            # Pathological: the target alone fills the window.  Keep as much of
            # the target as possible and at least one prompt token.
            tgt_ids = tgt_ids[: max(1, max_len - 1 - (1 if _bos is not None else 0))]
            reserved = len(tgt_ids) + (1 if _bos is not None else 0)

        budget = max_len - reserved
        if len(src_ids) > budget:
            src_ids = src_ids[-budget:]          # keep the TAIL of the prompt
            n_prompt_trimmed += 1

        if _bos is not None:
            src_ids = [_bos] + src_ids

        ids = torch.tensor(src_ids + tgt_ids, dtype=torch.long)
        lab = ids.clone()
        lab[: len(src_ids)] = IGNORE_INDEX

        input_ids.append(ids)
        labels.append(lab)

    if n_prompt_trimmed:
        print_rank_0(f"[FIX-2c] left-truncated the prompt on {n_prompt_trimmed}/"
                     f"{len(sources)} samples to keep the response span "
                     f"(max_seq_length={max_len}).")

    return dict(input_ids=input_ids, labels=labels)


def examples_preview(sources, targets):
    """Preserve the original debug print of the first fully-formatted example."""
    if not sources:
        return ""
    return sources[0] + targets[0]


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
def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

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



# Data process, tokenizer function, and dataset are followed instruction from: Code Alpaca: An Instruction-following LLaMA Model trained on code generation instructions
# Open Source webstie refer to: https://github.com/sahil280114/codealpaca/blob/master/train.py
def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    ids_list = tokenizer(
        strings,
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=False
    )['input_ids']

    input_ids = []
    input_ids_lens = []

    for ids in ids_list:
        input_ids.append(torch.tensor(ids))
        input_ids_lens.append(len(ids))

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )

def gradient_collection(model, mlp_warmup_grads, attention_warmup_grads, num_submatrix, num_submatrix_attentoion):
    pattern = re.compile(r'model\.layers\.(\d+)\.')
    for name, param in (model.module if hasattr(model, "module") else model).named_parameters():
        if param.grad is None:
            continue
        # Without DeepSpeed use param.grad directly; sanitize nan/inf from bf16 overflow
        grad = torch.nan_to_num(param.grad.detach().cpu().to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)
        match = pattern.search(name)
        layer_number = int(match.group(1)) if match else None
        if 'mlp' in name and num_submatrix > 0:
            module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
            if (module_name, layer_number) not in mlp_warmup_grads:
                mlp_warmup_grads[(module_name, layer_number)] = grad
            else:
                mlp_warmup_grads[(module_name, layer_number)] += grad

        if 'self_attn' in name and num_submatrix_attentoion > 0:
            module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
            if module_name is not None:
                if (module_name, layer_number) not in attention_warmup_grads:
                    attention_warmup_grads[(module_name, layer_number)] = grad
                else:
                    attention_warmup_grads[(module_name, layer_number)] += grad
        del grad
    return mlp_warmup_grads, attention_warmup_grads


# def gradient_collection_finetune(model, mlp_finetune_grads, attention_finetune_grads, num_submatrix, num_submatrix_attentoion):
#     from deepspeed.utils import safe_get_full_grad
#     pattern = re.compile(r'model\.layers\.(\d+)\.')
    
#     # Handle model wrapping
#     if hasattr(model, 'module'):
#         named_parameters = (model.module if hasattr(model, "module") else model).named_parameters()
#     else:
#         named_parameters = model.named_parameters()
    
#     for name, param in named_parameters:
#         # Skip parameters that don't require gradients
#         if not param.requires_grad:
#             continue
            
#         match = pattern.search(name)
#         layer_number = int(match.group(1)) if match else None
        
#         if 'mlp' in name and num_submatrix > 0:
#             grad = safe_get_full_grad(param)
#             if grad is None:
#                 continue
                
#             module_name = 'gate_proj' if 'gate_proj' in name else 'up_proj' if 'up_proj' in name else 'down_proj'
            
#             # Only process if this parameter was selected for training
#             if (module_name, layer_number) not in mlp_finetune_grads:
#                 mlp_finetune_grads[(module_name, layer_number)] = grad.detach().cpu().to(torch.float32)
#             else:
#                 existing_grad = mlp_finetune_grads[(module_name, layer_number)]
#                 new_grad = grad.detach().cpu().to(torch.float32)
                
#                 # Shape validation
#                 if existing_grad.shape != new_grad.shape:
#                     print(f"Shape mismatch for {module_name} layer {layer_number}: "
#                           f"existing {existing_grad.shape} vs new {new_grad.shape}")
#                     # Reinitialize with new shape
#                     mlp_finetune_grads[(module_name, layer_number)] = new_grad
#                 else:
#                     mlp_finetune_grads[(module_name, layer_number)] += new_grad
#             del grad
        
#         if 'self_attn' in name and num_submatrix_attentoion > 0:
#             module_name = 'q_proj' if 'q_proj' in name else 'k_proj' if 'k_proj' in name else 'v_proj' if 'v_proj' in name else None
            
#             if module_name is not None:
#                 grad = safe_get_full_grad(param)
#                 if grad is None:
#                     continue
                    
#                 if (module_name, layer_number) not in attention_finetune_grads:
#                     attention_finetune_grads[(module_name, layer_number)] = grad.detach().cpu().to(torch.float32)
#                 else:
#                     existing_grad = attention_finetune_grads[(module_name, layer_number)]
#                     new_grad = grad.detach().cpu().to(torch.float32)
                    
#                     if existing_grad.shape != new_grad.shape:
#                         print(f"Shape mismatch for {module_name} layer {layer_number}: "
#                               f"existing {existing_grad.shape} vs new {new_grad.shape}")
#                         attention_finetune_grads[(module_name, layer_number)] = new_grad
#                     else:
#                         attention_finetune_grads[(module_name, layer_number)] += new_grad
#                 del grad
    
#     return mlp_finetune_grads, attention_finetune_grads


def gradient_collection_finetune(model, mlp_finetune_grads, attention_finetune_grads,
                                 num_submatrix, num_submatrix_attentoion):

    pattern = re.compile(r'model\.layers\.(\d+)\.')

    if hasattr(model, 'module'):
        named_parameters = (model.module if hasattr(model, "module") else model).named_parameters()
    else:
        named_parameters = model.named_parameters()

    for name, param in named_parameters:
        match = pattern.search(name)
        layer_number = int(match.group(1)) if match else None

        # Skip if not part of a relevant layer
        if layer_number is None:
            continue

        # Determine if it's an MLP or Attention parameter
        is_mlp = 'mlp' in name and num_submatrix > 0
        is_attn = 'self_attn' in name and num_submatrix_attentoion > 0

        if not is_mlp and not is_attn:
            continue

        # Decide module name
        if is_mlp:
            module_name = 'gate_proj' if 'gate_proj' in name else \
                          'up_proj' if 'up_proj' in name else \
                          'down_proj'
            grad_dict = mlp_finetune_grads
        elif is_attn:
            if 'q_proj' in name:
                module_name = 'q_proj'
            elif 'k_proj' in name:
                module_name = 'k_proj'
            elif 'v_proj' in name:
                module_name = 'v_proj'
            else:
                continue  # skip irrelevant attention modules
            grad_dict = attention_finetune_grads

        # Fetch gradient — use param.grad directly (no deepspeed)
        grad = param.grad

        # If gradient is None or param is frozen, store a zero tensor of the same shape
        if grad is None or not param.requires_grad:
            grad_tensor = torch.zeros_like(param.data).cpu().to(torch.float32)
        else:
            grad_tensor = torch.nan_to_num(grad.detach().cpu().to(torch.float32), nan=0.0, posinf=1e4, neginf=-1e4)

        # Replace previous (we only keep latest epoch)
        grad_dict[(module_name, layer_number)] = grad_tensor

        # Cleanup
        del grad_tensor

    return mlp_finetune_grads, attention_finetune_grads

def gradient_collection_finetune2(model, mlp_finetune_grads, attention_finetune_grads,
                                 num_submatrix, num_submatrix_attentoion):
    # Alias to gradient_collection_finetune — same logic
    return gradient_collection_finetune(model, mlp_finetune_grads, attention_finetune_grads,
                                        num_submatrix, num_submatrix_attentoion)
#     from deepspeed.utils import safe_get_full_grad
#     import torch
#     import re

#     pattern = re.compile(r'model\.layers\.(\d+)\.')

#     if hasattr(model, 'module'):
#         named_parameters = (model.module if hasattr(model, "module") else model).named_parameters()
#     else:
#         named_parameters = model.named_parameters()

#     for name, param in named_parameters:
#         match = pattern.search(name)
#         layer_number = int(match.group(1)) if match else None

#         # Skip if not part of a relevant layer
#         if layer_number is None:
#             continue

#         # Determine if it's an MLP or Attention parameter
#         is_mlp = 'mlp' in name and num_submatrix > 0
#         is_attn = 'self_attn' in name and num_submatrix_attentoion > 0

#         if not is_mlp and not is_attn:
#             continue

#         # Decide module name
#         if is_mlp:
#             module_name = 'gate_proj' if 'gate_proj' in name else \
#                           'up_proj' if 'up_proj' in name else \
#                           'down_proj'
#             grad_dict = mlp_finetune_grads
#         elif is_attn:
#             if 'q_proj' in name:
#                 module_name = 'q_proj'
#             elif 'k_proj' in name:
#                 module_name = 'k_proj'
#             elif 'v_proj' in name:
#                 module_name = 'v_proj'
#             else:
#                 continue  # skip irrelevant attention modules
#             grad_dict = attention_finetune_grads

#         # Fetch gradient
#         grad = safe_get_full_grad(param)

#         # If gradient is None or param is frozen, store a zero tensor of the same shape
#         if grad is None or not param.requires_grad:
#             grad_tensor = torch.zeros_like(param.data).cpu().to(torch.float32)
#         else:
#             grad_tensor = grad.detach().cpu().to(torch.float32)

#         # Accumulate gradients instead of replacing
#         key = (module_name, layer_number)
#         if key in grad_dict:
#             grad_dict[key] += grad_tensor  # Cumulative addition
#         else:
#             grad_dict[key] = grad_tensor  # Initialize

#         # Cleanup
#         del grad_tensor

#     return mlp_finetune_grads, attention_finetune_grads

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




# for new datataset Oct 16
def get_instruction_or_prompt_v2(example):
    """
    Extract instruction/input from various dataset formats
    """
    # TRACE Benchmark format (most datasets)
    if 'prompt' in example:
        return example['prompt']
    # Alternative formats (if any TRACE datasets use these)
    elif 'question' in example:
        return example['question']
    elif 'text' in example:
        return example['text']
    # FLAN format
    elif 'inputs' in example:
        return example['inputs']
    # Standard format
    elif 'input' in example and example['input'] != '':
        return example['input']
    elif 'instruction' in example:
        return example['instruction']
    else:
        available_keys = list(example.keys())
        raise ValueError(f'Unsupported format. Available keys: {available_keys}')

def get_output_or_chosen_v2(example):
    """
    Extract output/answer from various dataset formats
    """
    # TRACE Benchmark format (most datasets)
    if 'answer' in example:
        return example['answer']
    # Alternative formats (if any TRACE datasets use these)
    elif 'response' in example:
        return example['response']
    elif 'label' in example:
        return example['label']
    # FLAN format  
    elif 'targets' in example:
        return example['targets']
    # Standard format
    elif 'output' in example:
        return example['output']
    else:
        available_keys = list(example.keys())
        raise ValueError(f'Unsupported format. Available keys: {available_keys}')

# NEW Dataset class for TRACE Benchmark
class SupervisedDatasetV2(Dataset):
    """Dataset for supervised fine-tuning - Version 2 (supports TRACE Benchmark format)"""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDatasetV2, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = read_json_file(data_path)
        logging.warning("Formatting inputs...")

        sources = [
            generate_prompt(instruction=get_instruction_or_prompt_v2(example))
            for example in list_data_dict
        ]

        targets = [f"{get_output_or_chosen_v2(example).replace('</s>', '')}{tokenizer.eos_token}" for example in
                   list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# NEW function for TRACE Benchmark datasets
def create_trace_benchmark_datasets(tokenizer, data_args, training_args, use_second_db=False):
    """
    Load datasets from TRACE-Benchmark format with eval.json, train.json, test.json
    Uses SupervisedDatasetV2 which supports the prompt/answer format
    """
    if use_second_db:
        base_path = data_args.dataset_name_or_path2
    else:
        base_path = data_args.dataset_name_or_path
    
    logging.warning(f"Loading TRACE Benchmark data from: {base_path}")
    
    # Load train and eval datasets
    train_path = os.path.join(base_path, "train.json")
    eval_path = os.path.join(base_path, "eval.json")
    
    assert os.path.exists(train_path), f"Train file not found: {train_path}"
    assert os.path.exists(eval_path), f"Eval file not found: {eval_path}"
    
    # Use the V2 dataset class that supports prompt/answer format
    train_dataset = SupervisedDatasetV2(data_path=train_path, tokenizer=tokenizer)
    eval_dataset = SupervisedDatasetV2(data_path=eval_path, tokenizer=tokenizer)
    
    # Apply train sample ratio if specified
    if hasattr(data_args, 'train_sample_ratio') and data_args.train_sample_ratio < 1.0:
        original_train_size = len(train_dataset)
        reduced_train_size = int(original_train_size * data_args.train_sample_ratio)
        
        print(f"Reducing training data from {original_train_size} to {reduced_train_size} samples ({data_args.train_sample_ratio*100:.0f}%)")
        
        train_indices = torch.randperm(original_train_size)[:reduced_train_size]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    print(f"Loaded TRACE Benchmark: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return train_dataset, eval_dataset, collator