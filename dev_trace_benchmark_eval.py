import os
import json
import torch
from transformers import AutoTokenizer, GenerationConfig
import tqdm
import math
import gc
from typing import List, Dict
from helpers.TRACE_metrics import (
    eval_FOMC,
    eval_SciQA,
    eval_CStance,
    eval_NumGLUE,
    eval_PapyrusF,
    eval_20Minuten,
    eval_MeetingBank,
)
import copy

# Prompt template matching your training format
PROMPT_TEMPLATE = '''<s> Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{prompt}

### Response:
'''

import re

def normalize_numeric_output(text: str) -> str:
    """Extracts the last number found in the text for NumGLUE evaluation."""
    text = text.strip()
    # Regex to find numbers (integers or decimals)
    # Matches '42', '42.5', '-10', etc.
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]  # Return the last number found (standard for CoT)
    return ""


def eval_MMLU(generated_texts, references):
    """Evaluation for MMLU (Multiple Choice A/B/C/D)"""
    correct = 0
    total = len(generated_texts)
    for gen, ref in zip(generated_texts, references):
        pred = normalize_to_ABCD(gen)
        target = normalize_to_ABCD(ref)
        if pred == target and target != "":
            correct += 1
    accuracy = correct / total if total > 0 else 0
    return {'accuracy': accuracy}

def eval_BBH(generated_texts, references):
    """Evaluation for BBH (Exact Match / finding the answer in text)"""
    correct = 0
    total = len(generated_texts)
    for gen, ref in zip(generated_texts, references):
        gen = gen.lower().strip()
        ref = ref.lower().strip()
        if ref in gen or gen == ref:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    return {'accuracy': accuracy}

def eval_PIQA(generated_texts, references):
    """Evaluation for PIQA (Binary Choice: 'Solution 1' vs 'Solution 2')"""
    correct = 0
    total = len(generated_texts)
    for gen, ref in zip(generated_texts, references):
        gen = gen.lower().strip()
        target_idx = "1" if "1" in ref else "2" if "2" in ref else ref
        
        pred_idx = None
        if "solution 1" in gen or "(1)" in gen or " 1 " in gen or gen == "1":
            pred_idx = "1"
        elif "solution 2" in gen or "(2)" in gen or " 2 " in gen or gen == "2":
            pred_idx = "2"
            
        if pred_idx == target_idx:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    return {'accuracy': accuracy}

def eval_StringLabel(generated_texts, references):
    """Accuracy for string-label classification (dbpedia, amazon, yahoo, agnews, yelp).
    Prefix-tolerant match: exact, pred-startswith-gold, or gold-startswith-first-word."""
    def _match(pred, gold):
        p = (pred or "").strip().lower()
        g = (gold or "").strip().lower()
        if not p or not g:
            return False
        if p == g or p.startswith(g):
            return True
        first_word = p.split()[0] if p.split() else ""
        return bool(first_word) and g.startswith(first_word)
    total = len(generated_texts)
    correct = sum(1 for gen, ref in zip(generated_texts, references) if _match(gen, ref))
    return {'accuracy': correct / total if total > 0 else 0}

def eval_Py150(generated_texts, references):
    """Char-level similarity for Py150 code completion (difflib ratio, as in reference project).
    Normalizes TRACE placeholder strings so literal '<EOL>' vs real newlines don't depress scores."""
    import difflib
    def _norm(s):
        s = (s or "").replace("<EOL>", "\n").replace("</s>", "").replace("<s>", "")
        return s.strip()
    sims = []
    for gen, ref in zip(generated_texts, references):
        g, r = _norm(gen), _norm(ref)
        sims.append(difflib.SequenceMatcher(None, g, r).ratio() if g and r else 0.0)
    return {'similarity': sum(sims) / len(sims) if sims else 0.0}

EVAL_FUNCS = {
    "FOMC": eval_FOMC,
    "ScienceQA": eval_SciQA,
    "C-STANCE": eval_CStance,
    "NumGLUE-cm": eval_NumGLUE,
    "NumGLUE-ds": eval_NumGLUE, 
    "20Minuten": eval_20Minuten,
    "MeetingBank": eval_MeetingBank,
    "MMLU": eval_MMLU,
    "BBH": eval_BBH,
    "PIQA": eval_PIQA,
    "MedMCQA": eval_MMLU, 
    "JEC-QA": eval_MMLU,  
    "Lima": None,
    "Py150": eval_Py150,
    "dbpedia": eval_StringLabel,
    "amazon":  eval_StringLabel,
    "yahoo":   eval_StringLabel,
    "agnews":  eval_StringLabel,
    "yelp":    eval_StringLabel,
    "BoolQA":  eval_StringLabel,
    "QQP":     eval_StringLabel,
}

def normalize_to_ABC(text: str) -> str:
    """Map any generation to one of {A,B,C} or '' if not found."""
    s = (text or "").strip().upper()
    # Fast path: pick first ASCII A/B/C that appears
    for ch in s:
        if ch in ("A", "B", "C"):
            return ch
    # Optional fallbacks for Chinese cues:
    # if "支持" in s: return "A"
    # if "反对" in s: return "B"
    # if "中立" in s: return "C"
    return ""

def normalize_to_ABCD(text: str) -> str:
    """Map any generation to one of {A,B,C,D} or '' if not found."""
    s = (text or "").strip().upper()
    for ch in s:
        if ch in ("A", "B", "C", "D"):
            return ch
    return ""

def load_hf_tokenizer(model_name_or_path, max_seq_length=2048, fast_tokenizer=True):
    """Load tokenizer matching your training setup"""
    if "Meta-Llama-3" in model_name_or_path or "Llama-3.1" in model_name_or_path or "Llama-3.2" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    else:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, add_bos_token=False)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    
    tokenizer.model_max_length = max_seq_length
    return tokenizer



# ── KV cache for generation (speed only; outputs must not change) ────────────
#   EVAL_USE_CACHE  auto (default) | on | off
#     off   the historical setting: use_cache=False, every new token re-runs the
#           whole prompt. Correct but slow: ~1-2 s per token on a 2k-token prompt.
#     on    use the KV cache without checking.
#     auto  before each dataset, generate the first EVAL_CACHE_CHECK_N prompts
#           twice (cache off, cache on) for EVAL_CACHE_CHECK_TOKENS tokens with the
#           task's own settings, and use the cache only if the token ids are
#           identical; otherwise stay on use_cache=False. This matters because a
#           gate that looks at the whole sequence (instead of each token) would
#           route differently under the cache, and the scores would change.
def _cache_matches_nocache(model, device, tokenizer, prompts, gen_kwargs,
                           n_prompts=2, n_tokens=16):
    """True if cached and uncached greedy generation give the same token ids
    on the first n_prompts prompts (same tokenisation as
    generate_text_completions)."""
    kw = dict(gen_kwargs)
    kw["max_new_tokens"] = min(int(kw.get("max_new_tokens", n_tokens)), n_tokens)
    kw["min_new_tokens"] = min(int(kw.get("min_new_tokens", 1)), kw["max_new_tokens"])
    for p in prompts[:n_prompts]:
        tok = tokenizer([p], padding='longest', return_tensors="pt",
                        truncation=True, max_length=3584)
        ids = tok.input_ids.to(device)
        am = tok.attention_mask.to(device)
        with torch.no_grad():
            a = model.generate(input_ids=ids, attention_mask=am,
                               **{**kw, "use_cache": False})
            b = model.generate(input_ids=ids, attention_mask=am,
                               **{**kw, "use_cache": True})
        same = a.shape == b.shape and bool(torch.equal(a.cpu(), b.cpu()))
        del ids, am, tok, a, b
        if not same:
            return False
    return True


@torch.no_grad()
def generate_text_completions(model, device, tokenizer, prompt_list, batch_size=1,
                               stop_sequences=None, verbose=False, **generation_options):
    """Generate text completions for prompts with explicit memory cleanup and robust error handling"""
    results = []
    return_sequences_count = generation_options.get("num_return_sequences", 1)
    progress_bar = tqdm.tqdm(total=len(prompt_list), desc="Generating responses")
    
    # Safe generation space
    SAFE_MAX_LENGTH = 3584

    for start_idx in range(0, len(prompt_list), batch_size):
        current_batch = prompt_list[start_idx:start_idx + batch_size]

        token_data = tokenizer(
            current_batch, 
            padding='longest', 
            return_tensors="pt",
            truncation=True, 
            max_length=SAFE_MAX_LENGTH
        )

        input_ids = token_data.input_ids.to(device)
        attention_masks = token_data.attention_mask.to(device)

        # 1. Initialize placeholders BEFORE try/except to avoid "referenced before assignment"
        batch_results = None
        generated_outputs = None

        try:
            # Direct generation - uses the explicit kwargs passed from caller
            generated_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                **generation_options
            )
        except Exception as error:
            print(f"Generation error: {error}")
            # If generation fails, fill with empty strings
            batch_results = [""] * len(current_batch) * return_sequences_count

        # 2. Decode Success Path
        if batch_results is None:
            if generated_outputs is not None:
                try:
                    # Slice off the prompt tokens to keep only the generated response
                    input_len = input_ids.shape[1]
                    new_tokens = generated_outputs[:, input_len:]

                    # ── DEBUG ──────────────────────────────────────────────
                    print(f"DEBUG raw token IDs: {new_tokens[0].tolist()[:20]}")
                    batch_results = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                    print(f"DEBUG decoded: {repr(batch_results[0][:50])}")
                    # ── END DEBUG ──────────────────────────────────────────
                    
                    # Decode the tokens to text
                    batch_results = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                except Exception as e:
                    print(f"Decoding error: {e}")
                    batch_results = [""] * len(current_batch) * return_sequences_count
            else:
                # Fallback safety net
                batch_results = [""] * len(current_batch) * return_sequences_count

        # 3. Cleanup Memory
        del input_ids, attention_masks, token_data
        if generated_outputs is not None:
            del generated_outputs
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results.extend(batch_results)
        progress_bar.update(len(current_batch))

    progress_bar.close()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

def get_result_value(result: dict, dataset_name: str):
    """Extract the main metric from evaluation result - MATCHES evaluation.py logic"""
    if result is None:
        return -1
    
    if dataset_name == "20Minuten":
        return round(result['sari'][0]['sari'] / 100, 6)
    if dataset_name in ["C-STANCE", "FOMC", "NumGLUE-cm", "NumGLUE-ds", "ScienceQA", 
                        "MMLU", "BBH", "PIQA", "MedMCQA", "JEC-QA",
                        "dbpedia", "amazon", "yahoo", "agnews", "yelp","BoolQA", "QQP"]:
        return round(result['accuracy'], 6)
    elif dataset_name == "Py150":
        return round(result.get('similarity', 0), 6) # Changed 3 to 5
    elif dataset_name == "MeetingBank":
        return round(result['rouge-L'], 6) # Changed 3 to 5
    elif dataset_name == "Lima":
        # Lima typically doesn't have automatic metrics
        return -1
    else:
        # Default: try to get accuracy or return 0
        return round(result.get('accuracy', 0), 6) # Changed 3 to 5


@torch.no_grad()
def evaluate_trace_benchmark(
        SETAtrainer, 
        model=None,
        tokenizer=None,
        data_path="data_files/TRACE-Benchmark/LLM-CL-Benchmark_500/",
        datasets=["FOMC", "20Minuten"],
        output_dir="trace_eval_results",
        max_samples=None,
        per_device_eval_batch_size=1,
        dtype='bf16',
        seed=1234,
        use_eval_split=True):

    torch.manual_seed(seed)
    
    print("\n" + "="*70)
    print(" EVALUATING SMT MODEL ON TRACE BENCHMARK DATASETS")
    print("="*70)

    # 1. ROBUST MODEL RETRIEVAL
    # Check if we have a trainer, otherwise use the provided model
    current_model = None
    if SETAtrainer is not None:
        current_model = SETAtrainer.model
        tokenizer = SETAtrainer.tokenizer
        print(f' Using model and tokenizer from SMT trainer...')
    elif model is not None:
        current_model = model
        print(f' Using provided model object...')
    else:
        raise ValueError("Must provide either 'SETAttrainer' or 'model' to evaluate_trace_benchmark")
    
    # ------------------------------------------------------
    
    print(f" Using tokenizer...")
    tokenizer.padding_side = "left"  # Important for generation
    print(f"  Tokenizer padding side: {tokenizer.padding_side}")
    
    # Setup model for evaluation
    device = next(current_model.parameters()).device
    print(f" Model device: {device}")
    
    if dtype == 'fp16':
        target_dtype = torch.float16
    elif dtype == 'bf16':
        target_dtype = torch.bfloat16
    elif dtype == 'fp32':
        target_dtype = torch.float32
    elif dtype == 'auto':
        target_dtype = next(current_model.parameters()).dtype
    else:
        target_dtype = torch.float32
    
    current_model = current_model.to(target_dtype)
    current_model.eval()
    print(f" Model dtype: {current_model.dtype}")
    
    # Check SMT architecture
    custom_layers = sum(1 for name, module in current_model.named_modules() 
                       if 'LinearLayer_MatrixSparsity' in str(type(module)))
    
    if custom_layers > 0:
        print(f" SMT model with {custom_layers} custom sparse layers detected!")
    else:
        print(" Warning: No custom sparse layers found")


    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0 # Last resort fallback
    
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list): 
        eos_token_id = eos_token_id[0] # Ensure int for generation config
    
    # Ensure model config matches
    current_model.config.pad_token_id = pad_token_id
    current_model.config.eos_token_id = eos_token_id

    print(f" DEBUG: pad_token_id={pad_token_id}, eos_token_id={eos_token_id}")

    # Generation config
    # generation_config = GenerationConfig(
    #     do_sample=False,
    #     num_beams=4,
    #     temperature=0.0,
    #     max_new_tokens=256,
    #     repetition_penalty=1.1,
    #     # pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
    #     pad_token_id=pad_token_id, 
    #     eos_token_id=tokenizer.eos_token_id,
    #     bos_token_id=getattr(tokenizer, 'bos_token_id', tokenizer.eos_token_id),
    # )
    
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=4, # We will try 4 first
        max_new_tokens=256,
        repetition_penalty=1.1,
        pad_token_id=pad_token_id, 
        eos_token_id=eos_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id else eos_token_id,
        use_cache=True 
    )

    evaluation_results = {}
    split_name = "eval" if use_eval_split else "test"
    
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Processing dataset: {dataset}")
        print(f"{'='*70}")
        
        # Construct path to dataset
        dataset_path = os.path.join(data_path, dataset, f"{split_name}.json")
        
        if not os.path.exists(dataset_path):
            print(f"File not found: {dataset_path}")
            evaluation_results[dataset] = {"error": f"{split_name}.json not found"}
            continue

  
        # Load test data
        print(f" Loading data from: {dataset_path}")
        with open(dataset_path, 'r') as f:
            test_data = json.load(f)
        
        print(f" Total samples in {dataset}: {len(test_data)}")
        
        # Limit samples if specified
        if max_samples is not None and len(test_data) > max_samples:
            print(f" Limiting evaluation to {max_samples} samples")
            test_data = test_data[:max_samples]
        
        # Create prompts using the TRACE Benchmark format
        prompts = [PROMPT_TEMPLATE.format(prompt=example['prompt']) for example in test_data]
        answers = [example['answer'] for example in test_data]
        
        print(f" Generated {len(prompts)} prompts")
        print(f"\n Example prompt:")
        print(f"   {prompts[0][:200]}...")
        print(f" Example answer: {answers[0][:100]}...\n")
        
        # ---------------------------------------------------------
        # ROBUST LOGGING ENABLE
        # ---------------------------------------------------------
        print(" Attempting to enable MoE logging...")
        if hasattr(current_model, "set_gating_logging"):
            # If it is the SMTModel_TASA wrapper
            current_model.set_gating_logging(True)
        else:
            # If it is the raw HF model, manually iterate to find MoE layers
            count = 0
            for module in current_model.modules():
                if hasattr(module, "enable_logging"):
                    module.enable_logging()
                    count += 1
            if count > 0:
                print(f"  Enabled logging on {count} internal MoE layers manually.")

        # Force greedy decoding for Math tasks
        # if dataset in ["NumGLUE-cm", "NumGLUE-ds"]:
        #     generation_config.num_beams = 1
        #     generation_config.do_sample = False
        # else:
        #     generation_config.num_beams = 4 # Keep your default for others

        # generation_config.num_beams = 1
        # generation_config.do_sample = False

        if tokenizer.pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token_id = pad_token_id
        else:
            pad_token_id = tokenizer.pad_token_id

        # FORCE THE MODEL CONFIG UPDATE HERE
        # This repairs the "freshly loaded" base model that lost its config
        if current_model.config.pad_token_id is None:
            current_model.config.pad_token_id = pad_token_id
            print(f" Manually restored model.config.pad_token_id to {pad_token_id}")

        print(f" Generating responses (batch_size={per_device_eval_batch_size})...")
        
        # try:
        #     generated_texts = generate_text_completions(
        #         model=current_model,
        #         device=device,
        #         tokenizer=tokenizer,
        #         prompt_list=prompts,
        #         batch_size=per_device_eval_batch_size, 
        #         stop_sequences=[[tokenizer.eos_token]],
        #         verbose=False,
        #         generation_config=generation_config
        #     )
        # except Exception as e:
        #     print(f" Generation failed: {e}")
        #     evaluation_results[dataset] = {"error": f"Generation failed: {e}"}
        #     continue

        # gen_kwargs = {
        #     "do_sample": False,
        #     "num_beams": 1,          # simpler greedy, easier to debug
        #     "max_new_tokens": 10,    # only need A/B/C — 1-3 tokens, not 256
        #     "min_new_tokens": 1,     # FORCE at least one non-EOS token
        #     "repetition_penalty": 1.1,
        #     "pad_token_id": pad_token_id,
        #     "eos_token_id": eos_token_id,
        #     "use_cache": False,      # disable KV cache — custom MoE layers may not support it
        # }

        _MAX_NEW_TOKENS = {
            "C-STANCE":   10,
            "FOMC":       10,
            "ScienceQA":  10,
            "MMLU":       10,
            "MedMCQA":    10,
            "JEC-QA":     10,
            "PIQA":       10,
            "NumGLUE-cm": 20,
            "NumGLUE-ds": 20,
            "BBH":        50,
            "MeetingBank":512,   # increased: refs avg 150-200 words
            "20Minuten":  256,
            "dbpedia":    20,
            "amazon":     20,
            "yahoo":      20,
            "agnews":     20,
            "yelp":       20,
            "Py150":      256,  # code completion can be longer
        }
        _max_new_tokens = _MAX_NEW_TOKENS.get(dataset, 64)  # 64 as safe fallback

        gen_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": _max_new_tokens,
            "min_new_tokens": 1,
            # Repetition penalty hurts abstractive summarisation (MeetingBank, 20Minuten)
            # where key terms naturally recur. Set to 1.0 for generation tasks.
            "repetition_penalty": 1.3 if dataset not in ["MeetingBank", "20Minuten"] else 1.0,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": False,
        }
        _cache_mode = os.environ.get("EVAL_USE_CACHE", "auto").strip().lower()
        if _cache_mode == "on":
            gen_kwargs["use_cache"] = True
        elif _cache_mode == "auto":
            try:
                _ok = _cache_matches_nocache(
                    current_model, device, tokenizer, prompts, gen_kwargs,
                    n_prompts=int(os.environ.get("EVAL_CACHE_CHECK_N", "2")),
                    n_tokens=int(os.environ.get("EVAL_CACHE_CHECK_TOKENS", "16")))
            except Exception as _e:
                print(f" [EVAL-CACHE] check failed ({type(_e).__name__}: {_e}) "
                      f"— keeping use_cache=False")
                _ok = False
            gen_kwargs["use_cache"] = _ok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f" [EVAL-CACHE] mode={_cache_mode} -> use_cache={gen_kwargs['use_cache']}"
              + ("  (cached == uncached on the check prompts)" if _cache_mode == "auto"
                 and gen_kwargs["use_cache"] else "")
              + ("  (outputs differ with the cache: kept the historical setting)"
                 if _cache_mode == "auto" and not gen_kwargs["use_cache"] else ""))


        print(f" Generating responses (batch_size={per_device_eval_batch_size})...")
        try:
            try:
                generated_texts = generate_text_completions(
                model=current_model,
                device=device,
                tokenizer=tokenizer,
                prompt_list=prompts,
                batch_size=per_device_eval_batch_size, 
                stop_sequences=[[tokenizer.eos_token]],
                verbose=False,
                **gen_kwargs  # <--- This passes the args directly
            )
            except TypeError as e:
                if "NoneType" in str(e) and generation_config.num_beams > 1:
                    print(f" [WARNING] Beam Search failed with padding error. Switching to Greedy Decoding (beams=1) for this task.")
                    # Fallback to Greedy Decoding
                    fallback_config = copy.deepcopy(generation_config)
                    fallback_config.num_beams = 1
                    generated_texts = generate_text_completions(
                        model=current_model,
                        device=device,
                        tokenizer=tokenizer,
                        prompt_list=prompts,
                        batch_size=per_device_eval_batch_size, 
                        stop_sequences=[[tokenizer.eos_token]],
                        verbose=False,
                        generation_config=fallback_config,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id
                    )
                else:
                    raise e # Re-raise if it's not the specific padding error

        except Exception as e:
            print(f" Generation failed: {e}")
            evaluation_results[dataset] = {"error": f"Generation failed: {e}"}
            continue


        generated_texts = [
            g.lstrip().removeprefix('</s>').strip()
            for g in generated_texts
        ]
        
        # ---------------------------------------------------------
        # ROBUST LOGGING EXTRACTION
        # ---------------------------------------------------------
        collected_logs = {}
        
        # Method A: Use Wrapper
        # if hasattr(current_model, "get_and_clear_gating_logs"):
        #     collected_logs = current_model.get_and_clear_gating_logs()
        #     current_model.set_gating_logging(False) # Disable
            
        # # Method B: Manual Iteration (if wrapper is missing)
        # else:
        #     for name, module in current_model.named_modules():
        #         if hasattr(module, "gating_logs") and module.gating_logs:
        #             collected_logs[name] = list(module.gating_logs) # Copy
        #             module.gating_logs = [] # Clear
        #             if hasattr(module, "disable_logging"):
        #                 module.disable_logging() # Disable

        # Save to JSON if we found anything
        if collected_logs:
            serializable_logs = {}
            for k, v in collected_logs.items():
                if v:
                    # Take snapshot of the first batch/step
                    first_step = v[0]
                    # Check if 'probs' is a tensor or list and handle accordingly
                    probs_data = first_step.get('probs')
                    if hasattr(probs_data, 'tolist'):
                        probs_data = probs_data[0].tolist()
                    elif isinstance(probs_data, list):
                        probs_data = probs_data[0] # Assume list of lists if already converted

                    serializable_logs[k] = {
                        "expert_names": first_step.get('expert_names', []),
                        "probs_snapshot": probs_data
                    }
            
            # Construct path 
            log_filename = f"{dataset}_moe_gating_snapshot.json"
            log_path = os.path.join(output_dir, log_filename)
            
            # --- CORRECTED LINES START ---
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # --- CORRECTED LINES END ---
            
            with open(log_path, "w") as f:
                json.dump(serializable_logs, f, indent=2)
            print(f" Saved MoE gating snapshot to: {log_path}")
        
        print(f" Generation complete!")
        print(f"\n Example generation:")
        print(f"   Prompt: {test_data[0]['prompt'][:100]}...")
        print(f"   Target: {answers[0][:100]}...")
        print(f"   Generated: {generated_texts[0][:100]}...\n")
        
        # Evaluate using dataset-specific function
        eval_func = EVAL_FUNCS.get(dataset)
        
        if eval_func is None:
            print(f" Warning: No evaluation function found for {dataset}")
            print(f"  Saving raw outputs only...")
            eval_result = None
            metric_value = -1
        else:
            print(f" Evaluating predictions using {eval_func.__name__}...")
            try:
                if dataset in ["NumGLUE-cm", "NumGLUE-ds"]:
                    cleaned_texts = [normalize_numeric_output(gen) for gen in generated_texts]
                    eval_result = eval_func(cleaned_texts, answers)

                elif dataset == "20Minuten":
                    eval_result = eval_func(prompts, generated_texts, answers)
                else:
                    # Pass model path for dynamic cleaning in MeetingBank
                    if dataset == "MeetingBank":
                        _model_path = getattr(
                            getattr(current_model, "config", None),
                            "_name_or_path", ""
                        ) or ""
                        eval_result = eval_func(generated_texts, answers,
                                                model_name_or_path=_model_path)
                    else:
                        eval_result = eval_func(generated_texts, answers)
                
                metric_value = get_result_value(eval_result, dataset)
                print(f" Evaluation result: {eval_result}")
                print(f" Main metric: {metric_value}")
            except Exception as e:
                print(f" Evaluation failed: {e}")
                # import traceback
                # traceback.print_exc()
                eval_result = None
                metric_value = -1
        
        # Save detailed results
        dataset_output_dir = os.path.join(output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Save predictions in the same format as inference.py
        predictions_file = os.path.join(dataset_output_dir, f"predictions_{split_name}.json")
        predictions_data = []
        for i, (prompt, answer, generated) in enumerate(zip(prompts, answers, generated_texts)):
            predictions_data.append({
                "prompt": test_data[i]['prompt'],  # Original prompt without template
                "answer": answer,
                "generated_text": generated,
            })
        
        with open(predictions_file, "w", encoding='utf-8') as fout:
            json.dump(predictions_data, fout, indent=4, ensure_ascii=False)
        
        print(f" Predictions saved to: {predictions_file}")
        
        # Store results
        evaluation_results[dataset] = {
            "metric_value": metric_value,
            "eval_result": eval_result,
            "samples_evaluated": len(test_data),
            "custom_sparse_layers": custom_layers,
            "predictions_file": predictions_file,
            "split": split_name
        }
    
    # Save overall summary
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    
    overall_summary = {
        "evaluation_config": {
            "max_samples": max_samples,
            "batch_size": per_device_eval_batch_size,
            "dtype": dtype,
            "datasets": datasets,
            "custom_sparse_layers": custom_layers,
            "split": split_name,
            "data_path": data_path
        },
        "results": evaluation_results
    }
    
    with open(summary_file, "w") as fout:
        json.dump(overall_summary, fout, indent=2, ensure_ascii=False)
    
    # Print summary table
    print(f"\n{'='*70}")
    print(" EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} | {'Metric':<10} | {'Samples':<10}")
    print(f"{'-'*20}-+-{'-'*10}-+-{'-'*10}")
    
    for dataset, results in evaluation_results.items():
        if "error" not in results:
            metric_val = results['metric_value']
            n_samples = results['samples_evaluated']
            print(f"{dataset:<20} | {metric_val:<10.5f} | {n_samples:<10}")
        else:
            print(f"{dataset:<20} | {'ERROR':<10} | {'-':<10}")
            print(f"  Error: {results['error']}")
    
    print(f"{'-'*20}-+-{'-'*10}-+-{'-'*10}")
    print(f" Summary saved to: {summary_file}")
    print("="*70 + "\n")
    
    return evaluation_results

# Integration function for your training script
def add_trace_evaluation_to_training(SETAtrainer, training_args, datasets=None):
    """
    Add this function call at the end of your SMT training
    
    Args:
        SETAtrainer: The trained SMT trainer
        training_args: Training arguments
        datasets: List of TRACE datasets to evaluate on
    """
    print("\n Starting TRACE Benchmark evaluation...")
    
    # Default datasets if none specified
    if datasets is None:
        datasets = ["FOMC", "20Minuten", "MeetingBank"]
    
    eval_results = evaluate_trace_benchmark(
        SETAtrainer=SETAtrainer,
        data_path="data_files/TRACE-Benchmark/LLM-CL-Benchmark_500/",
        datasets=datasets,
        output_dir=os.path.join(training_args.output_dir, "trace_evaluation"),
        max_samples=100,  # Limit for faster evaluation during training
        per_device_eval_batch_size=1,
        dtype='bf16',
        use_eval_split=True  # Use eval.json instead of test.json
    )
    
    return eval_results



def analyze_moe_gating(SETA_model, tokenizer, prompt, device="cuda"):
    """
    Runs a single inference, captures gating values, and prints a detailed report.
    """
    print(f"\n{'='*60}\n MoE Gating Analysis \n{'='*60}")
    
    # 1. Prepare Input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 2. Enable Logging
    SETA_model.set_gating_logging(True)
    
    # 3. Run Inference (Generate 1 token to see immediate gating)
    #    or just run a forward pass on the prompt
    with torch.no_grad():
        outputs = SETA_model.model(**inputs)
    
    # 4. Disable Logging & Get Data
    SETA_model.set_gating_logging(False)
    logs = SETA_model.get_and_clear_gating_logs()
    
    # 5. Analyze Logs
    # Logs structure: {layer_name: [ {logits: ..., probs: ..., names: ...}, ... ]}
    
    for layer_name, layer_data in logs.items():
        # Usually layer_data has 1 entry corresponding to the forward pass we just ran
        if not layer_data: continue
        
        last_pass = layer_data[-1] 
        probs = last_pass['probs'] # Shape: (batch, seq_len, num_experts)
        expert_names = last_pass['expert_names']
        
        # Calculate average usage over the sequence (the prompt)
        # Squeeze batch dim if 1
        if probs.shape[0] == 1:
            probs = probs.squeeze(0) # Shape: (seq_len, num_experts)
            
        # Get the mean probability assigned to each expert across the prompt tokens
        mean_probs = probs.mean(dim=0).tolist()
        
        # Find which expert was selected most often (argmax) per token
        selected_indices = torch.argmax(probs, dim=-1)
        
        print(f"\n[Layer: {layer_name}]")
        print(f"  Experts: {expert_names}")
        
        # Print detailed values
        for i, name in enumerate(expert_names):
            usage_pct = (selected_indices == i).sum().item() / len(selected_indices) * 100
            print(f"  - {name}: Avg Prob = {mean_probs[i]:.4f} | Selected as Top-1: {usage_pct:.1f}% of tokens")

        last_token_probs = probs[-1].tolist()
        print(f"  > Last Token Gating: ", end="")
        for i, p in enumerate(last_token_probs):
            print(f"{expert_names[i]}: {p:.4f} ", end=" | ")
        print()