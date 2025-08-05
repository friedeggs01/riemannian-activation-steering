from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map
from datasets import load_dataset
import torch
import traceback
import re
import pickle
from tqdm import tqdm
import time
import random
import numpy as np
import math
import builtins
import tempfile
import subprocess
import os
from torch import autocast

# 1. Load MBPP dataset
dataset = load_dataset("mbpp")
tasks = dataset["test"]
tasks = dataset["test"].select(range(10))  # small subset

# 2. Load DeepSeek Coder 1.3B Base model
model_id = "deepseek-ai/deepseek-coder-1.3b-base"

print("🔄 Loading DeepSeek model...")
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

config = AutoConfig.from_pretrained(model_id)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

device_map = infer_auto_device_map(model, max_memory={0: "23.5GiB"})

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cleanup_code(
    code: str,
    language_type: str = "python",
    dataset: str = None,
    issft: bool = False,
    stop_words = []
):
    """
    Cleans up the generated code.
    """

    if language_type.lower() == "python":
        if issft:
            code = _clean_python_code_for_sft(code)
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"])
    else:
        code = _truncate_code_at_stopwords(code, stop_words)
    code = code.split("[END]")[0].strip()
    return code

def _clean_python_code_for_sft(code):
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.index("```python")
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code

def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]

# 1. Prompt template (no [INST] format)
def make_prompt(task_prompt: str, tests):
    return f"You are an expert Python programmer, and here is your task: {task_prompt} Your code should pass these tests:\n\n{tests}\n[BEGIN]"

# 2. Helper to extract function name
def get_func_name(code: str):
    match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
    return match.group(1) if match else None

# 3. Evaluator
def evaluate(code: str, test_cases: list[str], timeout: float = 10.0) -> bool:
    """
    Evaluate Python code with given test cases in a temporary script using subprocess.
    Returns True if all tests pass, otherwise False.
    """
    try:
        # Tạo file tạm
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            # Ghi code + test cases vào file
            temp_file.write(code)
            temp_file.write("\n\n")
            for test in test_cases:
                temp_file.write(test + "\n")
            temp_file_path = temp_file.name

        # Chạy file bằng subprocess
        result = subprocess.run(
            ["python", temp_file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        return False

    except Exception as e:
        return False

    finally:
        # Xoá file tạm
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
# 4. Main loop
seed_everything(42)
batch_size = 5  
num_tasks = len(tasks)
num_batches = math.ceil(num_tasks / batch_size)
for temperature in [1.0]:
    print(f"\n🔥 Sampling with temperature = {temperature}")
    start_time = time.time()
    all_task_results = []
    for batch_idx in tqdm(range(num_batches), desc=f"🔄 Generating solutions in batch with T={temperature}"):
        # print(f"\n🔥 Sampling with temperature = {temperature}")
        batch_tasks = tasks[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        prompts = [make_prompt(task, list_test) for task, list_test in zip(batch_tasks['text'], batch_tasks['test_list'])]

        # # Generate in batch
        # with torch.inference_mode():
        #     batch_outputs = generator(
        #         prompts,
        #         max_new_tokens=1024,
        #         num_return_sequences=8,
        #         do_sample=True,
        #         temperature=temperature,
        #         return_full_text=True,
        #     )
        # Tokenize prompt
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        ).to(model.device)

        # Generate with acceleration
        with torch.inference_mode(), autocast(device_type='cuda', dtype=torch.float16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                num_return_sequences=8,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and regroup by prompts
        batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Nếu generator trả về list of list (nhiều outputs cho từng prompt)
        if isinstance(batch_outputs[0], list):
            grouped_outputs = batch_outputs
        else:
            # Một số model trả ra flat list — cần chia lại theo num_return_sequences
            grouped_outputs = [
                batch_outputs[i * 8:(i + 1) * 8] for i in range(len(batch_tasks['task_id']))
            ]

        for text, code, test_list, outputs in zip(batch_tasks["text"], batch_tasks["code"], batch_tasks["test_list"], grouped_outputs):
            responses = []
            results = []

            for out in outputs:
                # breakpoint()
                raw_code = out
                code_start = raw_code.find("def ")
                generated_code = raw_code[code_start:].strip() if code_start != -1 else raw_code.strip()

                # Align function names
                # breakpoint()
                expected_name = get_func_name(code)
                generated_name = get_func_name(generated_code)
                if expected_name and generated_name and expected_name != generated_name:
                    generated_code = re.sub(rf'def\s+{generated_name}', f'def {expected_name}', generated_code, count=1)

                generated_code = cleanup_code(generated_code)
                responses.append(generated_code)

                # Evaluate
                passed = evaluate(generated_code, test_list)
                results.append(passed)

            all_task_results.append({
                "problem": text,
                "responses": responses,
                "results": results
            })
               

    # 5. Summary
    elapsed_time = time.time() - start_time
    num_passed = sum(any(res["results"]) for res in all_task_results)
    print(f"\n🎯 Final Pass@8 Accuracy: {num_passed}/{len(tasks)} = {num_passed / len(tasks):.2%}")
    all_all_task_results = {
        'pass@n': num_passed / len(tasks),
        'elapsed_time_seconds': elapsed_time,
        'all_task_results': all_task_results        
    }
    with open(f"/home/ly/DataDistillation/results/steering_code/mbpp/deepseek-coder-1.3b-base/sampling_temp{temperature}.pkl", "wb") as f:
        pickle.dump(all_all_task_results, f)

