from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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

# 1. Load MBPP dataset
dataset = load_dataset("mbpp")
tasks = dataset["test"]
# tasks = dataset["test"].select(range(5))  # small subset

# 2. Load DeepSeek Coder 1.3B Base model
model_id = "deepseek-ai/deepseek-coder-1.3b-base"
# model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"

print("🔄 Loading DeepSeek model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

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

# 4. Define steering
def riemannian_block_update(h_last, T=20, alpha_k=1.0, eta_k=1.0, calpha_k=1):
    N, D = h_last.shape
    V = []
    
    if isinstance(alpha_k, (int, float)):
        alpha_k = [alpha_k] * N 
    if isinstance(eta_k, (int, float)):
        eta_k = [eta_k] * N
    
    alpha_k = [calpha_k * (np.linalg.norm(h_last[k]) / D) for k in range(len(h_last))]
    # print("alpha_k:", alpha_k)
    # print("c_alpha_k:", calpha_k, "alpha_k:", alpha_k)
    H_norm_fro = np.linalg.norm(h_last, ord='fro')
    H_bar = np.mean(h_last, axis=0)  # (D,)
    for k in range(N):
        h_k = h_last[k]
        epsilon = np.random.uniform(0, 1, size=D)  # (D,)
        vk0 = h_k - H_bar + epsilon  # perturbation
        vk0 = vk0 / np.linalg.norm(vk0) * np.sqrt(alpha_k[k])  # normalize and scale
        V.append(vk0)
    V = np.stack(V)

    losses = []
    for _ in range(T):
        for k in range(N):
            HV = h_last + V  # shape: (N, D)
            HV_T = HV.T  # shape: (D, N)
            M = np.eye(N) + HV @ HV_T  # shape: (N, N)
            M_inv = np.linalg.inv(M)
            losses.append(-np.log(np.linalg.det(M) + 1e-8))  # log-det loss
            L = 2 + 4 * (H_norm_fro + alpha_k[k])**2 + (2 / np.sqrt(alpha_k[k])) * (H_norm_fro + alpha_k[k])
            eta_k[k] = 1.0 / L
            # print(f"Iteration {_+1}, k={k}, alpha_{k}={alpha_k[k]}, eta_{k}={eta_k[k]}")
            g_k = -2 * HV_T @ M_inv[:, k]  # now g_k is (D,)
            v_k_prev = V[k]
            proj_grad = g_k - (1 / alpha_k[k]) * np.dot(v_k_prev, g_k) * v_k_prev
            d_k = eta_k[k] * proj_grad
            d_k_norm = np.linalg.norm(d_k)

            if d_k_norm != 0:
                V[k] = (
                    np.cos(d_k_norm / np.sqrt(alpha_k[k])) * v_k_prev -
                    np.sin(d_k_norm / np.sqrt(alpha_k[k])) * d_k / d_k_norm * np.sqrt(alpha_k[k])
                )
    return V, losses

def apply_steering_hook(model, tokenizer, split_id: int = 198, recalc_steer_after_n_tokens: int = 200, calpha_k: float = 0.0001):
    input_ids = None
    token_cnt = 0
    v_steering = None 
    
    def get_input_ids_hook(module, input, output):
        nonlocal input_ids
        # input_ids.shape = [batch size, token length so far]
        input_ids = torch.cat((input_ids, input[0].detach().clone()), dim=1) if input_ids is not None else input[0].detach().clone()
        # print("Appended tokens:", input_ids[0].tolist())

    def steering_hook(module, input, output):
        nonlocal token_cnt, v_steering, input_ids, tokenizer
        token_cnt += 1
        if input_ids is None or token_cnt < 0: # token_cnt = -1 to skip the first generated token
            return output 

        if isinstance(output, tuple):
            output_tensor = output[0]
            other_outputs = output[1:]
        else:
            output_tensor = output
            other_outputs = ()
            
        if (token_cnt % (recalc_steer_after_n_tokens + 50) == 0 and token_cnt > 50) or token_cnt == 50:
            print(f"Recompute steering applied at {token_cnt}%{recalc_steer_after_n_tokens} tokens")
            h_last_np = output_tensor[:, -1, :].detach().cpu().numpy().astype(np.float32)
            v_steering, _ = riemannian_block_update(h_last_np, T=20, alpha_k=1.0, eta_k=1.0, calpha_k=calpha_k)
            v_steering = torch.tensor(v_steering, dtype=output_tensor.dtype, device=output_tensor.device)
            # token_cnt = 0 
            
        if v_steering is not None:
            output_tensor[:, -1, :] += v_steering
        if isinstance(output, tuple):
            return (output_tensor, *other_outputs)
        else:
            return output_tensor

    embed_handle = model.model.model.embed_tokens.register_forward_hook(get_input_ids_hook)
    steering_handle = model.model.model.layers[14].mlp.register_forward_hook(steering_hook)
    return embed_handle, steering_handle

# 5. Main loop
seed_everything(42)
recalc_steer_after_n_tokens = 250
calpha_k = 100
for temperature in [1.0]:
    print(f"\n🔥 Sampling with temperature = {temperature}")
    start_time = time.time()
    all_task_results = []
    for task in tqdm(tasks, desc=f"🔄 Generating solutions with temp {temperature}"):
        # print(f"\n🔥 Sampling with temperature = {temperature}")
        # print("🔹 Task:", task["text"].strip())
        prompt = make_prompt(task["text"], task['test_list'])
        # Apply hook
        handle = apply_steering_hook(generator, generator.tokenizer, recalc_steer_after_n_tokens=recalc_steer_after_n_tokens, calpha_k=calpha_k)
        # Generate multiple candidate codes
        outputs = generator(
            prompt,
            max_new_tokens=1024,
            num_return_sequences=8,
            do_sample=True,
            temperature=temperature
        )
        # Clean up hook
        handle[0].remove()
        handle[1].remove()

        responses = []
        results = []

        for i, out in enumerate(outputs):
            raw_code = out["generated_text"]
            code_start = raw_code.find("def ")
            generated_code = raw_code[code_start:].strip() if code_start != -1 else raw_code.strip()

            # Align function names
            expected_name = get_func_name(task["code"])
            generated_name = get_func_name(generated_code)
            if expected_name and generated_name and expected_name != generated_name:
                generated_code = re.sub(rf'def\s+{generated_name}', f'def {expected_name}', generated_code, count=1)

            generated_code = cleanup_code(generated_code)
            responses.append(generated_code)

            # Run evaluation
            passed = evaluate(generated_code, task["test_list"])
            results.append(passed)
            # print(f"🔧 Variant {i+1} Generated code:\n{generated_code}")
            # print("✅ Passed" if passed else "❌ Failed", "\n" + "-" * 50)

        all_task_results.append({
            "problem": task["text"],
            "responses": responses,
            "results": results
        })

    # 6. Summary
    elapsed_time = time.time() - start_time
    num_passed = sum(any(res["results"]) for res in all_task_results)
    print(f"\n🎯 Final Pass@8 Accuracy: {num_passed}/{len(tasks)} = {num_passed / len(tasks):.2%}")
    all_all_task_results = {
        'pass@n': num_passed / len(tasks),
        'elapsed_time_seconds': elapsed_time,
        'all_task_results': all_task_results        
    }
    with open(f"/home/ly/DataDistillation/results/steering_code/mbpp/deepseek-coder-1.3b-base/steering_n8_{recalc_steer_after_n_tokens}_{calpha_k}_temp{temperature}.pkl", "wb") as f:
        pickle.dump(all_all_task_results, f)

# # 6. hehe
# recalc_steer_after_n_tokens = 250
# calpha_k = 1
# for temperature in [1.0]:
#     print(f"\n🔥 Sampling with temperature = {temperature}")
#     start_time = time.time()
#     all_task_results = []
#     for task in tqdm(tasks, desc=f"🔄 Generating solutions with temp {temperature}"):
#         # print(f"\n🔥 Sampling with temperature = {temperature}")
#         # print("🔹 Task:", task["text"].strip())
#         prompt = make_prompt(task["text"], task['test_list'])
#         # Apply hook
#         handle = apply_steering_hook(generator, generator.tokenizer, recalc_steer_after_n_tokens=recalc_steer_after_n_tokens, calpha_k=calpha_k)
#         # Generate multiple candidate codes
#         outputs = generator(
#             prompt,
#             max_new_tokens=1024,
#             num_return_sequences=8,
#             do_sample=True,
#             temperature=temperature
#         )
#         # Clean up hook
#         handle[0].remove()
#         handle[1].remove()

#         responses = []
#         results = []

#         for i, out in enumerate(outputs):
#             raw_code = out["generated_text"]
#             code_start = raw_code.find("def ")
#             generated_code = raw_code[code_start:].strip() if code_start != -1 else raw_code.strip()

#             # Align function names
#             expected_name = get_func_name(task["code"])
#             generated_name = get_func_name(generated_code)
#             if expected_name and generated_name and expected_name != generated_name:
#                 generated_code = re.sub(rf'def\s+{generated_name}', f'def {expected_name}', generated_code, count=1)

#             generated_code = cleanup_code(generated_code)
#             responses.append(generated_code)

#             # Run evaluation
#             passed = evaluate(generated_code, task["test_list"])
#             results.append(passed)
#             # print(f"🔧 Variant {i+1} Generated code:\n{generated_code}")
#             # print("✅ Passed" if passed else "❌ Failed", "\n" + "-" * 50)

#         all_task_results.append({
#             "problem": task["text"],
#             "responses": responses,
#             "results": results
#         })

#     # 6. Summary
#     elapsed_time = time.time() - start_time
#     num_passed = sum(any(res["results"]) for res in all_task_results)
#     print(f"\n🎯 Final Pass@8 Accuracy: {num_passed}/{len(tasks)} = {num_passed / len(tasks):.2%}")
#     all_all_task_results = {
#         'pass@n': num_passed / len(tasks),
#         'elapsed_time_seconds': elapsed_time,
#         'all_task_results': all_task_results        
#     }
#     with open(f"/home/ly/DataDistillation/results/steering_code/mbpp/deepseek-coder-1.3b-base/steering_n8_{recalc_steer_after_n_tokens}_{calpha_k}_temp{temperature}.pkl", "wb") as f:
#         pickle.dump(all_all_task_results, f)
