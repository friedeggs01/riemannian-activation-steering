#!/usr/bin/env python3
"""
Debugging script to diagnose evaluation issues in the S-DAD pipeline.
"""

import os
import sys
import pickle
import re
from pathlib import Path

# Add parent directory to path to import properly
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.parser import extract_answer, strip_string, parse_ground_truth
from src.evaluation.grader import math_equal_process

def extract_candidate_answer(candidate, dataset="math"):
    """Extract answer from candidate solution with debugging info."""
    text = candidate.get("text", "")
    print(f"Original text: {text[-200:]}")  # Print the end of the text where answer likely is
    
    # Try standard extraction
    try:
        answer = extract_answer(text, data_name=dataset)
        print(f"Standard extraction result: '{answer}'")
        if answer and answer.strip():
            answer = strip_string(answer)
            print(f"After strip_string: '{answer}'")
            return answer
    except Exception as e:
        print(f"Error in standard extraction: {e}")
    
    # Try extracting from "Answer:" tag
    if "Answer:" in text or "answer:" in text:
        print("Found Answer: tag, extracting...")
        if "Answer:" in text:
            parts = text.split("Answer:")
        else:
            parts = text.split("answer:")
        
        answer_text = parts[-1].strip()
        print(f"Text after Answer: tag: '{answer_text}'")
        
        # Get first sentence
        if "." in answer_text:
            answer_text = answer_text.split(".")[0].strip()
            print(f"First sentence: '{answer_text}'")
        
        # Extract number if present
        number_match = re.search(r'-?\d*\.?\d+', answer_text)
        if number_match:
            number = number_match.group(0)
            print(f"Extracted number: '{number}'")
            return number
        
        # Strip and return
        stripped = strip_string(answer_text)
        print(f"Stripped answer: '{stripped}'")
        return stripped
    
    # Try finding boxed answer
    if "\\boxed" in text:
        print("Found \\boxed, extracting...")
        # Extract text inside \boxed{}
        boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', text)
        if boxed_matches:
            answer = boxed_matches[-1]
            print(f"Boxed answer: '{answer}'")
            answer = strip_string(answer)
            print(f"Stripped boxed answer: '{answer}'")
            return answer
    
    # Find all numbers in text
    number_matches = re.findall(r'-?\d*\.?\d+', text)
    if number_matches:
        print(f"All numbers found: {number_matches}")
        last_number = number_matches[-1]
        print(f"Using last number: '{last_number}'")
        return last_number
    
    print("No answer extracted!")
    return ""

def evaluate_candidate(candidate, ground_truth, dataset="math"):
    """Evaluate if a candidate answer is correct with detailed logging."""
    print(f"\n{'='*40}\nEVALUATING CANDIDATE\n{'='*40}")
    
    # Extract answer
    answer = extract_candidate_answer(candidate, dataset)
    print(f"Extracted answer: '{answer}'")
    
    if not answer:
        print("No answer extracted, returning False")
        return False
    
    # Parse ground truth
    try:
        print(f"Ground truth: '{ground_truth}'")
        _, gt = parse_ground_truth(
            {"problem": "", "answer": ground_truth, "solution": ""},
            data_name=dataset
        )
        print(f"Parsed ground truth: '{gt}'")
    except Exception as e:
        print(f"Error parsing ground truth: {e}, using raw ground truth")
        gt = ground_truth
    
    # Try basic string comparison
    if answer.strip() == gt.strip():
        print(f"MATCH by direct string comparison! '{answer}' == '{gt}'")
        return True
    
    # Try math_equal_process
    try:
        print(f"Trying math_equal_process with '{answer}' and '{gt}'")
        result = math_equal_process((0, answer, gt))
        if result:
            print(f"MATCH by math_equal_process!")
            return True
        else:
            print(f"No match by math_equal_process")
    except Exception as e:
        print(f"Error in math_equal_process: {e}")
    
    # Try numeric comparison
    try:
        # Check if both are numeric
        answer_val = float(answer.replace(',', ''))
        gt_val = float(gt.replace(',', ''))
        
        print(f"Comparing as numbers: {answer_val} vs {gt_val}")
        
        if abs(answer_val - gt_val) < 1e-6 or abs(answer_val - gt_val) / max(abs(gt_val), 1e-10) < 1e-4:
            print(f"MATCH by numeric comparison!")
            return True
        else:
            print(f"No match by numeric comparison")
    except ValueError:
        print("Not comparable as numbers")
    
    # Try relaxed string comparison
    answer_clean = answer.lower().strip().replace(' ', '').replace(',', '')
    gt_clean = gt.lower().strip().replace(' ', '').replace(',', '')
    
    print(f"Relaxed comparison: '{answer_clean}' vs '{gt_clean}'")
    
    if answer_clean == gt_clean:
        print(f"MATCH by relaxed string comparison!")
        return True
    
    # Try to extract and compare numbers from both
    try:
        answer_nums = re.findall(r'-?\d*\.?\d+', answer)
        gt_nums = re.findall(r'-?\d*\.?\d+', gt)
        
        print(f"Numbers in answer: {answer_nums}")
        print(f"Numbers in ground truth: {gt_nums}")
        
        if answer_nums and gt_nums and answer_nums[-1] == gt_nums[-1]:
            print(f"MATCH by comparing extracted numbers!")
            return True
    except:
        print("Error comparing numbers")
    
    print("NO MATCHES FOUND - returning False")
    return False

def debug_evaluation(result_file, num_problems=3):
    """Debug the evaluation process for the first few problems."""
    print(f"Loading data from {result_file}")
    
    with open(result_file, 'rb') as f:
        data = pickle.load(f)
    
    results = data.get("results", [])
    print(f"Found {len(results)} results")
    
    # Examine the first few problems
    for i, result in enumerate(results[:num_problems]):
        if i >= num_problems:
            break
            
        print(f"\n\n{'#'*50}\nPROBLEM {i+1}\n{'#'*50}")
        print(f"Problem: {result.get('problem', 'Not found')}")
        print(f"Ground Truth: {result.get('ground_truth', 'Not found')}")
        
        # Examine first two candidates
        candidates = result.get("candidates", [])
        print(f"Found {len(candidates)} candidates")
        
        for j, candidate in enumerate(candidates[:2]):
            print(f"\n{'*'*40}\nCandidate {j+1}\n{'*'*40}")
            is_correct = evaluate_candidate(candidate, result.get("ground_truth", ""))
            print(f"Final evaluation result: {is_correct}")

if __name__ == "__main__":
    # Check for command line arguments or use default
    if len(sys.argv) > 1:
        result_file = sys.argv[1]
    else:
        result_file = "results/sdad_20250511_192335/sdad_qwen2.5-math-1.5b-instruct_0_200_raw.pkl"
    
    if len(sys.argv) > 2:
        num_problems = int(sys.argv[2])
    else:
        num_problems = 3
        
    debug_evaluation(result_file, num_problems)