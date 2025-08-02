#!/usr/bin/env python3
"""
Evaluation script for S-DAD (Shapley-DPP Adaptive Decoding) algorithm.

This script evaluates the results of the S-DAD algorithm, including:
- Calculating pass@k metrics for different k values
- Comparing with baselines (greedy, temperature sampling, beam search, nucleus)
- Visualizing the solution tree and DPP kernel properties
"""

import os
import re
import sys
import argparse
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Union

# Add parent directory to path to import properly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.parser import extract_answer, strip_string, parse_ground_truth
from src.evaluation.grader import math_equal_process
# from src.evaluation.utils import load_answers

def parse_args():
    """Parse command-line arguments for S-DAD evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate S-DAD results for math problem solving")
    
    # Input options
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file with S-DAD results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results"
    )
    
    # Baseline comparison options
    parser.add_argument(
        "--compare_baselines",
        action="store_true",
        help="Compare with baseline methods"
    )
    parser.add_argument(
        "--baseline_greedy",
        type=str,
        default=None,
        help="Input file for greedy baseline"
    )
    parser.add_argument(
        "--baseline_temperature",
        type=str,
        default=None,
        help="Input file for temperature sampling baseline"
    )
    parser.add_argument(
        "--baseline_beam",
        type=str,
        default=None,
        help="Input file for beam search baseline"
    )
    parser.add_argument(
        "--baseline_nucleus",
        type=str,
        default=None,
        help="Input file for nucleus sampling baseline"
    )
    
    # Evaluation options
    parser.add_argument(
        "--k_values",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated list of k values for pass@k evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math",
        choices=["math", "gsm8k"],
        help="Dataset being evaluated"
    )
    
    # Output options
    parser.add_argument(
        "--save_table",
        action="store_true",
        help="Save evaluation table to CSV file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate evaluation plots"
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Directory for plots"
    )
    
    return parser.parse_args()

def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load results from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Dictionary with results data
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different formats
    if isinstance(data, dict) and "results" in data:
        # Standard format with metadata and results
        return data
    elif isinstance(data, list):
        # List of results with no metadata
        return {"results": data, "metadata": {}}
    else:
        # Unknown format
        raise ValueError(f"Unknown data format in {file_path}")

def extract_candidate_answer(candidate: Dict[str, Any], dataset: str = "math") -> str:
    """
    Extract the answer from a candidate solution.

    Args:
        candidate: Candidate solution dictionary
        dataset: Dataset name for extraction logic

    Returns:
        Extracted answer string
    """
    text = candidate.get("text", "")

    try:
        # First try to extract with the standard function
        answer = extract_answer(text, data_name=dataset)
        if answer and answer.strip():
            return strip_string(answer)

        # If that didn't work, try more aggressive methods for math problems
        if "Answer:" in text or "answer:" in text:
            # Find answer after "Answer:" tag
            if "Answer:" in text:
                parts = text.split("Answer:")
            else:
                parts = text.split("answer:")

            answer_text = parts[-1].strip()

            # Get the first sentence of the answer
            if "." in answer_text:
                answer_text = answer_text.split(".")[0].strip()

            # Take only the final number if it seems to be a calculated result
            number_match = re.search(r'-?\d*\.?\d+', answer_text)
            if number_match:
                return number_match.group(0)

            return strip_string(answer_text)

        # Try to find the last number in the text as a last resort
        number_matches = re.findall(r'-?\d*\.?\d+', text)
        if number_matches:
            return number_matches[-1]

        return ""
    except Exception as e:
        print(f"Error extracting answer: {e}")
        # Last resort fallback
        return text.strip()

def evaluate_candidate(candidate: Dict[str, Any], ground_truth: str, dataset: str = "math") -> bool:
    """
    Evaluate if a candidate answer is correct.

    Args:
        candidate: Candidate solution
        ground_truth: Ground truth answer
        dataset: Dataset name

    Returns:
        True if the candidate is correct, False otherwise
    """
    # Extract answer from candidate
    answer = extract_candidate_answer(candidate, dataset)

    if not answer:
        return False

    try:
        # Parse ground truth
        try:
            _, gt = parse_ground_truth(
                {"problem": "", "answer": ground_truth, "solution": ""},
                data_name=dataset
            )

            # Critical fix: If parse_ground_truth returns empty string, use the original ground_truth
            if not gt or gt.strip() == '':
                print(f"Warning: parse_ground_truth returned empty string, using original ground truth")
                gt = ground_truth
        except Exception as e:
            print(f"Warning: Error parsing ground truth: {e}")
            gt = ground_truth

        # Try direct string comparison first
        if answer.strip() == gt.strip():
            return True

        # Try with math_equal_process
        try:
            result = math_equal_process((0, answer, gt))
            if result:
                return True
        except Exception as e:
            print(f"Warning: Error in math_equal_process: {e}")

        # Try numeric comparison for numbers
        try:
            # Check if both are numeric values
            answer_val = float(answer.replace(',', ''))
            gt_val = float(gt.replace(',', ''))

            # Check if they're close enough
            return abs(answer_val - gt_val) < 1e-6 or abs(answer_val - gt_val) / max(abs(gt_val), 1e-10) < 1e-4
        except ValueError:
            # Not numeric values
            pass

        # Try relaxed string comparison
        answer_clean = answer.lower().strip().replace(' ', '').replace(',', '')
        gt_clean = gt.lower().strip().replace(' ', '').replace(',', '')
        if answer_clean == gt_clean:
            return True

        return False
    except Exception as e:
        print(f"Error evaluating answer: {e}")

        # Last resort: try direct numeric comparison
        try:
            # Try to extract numbers from both strings
            import re
            answer_nums = re.findall(r'-?\d*\.?\d+', answer)
            gt_nums = re.findall(r'-?\d*\.?\d+', ground_truth)

            if answer_nums and gt_nums:
                return answer_nums[-1] == gt_nums[-1]
        except:
            pass

        return False

def calculate_empirical_pass_at_k(candidates: List[Dict[str, Any]], 
                                 ground_truth: str, 
                                 k: int, 
                                 dataset: str = "math") -> float:
    """
    Calculate empirical pass@k by checking if any of the top k candidates have the correct answer.
    
    Args:
        candidates: List of candidate solutions
        ground_truth: Ground truth answer
        k: Value of k (number of candidates to consider)
        dataset: Dataset name
        
    Returns:
        1.0 if any of the top k candidates are correct, 0.0 otherwise
    """
    if not candidates:
        return 0.0
    
    # Take the top k candidates
    top_k = candidates[:min(k, len(candidates))]
    
    # Check if any are correct
    for candidate in top_k:
        if evaluate_candidate(candidate, ground_truth, dataset):
            return 1.0
    
    return 0.0

def calculate_theoretical_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate theoretical pass@k given the number of candidates and correct answers.
    
    Args:
        n: Total number of candidates
        c: Number of correct candidates
        k: Value of k
        
    Returns:
        Theoretical pass@k value
    """
    from math import comb
    
    if c == 0:
        return 0.0
    
    if k >= n:
        return 1.0 if c > 0 else 0.0
    
    # Calculate 1 - P(no correct answers in k draws)
    # This is: 1 - (n-c choose k) / (n choose k)
    if n - c < k:
        return 1.0  # Can't pick k without getting at least one correct
    
    p_no_correct = comb(n - c, k) / comb(n, k)
    return 1.0 - p_no_correct

def evaluate_sdad_results(results: Dict[str, Any], 
                          k_values: List[int], 
                          dataset: str = "math") -> Dict[str, Any]:
    """
    Evaluate S-DAD results for the given k values.
    
    Args:
        results: Dictionary with S-DAD results
        k_values: List of k values for pass@k evaluation
        dataset: Dataset name
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        "num_problems": 0,
        "num_solved_greedy": 0,  # Problems solved by the greedy baseline
        "problems_with_shy_tokens": 0,  # Problems where shy tokens were found
        "num_candidates_per_problem": [],  # Number of candidates generated per problem
        "pass_at_k": {k: [] for k in k_values},  # Pass@k for each problem
        "correct_candidates": [],  # List indicating whether each candidate is correct
    }
    
    # Process each problem
    for problem_result in tqdm(results["results"], desc="Evaluating problems"):
        metrics["num_problems"] += 1
        
        ground_truth = problem_result.get("ground_truth", "")
        candidates = problem_result.get("candidates", [])
        shy_tokens_found = problem_result.get("shy_tokens_found", False)
        
        if shy_tokens_found:
            metrics["problems_with_shy_tokens"] += 1
        
        # Count candidates
        metrics["num_candidates_per_problem"].append(len(candidates))
        
        # Check if greedy solution is correct
        if candidates and candidates[0].get("source") == "greedy":
            greedy_correct = evaluate_candidate(candidates[0], ground_truth, dataset)
            if greedy_correct:
                metrics["num_solved_greedy"] += 1
        
        # Mark each candidate as correct or not
        for candidate in candidates:
            is_correct = evaluate_candidate(candidate, ground_truth, dataset)
            candidate["is_correct"] = is_correct
            metrics["correct_candidates"].append(is_correct)
        
        # Calculate pass@k
        for k in k_values:
            pass_at_k = calculate_empirical_pass_at_k(candidates, ground_truth, k, dataset)
            metrics["pass_at_k"][k].append(pass_at_k)
    
    # Calculate summary metrics
    metrics["mean_candidates_per_problem"] = np.mean(metrics["num_candidates_per_problem"])
    metrics["median_candidates_per_problem"] = np.median(metrics["num_candidates_per_problem"])
    metrics["greedy_accuracy"] = metrics["num_solved_greedy"] / metrics["num_problems"]
    metrics["problems_with_shy_tokens_pct"] = metrics["problems_with_shy_tokens"] / metrics["num_problems"]
    
    # Calculate mean pass@k
    for k in k_values:
        metrics["pass_at_k_mean"] = {k: np.mean(metrics["pass_at_k"][k]) for k in k_values}
    
    # Calculate overall metrics
    metrics["total_correct_candidates"] = sum(metrics["correct_candidates"])
    metrics["total_candidates"] = len(metrics["correct_candidates"])
    metrics["overall_accuracy"] = metrics["total_correct_candidates"] / metrics["total_candidates"] if metrics["total_candidates"] > 0 else 0.0
    
    return metrics

def load_and_evaluate_baseline(file_path: str, 
                              k_values: List[int], 
                              dataset: str = "math") -> Dict[str, Any]:
    """
    Load and evaluate baseline results.
    
    Args:
        file_path: Path to the baseline results file
        k_values: List of k values for pass@k evaluation
        dataset: Dataset name
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load baseline results
    try:
        baseline_data = load_results(file_path)

        # Extract baseline type from metadata
        metadata = baseline_data.get("metadata", {})
        method = metadata.get("method", "unknown")

        # Normalize method name
        method_name = "unknown"
        
        # Use more specific pattern matching to identify method type
        if "greedy" in method.lower() or "greedy" in file_path.lower():
            method_name = "greedy"
        # Check for nucleus/top_p first (since it might also contain "temp" keyword)
        elif "nucleus" in method.lower() or "top_p" in file_path.lower() or "topp" in file_path.lower():
            method_name = "nucleus"
        elif "beam" in method.lower() or "beam" in file_path.lower():
            method_name = "beam"
        elif "temperature" in method.lower() or "temp" in file_path.lower():
            method_name = "temperature"
            
        # Check the file path for additional detection
        if method_name == "unknown" or method_name == "temperature":
            # More aggressive pattern matching from filename
            basename = os.path.basename(file_path)
            if "nucleus" in basename or "topp" in basename:
                method_name = "nucleus"
            elif "beam" in basename:
                method_name = "beam"

        # Check if the method failed (for graceful handling)
        if "failed" in method.lower():
            print(f"WARNING: {method_name} baseline failed during execution. Using empty results.")
            # Return dummy metrics
            return {
                "method": method_name,
                "metadata": metadata,
                "num_problems": 0,
                "pass_at_k": {k: [] for k in k_values},
                "pass_at_k_mean": {k: 0.0 for k in k_values},
                "status": "failed"
            }

        # Calculate pass@k for baseline
        results = baseline_data.get("results", [])
        if not results:
            print(f"WARNING: No results found in {method_name} baseline. Using empty results.")
            # Return dummy metrics
            return {
                "method": method_name,
                "metadata": metadata,
                "num_problems": 0,
                "pass_at_k": {k: [] for k in k_values},
                "pass_at_k_mean": {k: 0.0 for k in k_values},
                "status": "empty"
            }

        # Evaluate baseline results
        metrics = {"method": method_name, "metadata": metadata, "status": "success"}

        # Process each problem
        metrics["num_problems"] = len(results)
    except Exception as e:
        print(f"ERROR: Failed to load or process {file_path}: {e}")
        # Return dummy metrics for graceful failure
        return {
            "method": os.path.basename(file_path).split('_')[0],
            "metadata": {"error": str(e)},
            "num_problems": 0,
            "pass_at_k": {k: [] for k in k_values},
            "pass_at_k_mean": {k: 0.0 for k in k_values},
            "status": "error"
        }
    metrics["pass_at_k"] = {k: [] for k in k_values}
    
    for problem_result in tqdm(results, desc=f"Evaluating {method_name} baseline"):
        ground_truth = problem_result.get("ground_truth", "")
        
        # Extract candidates based on baseline type
        candidates = []

        # Try multiple possible formats of results
        if method_name == "greedy":
            # For greedy, use the single response
            response = problem_result.get("response", problem_result.get("answer", ""))
            if isinstance(response, list) and response:
                response = response[0]
            candidates = [{"text": response}]
        else:
            # For multi-response methods, check all possible fields
            responses = []

            # Check various field names that might contain responses
            for field_name in ["response", "responses", "answer", "answers", "branches", "generations"]:
                if field_name in problem_result:
                    field_value = problem_result[field_name]
                    if field_value:
                        responses = field_value
                        break

            # Handle branches specially (typical in S-DAD and branching methods)
            branches = problem_result.get("branches", [])
            if branches:
                # For branching methods
                candidates = []
                for b in branches:
                    if isinstance(b, dict):
                        if "text" in b:
                            candidates.append({"text": b["text"]})
                        else:
                            # Use first string value found in the dict
                            for val in b.values():
                                if isinstance(val, str):
                                    candidates.append({"text": val})
                                    break
                    else:
                        candidates.append({"text": b})
            elif responses:
                # Handle various formats of responses
                if isinstance(responses, list):
                    # List of responses
                    if responses and isinstance(responses[0], list):
                        # Nested list, flatten one level
                        flattened = []
                        for resp_group in responses:
                            if isinstance(resp_group, list):
                                flattened.extend(resp_group)
                            else:
                                flattened.append(resp_group)
                        responses = flattened

                    # Create candidate objects
                    candidates = []
                    for r in responses:
                        if isinstance(r, dict):
                            if "text" in r:
                                candidates.append({"text": r["text"]})
                            else:
                                # Use the first string value in the dict
                                for val in r.values():
                                    if isinstance(val, str):
                                        candidates.append({"text": val})
                                        break
                        else:
                            candidates.append({"text": r})
                elif isinstance(responses, str):
                    # Single string response
                    candidates = [{"text": responses}]
                elif isinstance(responses, dict):
                    # Dictionary with responses
                    for key, val in responses.items():
                        if isinstance(val, str):
                            candidates.append({"text": val})

            # If still no candidates, try to extract directly from problem_result
            if not candidates:
                for key, val in problem_result.items():
                    if isinstance(val, str) and key not in ["problem", "ground_truth", "question"]:
                        candidates.append({"text": val})
                        break

        # Ensure we have at least one candidate
        if not candidates:
            candidates = [{"text": ""}]
        
        # Calculate pass@k
        for k in k_values:
            pass_at_k = calculate_empirical_pass_at_k(candidates, ground_truth, k, dataset)
            metrics["pass_at_k"][k].append(pass_at_k)
    
    # Calculate mean pass@k
    metrics["pass_at_k_mean"] = {k: np.mean(metrics["pass_at_k"][k]) for k in k_values}
    
    return metrics

def plot_pass_at_k(metrics_list: List[Dict[str, Any]], 
                  k_values: List[int], 
                  output_dir: str,
                  method_names: Dict[str, str] = None):
    """
    Generate a pass@k comparison plot.
    
    Args:
        metrics_list: List of metrics dictionaries for different methods
        k_values: List of k values used in the metrics
        output_dir: Output directory for the plot
        method_names: Dictionary mapping method IDs to display names
    """
    plt.figure(figsize=(10, 6))
    
    # Set up colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Plot lines for each method
    for i, metrics in enumerate(metrics_list):
        method = metrics.get("method", f"Method {i+1}")
        
        # Use custom name if provided
        if method_names and method in method_names:
            method_label = method_names[method]
        else:
            method_label = method
        
        # Get pass@k values
        pass_at_k_values = [metrics["pass_at_k_mean"][k] for k in k_values]
        
        # Plot line
        plt.plot(k_values, pass_at_k_values, marker='o', label=method_label, color=colors[i % len(colors)])
    
    # Customize plot
    plt.xscale('log')
    plt.title("Pass@k Comparison", fontsize=14)
    plt.xlabel("k", fontsize=12)
    plt.ylabel("Pass@k", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right')
    
    # Add value labels at the right edge
    for i, metrics in enumerate(metrics_list):
        method = metrics.get("method", f"Method {i+1}")
        pass_at_k_values = [metrics["pass_at_k_mean"][k] for k in k_values]
        plt.text(k_values[-1] * 1.05, pass_at_k_values[-1], 
                f"{pass_at_k_values[-1]:.3f}", 
                color=colors[i % len(colors)],
                verticalalignment='center')
    
    # Set x-axis ticks to k values
    plt.xticks(k_values, [str(k) for k in k_values])
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "pass_at_k_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Pass@k comparison plot saved to {plot_path}")

def plot_token_efficiency(metrics_list: List[Dict[str, Any]], 
                         output_dir: str,
                         method_names: Dict[str, str] = None):
    """
    Generate a token efficiency comparison plot.
    
    Args:
        metrics_list: List of metrics dictionaries for different methods
        output_dir: Output directory for the plot
        method_names: Dictionary mapping method IDs to display names
    """
    # Only include methods with token counts
    valid_metrics = []
    for metrics in metrics_list:
        if "total_tokens" in metrics and "num_problems_solved" in metrics:
            valid_metrics.append(metrics)
    
    if not valid_metrics:
        print("No token efficiency data available for plotting")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Set up colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Prepare data
    methods = []
    tokens_per_problem = []
    
    for i, metrics in enumerate(valid_metrics):
        method = metrics.get("method", f"Method {i+1}")
        
        # Use custom name if provided
        if method_names and method in method_names:
            method_label = method_names[method]
        else:
            method_label = method
        
        # Calculate tokens per solved problem
        tokens = metrics["total_tokens"]
        solved = metrics["num_problems_solved"]
        if solved > 0:
            efficiency = tokens / solved
            
            methods.append(method_label)
            tokens_per_problem.append(efficiency)
    
    # Sort by efficiency (lower is better)
    sorted_indices = np.argsort(tokens_per_problem)
    methods = [methods[i] for i in sorted_indices]
    tokens_per_problem = [tokens_per_problem[i] for i in sorted_indices]
    
    # Plot bar chart
    bars = plt.barh(methods, tokens_per_problem, color=colors[:len(methods)])
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width * 1.01, bar.get_y() + bar.get_height()/2, 
                f"{width:.0f}", 
                ha='left', va='center')
    
    # Customize plot
    plt.title("Token Efficiency Comparison", fontsize=14)
    plt.xlabel("Tokens Per Solved Problem (lower is better)", fontsize=12)
    plt.grid(alpha=0.3, axis='x')
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "token_efficiency_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Token efficiency comparison plot saved to {plot_path}")

def generate_summary_table(metrics_list: List[Dict[str, Any]], 
                          k_values: List[int], 
                          dataset: str,
                          method_names: Dict[str, str] = None) -> pd.DataFrame:
    """
    Generate a summary table of evaluation metrics.
    
    Args:
        metrics_list: List of metrics dictionaries for different methods
        k_values: List of k values used in the metrics
        dataset: Dataset name
        method_names: Dictionary mapping method IDs to display names
        
    Returns:
        DataFrame with summarized metrics
    """
    # Prepare data for the table
    data = []
    
    for metrics in metrics_list:
        method = metrics.get("method", "unknown")
        
        # Use custom display name if provided
        if method_names and method in method_names:
            display_method = method_names[method]
        else:
            display_method = method
        
        # Create row for this method
        row = {"Method": display_method}
        
        # Add pass@k metrics
        if "pass_at_k_mean" in metrics:
            for k in k_values:
                if k in metrics["pass_at_k_mean"]:
                    row[f"Pass@{k}"] = metrics["pass_at_k_mean"][k]
        
        # Add accuracy if available
        if "greedy_accuracy" in metrics:
            row["Greedy Accuracy"] = metrics["greedy_accuracy"]
        
        # Add token efficiency if available
        if "total_tokens" in metrics and "num_problems_solved" in metrics:
            tokens = metrics["total_tokens"]
            solved = metrics["num_problems_solved"]
            if solved > 0:
                row["Tokens/Solved"] = tokens / solved
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns
    columns = ["Method"]
    for k in k_values:
        columns.append(f"Pass@{k}")
    if "Greedy Accuracy" in df.columns:
        columns.append("Greedy Accuracy")
    if "Tokens/Solved" in df.columns:
        columns.append("Tokens/Solved")
    
    df = df[columns]
    
    # Format values as percentages
    for col in df.columns:
        if col != "Method" and col != "Tokens/Solved":
            df[col] = df[col].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A")
    
    # Format token efficiency as integers
    if "Tokens/Solved" in df.columns:
        df["Tokens/Solved"] = df["Tokens/Solved"].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "N/A")
    
    return df

def main():
    """Main function for S-DAD evaluation."""
    args = parse_args()
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Handle output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle plot directory
    if args.plot_dir is None:
        args.plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(args.plot_dir, exist_ok=True)
    
    print(f"Loading S-DAD results from {args.input}...")
    sdad_data = load_results(args.input)
    
    print(f"Evaluating S-DAD results for {len(k_values)} k values: {k_values}...")
    sdad_metrics = evaluate_sdad_results(sdad_data, k_values, args.dataset)
    sdad_metrics["method"] = "sdad"
    
    # Store all metrics for comparison
    all_metrics = [sdad_metrics]
    
    # Load and evaluate baselines if requested
    if args.compare_baselines:
        # Look for baselines if not explicitly provided
        if not args.baseline_greedy:
            # Try to find greedy baseline in same directory
            parent_dir = os.path.dirname(args.input)
            potential_files = [
                os.path.join(parent_dir, "../baseline_greedy/baseline_greedy_raw.pkl"),
                os.path.join(parent_dir, "../baseline/baseline_greedy_raw.pkl")
            ]
            for file in potential_files:
                if os.path.exists(file):
                    args.baseline_greedy = file
                    break
        
        if not args.baseline_temperature:
            # Try to find temperature baseline in same directory
            parent_dir = os.path.dirname(args.input)
            potential_files = [
                os.path.join(parent_dir, "../baseline_temp/baseline_temp_multi_raw.pkl"),
                os.path.join(parent_dir, "../baseline_temperature/baseline_temperature_raw.pkl")
            ]
            for file in potential_files:
                if os.path.exists(file):
                    args.baseline_temperature = file
                    break
        
        if not args.baseline_beam:
            # Try to find beam search baseline in same directory
            parent_dir = os.path.dirname(args.input)
            potential_files = [
                os.path.join(parent_dir, "../baseline_beam/baseline_beam_width5_raw.pkl"),
                os.path.join(parent_dir, "../baseline_beam/baseline_beam_raw.pkl")
            ]
            for file in potential_files:
                if os.path.exists(file):
                    args.baseline_beam = file
                    break
        
        if not args.baseline_nucleus:
            # Try to find nucleus sampling baseline in same directory
            parent_dir = os.path.dirname(args.input)
            potential_files = [
                os.path.join(parent_dir, "../baseline_nucleus/baseline_nucleus_topp095_temp10_raw.pkl"),
                os.path.join(parent_dir, "../baseline_nucleus/baseline_nucleus_raw.pkl")
            ]
            for file in potential_files:
                if os.path.exists(file):
                    args.baseline_nucleus = file
                    break
        
        # Evaluate greedy baseline
        if args.baseline_greedy and os.path.exists(args.baseline_greedy):
            print(f"Evaluating greedy baseline from {args.baseline_greedy}...")
            greedy_metrics = load_and_evaluate_baseline(args.baseline_greedy, k_values, args.dataset)
            all_metrics.append(greedy_metrics)
        
        # Evaluate temperature baseline
        if args.baseline_temperature and os.path.exists(args.baseline_temperature):
            print(f"Evaluating temperature baseline from {args.baseline_temperature}...")
            temperature_metrics = load_and_evaluate_baseline(args.baseline_temperature, k_values, args.dataset)
            all_metrics.append(temperature_metrics)
        
        # Evaluate beam search baseline
        if args.baseline_beam and os.path.exists(args.baseline_beam):
            print(f"Evaluating beam search baseline from {args.baseline_beam}...")
            beam_metrics = load_and_evaluate_baseline(args.baseline_beam, k_values, args.dataset)
            all_metrics.append(beam_metrics)
        
        # Evaluate nucleus sampling baseline
        if args.baseline_nucleus and os.path.exists(args.baseline_nucleus):
            print(f"Evaluating nucleus sampling baseline from {args.baseline_nucleus}...")
            nucleus_metrics = load_and_evaluate_baseline(args.baseline_nucleus, k_values, args.dataset)
            all_metrics.append(nucleus_metrics)
    
    # Create method name mapping for better labels
    method_names = {
        "sdad": "S-DAD",
        "greedy": "Greedy",
        "temperature": "Temperature (T=1.0)",
        "beam": "Beam Search",
        "nucleus": "Nucleus (Top-p=0.95)"
    }

    # Enhance method names with details from metadata if available
    for metric in all_metrics:
        method = metric.get("method")
        metadata = metric.get("metadata", {})

        if method == "temperature" and "temperature" in metadata:
            method_names[method] = f"Temperature (T={metadata['temperature']})"
        elif method == "nucleus" and "top_p" in metadata:
            top_p = metadata.get("top_p", 0.95)
            temperature = metadata.get("temperature", 1.0)
            method_names[method] = f"Nucleus (p={top_p}, T={temperature})"
        elif method == "beam" and "num_beams" in metadata:
            method_names[method] = f"Beam Search (width={metadata['num_beams']})"

    # Generate summary table
    print("Generating summary table...")
    summary_table = generate_summary_table(all_metrics, k_values, args.dataset, method_names)
    print(summary_table)
    
    # Save summary table if requested
    if args.save_table:
        table_path = os.path.join(args.output_dir, "evaluation_summary.csv")
        summary_table.to_csv(table_path, index=False)
        print(f"Summary table saved to {table_path}")
        
        # Also save as markdown for readability
        md_path = os.path.join(args.output_dir, "evaluation_summary.md")
        with open(md_path, 'w') as f:
            f.write(f"# Evaluation Summary for {args.dataset.upper()}\n\n")
            f.write(summary_table.to_markdown(index=False))
        print(f"Markdown summary saved to {md_path}")
    
    # Generate plots if requested
    if args.plot:
        # Plot pass@k comparison
        print("Generating pass@k comparison plot...")
        plot_pass_at_k(all_metrics, k_values, args.plot_dir, method_names)
        
        # Plot token efficiency comparison
        print("Generating token efficiency comparison plot...")
        plot_token_efficiency(all_metrics, args.plot_dir, method_names)
    
    # Save detailed S-DAD metrics
    detailed_path = os.path.join(args.output_dir, "sdad_evaluation_details.json")
    with open(detailed_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        detailed_metrics = {}
        for key, value in sdad_metrics.items():
            if isinstance(value, dict):
                detailed_metrics[key] = {k: float(v) if isinstance(v, np.ndarray) or isinstance(v, np.number) else v 
                                         for k, v in value.items()}
            elif isinstance(value, np.ndarray) or isinstance(value, np.number):
                detailed_metrics[key] = float(value)
            else:
                detailed_metrics[key] = value
        
        json.dump(detailed_metrics, f, indent=2)
    
    print(f"Detailed S-DAD metrics saved to {detailed_path}")
    
    # Print S-DAD summary
    print("\n===== S-DAD EVALUATION SUMMARY =====")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Number of problems: {sdad_metrics['num_problems']}")
    print(f"Problems with shy tokens: {sdad_metrics['problems_with_shy_tokens']} ({sdad_metrics['problems_with_shy_tokens_pct']:.1%})")
    print(f"Mean candidates per problem: {sdad_metrics['mean_candidates_per_problem']:.1f}")
    
    print("\nPass@k metrics:")
    for k in k_values:
        print(f"  Pass@{k}: {sdad_metrics['pass_at_k_mean'][k]:.1%}")
    
    print("\nTo analyze individual problems, load the S-DAD results file and examine the 'candidates' field.")

if __name__ == "__main__":
    main()