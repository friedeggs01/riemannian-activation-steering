#!/usr/bin/env python3
"""
Script to reformat GSM8K dataset by converting "question" field to "problem" field.
"""

import json
import os
from pathlib import Path

def reformat_gsm8k_file(input_file, output_file=None):
    """
    Reformat GSM8K file by changing "question" field to "problem".
    
    Args:
        input_file: Path to the input GSM8K JSONL file
        output_file: Path to the output file (defaults to same as input with _reformatted suffix)
    """
    if output_file is None:
        # Create output filename by adding _reformatted before extension
        input_path = Path(input_file)
        output_file = input_path.with_name(f"{input_path.stem}_reformatted{input_path.suffix}")
    
    print(f"Reformatting {input_file} to {output_file}")
    
    # Read and process the file line by line
    reformatted_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON
                data = json.loads(line)
                
                # Rename "question" to "problem" if it exists
                if "question" in data:
                    data["problem"] = data.pop("question")
                
                # Append reformatted JSON to list
                reformatted_lines.append(json.dumps(data, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"Error parsing JSON at line {line_num}, skipping")
                continue
    
    # Write reformatted lines to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in reformatted_lines:
            f.write(line + '\n')
    
    print(f"Successfully reformatted {len(reformatted_lines)} lines")
    return output_file

def main():
    # Get GSM8K test file path
    gsm8k_test_path = os.path.join('data', 'gsm8k', 'test.jsonl')
    
    # Reformat the file
    output_file = reformat_gsm8k_file(gsm8k_test_path)
    
    # Create a backup of the original file
    backup_file = gsm8k_test_path + '.backup'
    print(f"Creating backup of original file at {backup_file}")
    with open(gsm8k_test_path, 'r', encoding='utf-8') as src:
        with open(backup_file, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
    
    # Replace the original file with the reformatted one
    print(f"Replacing original file with reformatted version")
    with open(output_file, 'r', encoding='utf-8') as src:
        with open(gsm8k_test_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
    
    # Clean up the temporary file
    os.remove(output_file)
    
    print(f"Done! Original file backed up at {backup_file}")
    print(f"GSM8K file reformatted - 'question' field renamed to 'problem'")

if __name__ == "__main__":
    main()