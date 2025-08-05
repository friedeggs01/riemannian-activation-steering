# SDAD vs Best-of-N Comparison

This document describes how to run the SDAD vs Best-of-N comparison experiments and generate pass@k comparison plots for MATH500 and GSM8K datasets.

## Experiment Overview

The experiment compares two decoding strategies:

1. **SDAD (Shapley-DPP Adaptive Decoding)**: A decoding strategy that uses uncertainty detection, branching, and Shapley-weighted DPP to generate diverse candidates.

2. **Best-of-N**: A simple approach that generates N candidate answers using temperature sampling and selects the best one using majority voting.

## Efficient Approach

The experiment can be run in two modes:

- **Standard Mode**: Runs Best-of-N for each value of N separately (uses `run_sdad_bestofn_experiment.sh`)
- **Efficient Mode**: Runs Best-of-N only once with N=8, then simulates smaller N values by using subsets of the candidates (uses `run_sdad_bestofn_efficient.sh`)

The efficient approach is recommended as it significantly reduces computation time.

## Running the Experiment

### Step 1: Ensure Data is Properly Formatted

Make sure the GSM8K dataset has the "problem" field (not "question"):

```bash
python reformat_gsm8k.py
```

### Step 2: Run the Experiment Script

To run the experiment using the efficient approach:

```bash
chmod +x experiments/temperature_tuning/run_sdad_bestofn_efficient.sh
./experiments/temperature_tuning/run_sdad_bestofn_efficient.sh
```

```bash
chmod +x experiments/temperature_tuning/run_sdad_change_temp.sh
./experiments/temperature_tuning/run_sdad_change_temp.sh
```

```bash
chmod +x experiments/temperature_tuning/run_steering_baseline.sh
./experiments/temperature_tuning/run_steering_baseline.sh
```

This script will:
1. Create necessary output directories
2. Run SDAD on both MATH500 and GSM8K datasets
3. Run Best-of-8 on both datasets (saving all candidates)
4. Generate the comparison plot using the efficient approach

## Script Files

- `run_sdad_bestofn_efficient.sh`: Efficient script to run the experiment once
- `run_sdad_bestofn_experiment.sh`: Standard script that runs multiple Best-of-N experiments
- `plot_sdad_bestofn_comparison.py`: Script to generate the comparison plot
- `run_sdad_fixed.py`: Implementation of the SDAD algorithm
- `run_bestofn.py`: Implementation of the Best-of-N approach

## Analyzing Results
```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/sdad_bestofn_comparison/Qwen2.5-Math-1.5B-Instruct$/math   --plot   --output_dir /home/ly/DataDistillation/results/output
```
```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/sdad_bestofn_comparison/Qwen2.5-Math-1.5B-Instruct$/aime24   --plot   --output_dir /home/ly/DataDistillation/results/output
```
```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/sdad_bestofn_comparison/Qwen2.5-1.5B$/math --dataset "math"   --plot   --output_dir /home/ly/DataDistillation/results/output
```

```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/math   --plot   --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/math
```

```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/math   --plot --dataset"math"  --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/math
```

python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/math/temperature-sampling/new   --plot --dataset "math"    --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/math/temperature-sampling/new

python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/math/temperature-sampling/new   --plot --dataset "math"    --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/math/temperature-sampling/new
```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/gsm8k   --plot   --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/gsm8k
```
```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/gsm8k   --plot   --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/gsm8k
```

```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/aime24   --plot   --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/aime24
```
```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/aime24   --plot   --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/aime24
```

python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies_copy.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/aime24/temperature-sampling/before   --plot --dataset "aime24"  --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/aime24/temperature-sampling/before

```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/olympiadbench   --plot   --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/olympiadbench
```
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/olympiadbench/temperature-sampling/new   --plot --dataset "olympiadbench"   --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-Math-1.5B-Instruct$/olympiadbench/temperature-sampling/new

python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/olympiadbench/temperature-sampling/new   --plot --dataset "olympiadbench"   --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/olympiadbench/temperature-sampling/new
```bash
python3 /home/ly/DataDistillation/experiments/temperature_tuning/evaluate_strategies.py   --input /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/olympiadbench   --plot   --output_dir /home/ly/DataDistillation/results/steering/Qwen2.5-1.5B$/olympiadbench
```

The results will be saved in:

```
results/sdad_bestofn_comparison/plots/sdad_bestofn_passk_comparison.png
```

The output structure will be:

```
results/sdad_bestofn_comparison/
├── math/                 # MATH500 results
│   ├── sdad_math500_raw.pkl
│   └── bestofn_n8_math500_raw.pkl  # Only one file with efficient approach
├── gsm8k/                # GSM8K results
│   ├── sdad_gsm8k_raw.pkl
│   └── bestofn_n8_gsm8k_raw.pkl   # Only one file with efficient approach
└── plots/                # Generated plots
    └── sdad_bestofn_passk_comparison.png
```

## Advanced Usage

You can also run the plotting script directly with custom options:

```bash
python experiments/temperature_tuning/plot_sdad_bestofn_comparison.py \
  --results_dir "custom/results/directory" \
  --output_dir "custom/output/directory" \
  --datasets "math,gsm8k" \
  --k_values "1,2,3,4,5,6,7,8" \
  --efficient_bestofn
```

Options:
- `--results_dir`: Directory containing result files
- `--output_dir`: Directory to save plots
- `--datasets`: Comma-separated list of datasets to include
- `--k_values`: Comma-separated list of k values for pass@k calculation
- `--efficient_bestofn`: Flag to use the efficient approach (simulate smaller N values)