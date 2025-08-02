# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains boilerplate code for LLM inference, focusing on interacting with and running inference on large language models (LLMs). The codebase is designed to work with transformers-based models, particularly Qwen2 models, with features for attention manipulation, caching, and experimental utilities.

## Setup and Installation

```bash
# Install the package in development mode
pip install -e .
```

## Key Components

1. **Inference Engine**: The core class that handles loading models, tokenizers, and provides methods for model inference.
   - Located in `src/inference.py` and `src/inference_utils.py`
   - Supports different model types with a focus on causal language models

2. **Hookers**: Classes that hook into the model's attention mechanism for analysis or manipulation.
   - Located in `src/hooker.py`
   - Includes `BaseHooker`, `WriteAttentionHooker`, and `ZeroOutHooker`

3. **Cache Management**: Utilities for caching model responses and generation statistics.
   - `CacheManager` in `src/inference_utils.py`

4. **Prompt Templates**: Predefined prompt templates for different models and tasks.
   - Located in `src/prompts/`

## Common Tasks

### Running Inference

```bash
# Example of running a basic inference script
python experiments/inference_only/yue_infer.py
```

### Running Temperature Tuning Experiments

```bash
# Run temperature tuning experiments
python experiments/temperature_tuning/run.py --batch_size 4 --input_start 0 --input_end 100
```

### Analyzing Results

Analysis notebooks are available in the `experiments/temperature_tuning/` directory:
- `analyze.ipynb`
- `count_confident_token.ipynb`

## Configuration

Default generation parameters are stored in `config/default_generation_config.json` and include:

- `attention_implementation`: "flash_attention_2"
- `max_new_tokens`: 2048 
- `temperature`: 0.6
- `top_k`: 50
- `top_p`: 0.95
- `repetition_penalty`: 1.0
- `num_return_sequences`: 1
- `do_sample`: false
- `batch_size_per_device`: 4

## Code Architecture Details

### Inference Flow

1. An `InferenceEngine` instance is created with a model repository.
2. The model and tokenizer are loaded, with options for using auto models or custom model classes.
3. `generate()` method handles text generation with configurable parameters.
4. For attention manipulation, `hooker` classes can be provided to the model's forward pass.

### Extending for New Models

The codebase has specialized handling for Qwen models, but can be extended for other models by:

1. Updating the model selection logic in `InferenceEngine.init_model_tokenizer()`
2. Adding new model-specific prompt templates
3. Customizing the hooker classes if needed for different attention mechanisms

### Cache Management

The `CacheManager` class provides efficient caching of model responses and can:
1. Load previous inference results from disk
2. Run inference only on new inputs
3. Save results for future use

## Important Notes

- The codebase includes experimental features for analyzing model attention patterns
- Default paths for config files and cache are relative to the project root
- For distributed inference, the code has utilities for handling batched prompts across devices