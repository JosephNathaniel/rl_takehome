#!/bin/bash

# Test script to run the HF GRPO training with minimal settings
echo "Testing HF GRPO training script..."

cd /workspace/rl_takehome

# Run with minimal settings for testing
python main_hf_grpo.py \
    --model_name "Qwen/Qwen2-0.5B-Instruct" \
    --dataset_name "gsm8k" \
    --evaluator_name "gsm8k" \
    --output_dir "output/Qwen2-0.5B-Instruct_hfgrpo" \
    --max_samples 50 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-6 \
    --save_steps 25 \
    --eval_steps 25 \
    --verbose

echo "Test completed!"
