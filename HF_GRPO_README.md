# Hugging Face GRPO Implementation

This script (`main_hf_grpo.py`) provides an implementation of GRPO (Group Relative Policy Optimization) using Hugging Face's TRL (Transformers Reinforcement Learning) library, as an alternative to the custom implementation in `main.py`.

## Key Differences from Custom Implementation

### 1. **Library Usage**
- **Custom (`main.py`)**: Implements GRPO from scratch using PyTorch
- **HF (`main_hf_grpo.py`)**: Uses TRL's `GRPOTrainer` and `GRPOConfig`

### 2. **Training Loop**
- **Custom**: Manual training loop with gradient accumulation, optimizer steps, and loss computation
- **HF**: Uses `trainer.train()` which handles the training loop automatically

### 3. **Model Architecture**
- **Custom**: Uses regular `AutoModelForCausalLM`
- **HF**: Can use `AutoModelForCausalLMWithValueHead` (optional value head for better reward modeling)

### 4. **Dataset Format**
- **Custom**: Uses custom `DataLoader` class with iterator protocol
- **HF**: Converts data to HuggingFace `Dataset` format expected by the trainer

### 5. **Reward Function Integration**
- **Custom**: Reward computation integrated directly into loss calculation
- **HF**: Reward function provided as callable to the trainer

### 6. **Configuration**
- **Custom**: Manual argument parsing and optimizer setup
- **HF**: Uses `GRPOConfig` which inherits from `TrainingArguments`

## Usage

### Basic Training
```bash
python main_hf_grpo.py --output_dir output_hf_grpo
```

### With Custom Parameters
```bash
python main_hf_grpo.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset_name "gsm8k" \
    --output_dir "my_grpo_output" \
    --max_samples 1000 \
    --num_train_epochs 2 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4
```

### Quick Test
```bash
bash test_hf_grpo.sh
```

## Arguments

The script supports all the same conceptual arguments as the custom implementation, but with some naming differences:

- `--num_train_epochs`: Number of training epochs (replaces `num_train_iters`)
- `--per_device_train_batch_size`: Batch size per device
- `--kl_penalty_beta`: KL penalty weight (same as `kl_weight_beta` in custom)
- `--max_samples`: Limit dataset size for faster testing

## Advantages of HF Implementation

1. **Less Code**: Much simpler implementation using proven trainer
2. **Better Integration**: Works seamlessly with HF ecosystem
3. **Automatic Features**: Logging, checkpointing, distributed training support
4. **Optimizations**: Built-in optimizations and best practices
5. **Maintenance**: Updates and bug fixes from HF team

## Advantages of Custom Implementation

1. **Full Control**: Complete control over training loop and loss computation
2. **Transparency**: Clear understanding of exactly what's happening
3. **Customization**: Easy to modify specific aspects of the algorithm
4. **Learning**: Better for understanding GRPO mechanics

## Requirements

- `trl >= 0.14.0` (included in requirements.txt)
- Same dependencies as the custom implementation

## Expected Output

The script will:
1. Load the model and tokenizer
2. Prepare the GSM8K dataset
3. Create the GRPO trainer
4. Run training with periodic evaluation
5. Save the final model and evaluation results

Results are saved to the specified output directory with the same structure as the custom implementation.
