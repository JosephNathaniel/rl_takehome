"""
Implementation using Hugging Face's TRL library GRPO trainer for the same GSM8K task.
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional
from dataclasses import dataclass
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from trl import GRPOConfig, GRPOTrainer

import llms
import utils
import evaluator
import rl_datasets


@dataclass
class GRPOArguments:
    """Arguments for GRPO training configuration."""
    # Model and data
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"
    dataset_name: str = "gsm8k"
    evaluator_name: str = "gsm8k"
    
    # Output and logging
    output_dir: str = "output/Qwen2-0.5B-Instruct_hfgrpo"
    verbose: bool = True
    save_steps: int = 100
    eval_steps: int = 20
    
    # GRPO specific hyperparameters
    learning_rate: float = 5e-6
    num_train_epochs: int = 1
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 1
    warmup_ratio: float = 0.18
    max_grad_norm: float = 0.1
    weight_decay: float = 0.1
    
    # Generation parameters  
    temperature: float = 0.9
    num_generations: int = 16
    max_prompt_length: int = 256
    max_completion_length: int = 786
    
    # GRPO loss parameters
    kl_penalty_loss_type: str = "kl"  # "kl" or "abs"
    kl_penalty_beta: float = 0.04
    
    # Reference model parameters
    ref_model_mixup_alpha: float = 0.1
    ref_model_sync_steps: int = 200
    
    # Other parameters
    seed: int = 7111994
    max_samples: int = 1000  # Limit samples for faster training


def prepare_dataset_for_grpo(
    train_loader: rl_datasets.DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    max_samples: int = None
) -> Dataset:
    """
    Prepare the dataset in the format expected by GRPO trainer.
    
    Args:
        train_loader: The data loader containing questions and answers
        tokenizer: Tokenizer for processing text
        max_samples: Maximum number of samples to use (for faster training)
        
    Returns:
        Dataset formatted for GRPO training
    """
    prompts = []
    answers = []
    questions = []  # Add a list to collect questions
    
    # Reset loader and collect samples
    train_loader.reset()
    count = 0
    
    try:
        while True:
            if max_samples and count >= max_samples:
                break
                
            question, answer = next(train_loader)
            
            # Format prompt as expected by the model
            prompt_messages = [
                {'role': 'system', 'content': train_loader.system_prompt},
                {'role': 'user', 'content': question}
            ]
            
            # Apply chat template
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            prompts.append(prompt_text)
            answers.append(answer)
            questions.append(question)  # Add question to the list
            count += 1
            
    except StopIteration:
        pass
    
    print(f"Prepared {len(prompts)} samples for GRPO training")
    
    # Create HF dataset
    dataset = Dataset.from_dict({
        'prompt': prompts,
        'question': questions,  # Add this field
        'ground_truth_answer': answers
    })
    
    return dataset


class GSM8KRewardFunction:
    """
    Reward function wrapper for use with HF GRPO trainer.
    """
    
    def __init__(self, eval_class: evaluator.RewardEvaluator, device: str):
        self.eval_class = eval_class
        self.device = device
        self.__name__ = "GSM8KRewardFunction"  # Add __name__ attribute for GRPO trainer
    
    def __call__(self, **kwargs) -> List[float]:
        """
        Compute rewards for a batch of samples.
        
        Args:
            **kwargs: Keyword arguments including:
                - prompts: List of prompts (strings or message dicts)
                - completions: List of generated completions (strings or message dicts)
                - completions_ids: List of tokenized completions
                - trainer_state: Current trainer state
                - Any dataset columns (e.g., ground_truth_answer)
            
        Returns:
            List of reward scores for each sample
        """
        # Extract required arguments
        prompts = kwargs.get('prompts', [])
        completions = kwargs.get('completions', [])
        
        # Extract dataset columns that might be present
        ground_truth_answers = kwargs.get('ground_truth_answer', [''] * len(completions))
        questions = kwargs.get('question', [''] * len(completions))
        
        # Optional arguments for advanced usage
        completions_ids = kwargs.get('completions_ids', None)
        trainer_state = kwargs.get('trainer_state', None)
        
        rewards = []
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            # Handle both string and conversational formats
            if isinstance(prompt, str):
                question_text = self._extract_question_from_prompt(prompt)
            else:
                # Conversational format - extract from messages
                question_text = self._extract_question_from_messages(prompt)
            
            # Use question from dataset if available, otherwise extract from prompt
            if i < len(questions) and questions[i]:
                question_text = questions[i]
            
            # Handle completion format
            if isinstance(completion, str):
                response_text = completion
            else:
                # Conversational format - extract content from message
                response_text = completion.get('content', str(completion))
            
            # Get ground truth answer
            answer = ground_truth_answers[i] if i < len(ground_truth_answers) else ''
            
            # Format for evaluator
            mock_prompts = [[{'content': question_text}]]
            mock_completions = [[{'content': response_text}]]
            answers = [answer]
            
            try:
                # Get rewards
                rewards_per_func, _ = self.eval_class.compute_rewards(
                    prompts=mock_prompts,
                    completions=mock_completions,
                    answer=answers,
                    device=self.device
                )
                
                # Sum all reward functions
                total_reward = rewards_per_func.sum().item()
                rewards.append(total_reward)
                
            except Exception as e:
                # Fallback to 0 reward if evaluation fails
                print(f"Warning: Reward computation failed for sample {i}: {e}")
                rewards.append(0.0)
        
        return rewards
    
    def _extract_question_from_prompt(self, prompt: str) -> str:
        """
        Extract the question from the formatted prompt.
        """
        # Simple extraction - look for user content
        lines = prompt.split('\n')
        for i, line in enumerate(lines):
            if 'Question:' in line and i + 1 < len(lines):
                return lines[i + 1].strip()
        
        # Fallback: return the last part after the last newline
        return prompt.split('\n')[-1].strip()
    
    def _extract_question_from_messages(self, messages) -> str:
        """
        Extract the question from conversational format messages.
        
        Args:
            messages: List of message dictionaries or single message dict
            
        Returns:
            Question text as string
        """
        if isinstance(messages, dict):
            # Single message
            return messages.get('content', str(messages))
        elif isinstance(messages, list):
            # List of messages - find the user message
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    if role == 'user':
                        return msg.get('content', '')
            # Fallback: return content of last message
            if messages and isinstance(messages[-1], dict):
                return messages[-1].get('content', str(messages[-1]))
        
        # Final fallback
        return str(messages)


def evaluate_model_on_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_loader: rl_datasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: GRPOArguments,
    step: int
) -> Dict[str, float]:
    """
    Evaluate the model on the test set.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        test_loader: Test data loader
        eval_class: Evaluator for computing rewards
        device: Device to run on
        args: Training arguments
        step: Current training step
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating model at step {step}...")
    
    model.eval()
    total_reward = 0.0
    total_accuracy = 0.0
    num_samples = 0
    
    test_loader.reset()
    
    # Evaluate on a subset of test set for efficiency
    max_eval_samples = 100
    
    with torch.no_grad():
        # Create progress bar
        pbar = tqdm(total=max_eval_samples, desc=f"Evaluating at step {step}", unit="samples")
        
        try:
            while num_samples < max_eval_samples:
                question, answer = next(test_loader)
                
                # Format prompt
                prompt_messages = [
                    {'role': 'system', 'content': test_loader.system_prompt},
                    {'role': 'user', 'content': question}
                ]
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Generate response
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_completion_length,
                        temperature=args.temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # Decode response
                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                # Score response
                mock_prompts = [[{'content': question}]]
                mock_completions = [[{'content': response}]]
                answers = [answer]
                
                rewards_per_func, metrics = eval_class.compute_rewards(
                    prompts=mock_prompts,
                    completions=mock_completions,
                    answer=answers,
                    device=device
                )
                
                total_reward += rewards_per_func.sum().item()
                total_accuracy += metrics['accuracy']
                num_samples += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'avg_reward': f"{total_reward/num_samples:.3f}",
                    'accuracy': f"{total_accuracy/num_samples*100:.1f}%"
                })
                
        except StopIteration:
            pass
        finally:
            pbar.close()
    
    model.train()
    
    avg_reward = total_reward / num_samples if num_samples > 0 else 0.0
    avg_accuracy = total_accuracy / num_samples * 100 if num_samples > 0 else 0.0
    
    eval_metrics = {
        'eval/reward': avg_reward,
        'eval/accuracy': avg_accuracy,
        'eval/num_samples': num_samples
    }
    
    print(f"Evaluation Results at Step {step}:")
    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"  Accuracy: {avg_accuracy:.2f}%")
    print(f"  Samples Evaluated: {num_samples}")
    
    return eval_metrics


def main():
    """Main training function using HF GRPO."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="GRPO training with Hugging Face TRL")
    
    # Add all arguments from GRPOArguments
    for field_name, field in GRPOArguments.__dataclass_fields__.items():
        field_type = field.type
        default_value = field.default
        
        if field_type == bool:
            parser.add_argument(f"--{field_name}", action="store_true", 
                              default=default_value, help=f"{field_name} (default: {default_value})")
        else:
            parser.add_argument(f"--{field_name}", type=field_type, 
                              default=default_value, help=f"{field_name} (default: {default_value})")
    
    parsed_args = parser.parse_args()
    args = GRPOArguments(**vars(parsed_args))
    
    # Seed everything
    utils.seed_everything(args.seed)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')
    
    print(f"Using device: {device}")
    print(f"Training arguments: {args}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    base_model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)
    
    model = base_model
    
    # Load datasets
    print("Loading datasets...")
    train_loader, test_loader = rl_datasets.get_dataloaders(args.dataset_name)
    
    # Prepare dataset for GRPO
    train_dataset = prepare_dataset_for_grpo(
        train_loader, tokenizer, max_samples=args.max_samples
    )
    
    # Setup evaluator
    eval_class = evaluator.get_evaluator(args.evaluator_name)
    
    # Create reward function
    reward_function = GSM8KRewardFunction(eval_class, device)
    
    # Configure GRPO training arguments
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # GRPO specific parameters
        beta=args.kl_penalty_beta,  # Use beta instead of kl_penalty_beta
        
        # Generation parameters
        temperature=args.temperature,
        # max_steps=args.max_completion_length, 
        generation_batch_size=args.num_generations,  # Set generation_batch_size to match num_generations
        
        # Other parameters
        dataloader_drop_last=True,
        report_to=None,  # Disable wandb for now
        run_name=f"grpo_gsm8k_{args.model_name.split('/')[-1]}",
    )
    
    # Create GRPO trainer
    print("Creating GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=[reward_function],
    )
    
    # Custom evaluation callback
    class EvaluationCallback:
        def __init__(self, test_loader, eval_class, device, args):
            self.test_loader = test_loader
            self.eval_class = eval_class
            self.device = device
            self.args = args
            self.eval_results = {}
        
        def on_evaluate(self, trainer, step):
            """Called during evaluation."""
            eval_metrics = evaluate_model_on_test_set(
                trainer.model, trainer.tokenizer, self.test_loader,
                self.eval_class, self.device, self.args, step
            )
            self.eval_results[step] = eval_metrics
            
            # Save evaluation results
            eval_path = os.path.join(self.args.output_dir, 'eval_results.json')
            with open(eval_path, 'w') as f:
                json.dump(self.eval_results, f, indent=4)
    
    # Add evaluation callback
    eval_callback = EvaluationCallback(test_loader, eval_class, device, args)
    
    # Perform initial evaluation
    print("Performing initial evaluation...")
    eval_callback.on_evaluate(trainer, 0)
    
    # Train the model
    print("Starting GRPO training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training encountered an error: {e}")
        print("Saving current model state...")
        trainer.save_model(os.path.join(args.output_dir, "checkpoint-error"))
        raise
    
    # Final evaluation
    print("Performing final evaluation...")
    final_step = training_args.num_train_epochs * len(train_dataset) // training_args.per_device_train_batch_size
    eval_callback.on_evaluate(trainer, final_step)
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    
    print("Training completed successfully!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
