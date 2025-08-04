"""
Abstract base class and implementations for reward computation in RL training.

"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any

class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    
    The main methods that need to be implemented are:
    - compute_rewards: Computes rewards for a batch of completions
    - get_reward_breakdown: Converts raw reward scores to a labeled dictionary
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            prompts: List of prompt messages in chat format
                    [{"role": "user", "content": "..."}, ...]
            completions: List of completion messages in chat format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Ground truth answer(s) for the prompts
            device: Device to place tensors on ("cpu" or "cuda")
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
                            containing individual reward function scores
            metrics: Dictionary of aggregated metrics including mean rewards
                    per function and total reward
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward scores tensor to a labeled dictionary.
        
        Args:
            reward_scores: Tensor of raw scores from compute_rewards
            
        Returns:
            Dictionary mapping reward function names to their scores
        """
        pass


def get_evaluator(name: str) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.
    
    Args:
        name: Name of the task/dataset to get evaluator for
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "gsm8k":
        return GSM8kEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")



class GSM8kEvaluator(RewardEvaluator):
    """
    Reward evaluator for the GSM8K math problem dataset.
    
    Implements reward functions for:
    - Answer correctness
    - Integer format validation
    - XML formatting (strict and soft)
    - XML tag counting
    """
    ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
    
    def __init__(self):
        self.num_reward_functions = 5
    
    # New extractor, that handles missing tags
    # This method tries to be fairly forgiving, pulling out whatever sits between <answer> and </answer>
    def _extract_xml_answer(self, text: str) -> str:
        """
        Extract answer from <answer>…</answer>. 
        Returns the inner text (stripped), or "" if tags are missing.
        """
        m = self.ANSWER_RE.search(text)
        return m.group(1).strip() if m else ""
    
    def _correctness_reward(self, prompts, completions, answer) -> List[float]:
        """Reward for correct answer."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

    def _int_format_reward(self, completions) -> List[float]:
        """Reward for integer format."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted]

    # New strict format reward that allows newlines inside the blocks
    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format, allowing newlines inside the blocks."""
        pattern = re.compile(
            r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$",
            flags=re.DOTALL
        )
        responses = [c[0]["content"] for c in completions]
        return [0.5 if pattern.match(r) else 0.0 for r in responses]

    # New soft format reward that allows newlines in reasoning/answer
    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format, allowing newlines in reasoning/answer."""
        pattern = re.compile(
            r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>",
            flags=re.DOTALL
        )
        responses = [c[0]["content"] for c in completions]
        # use search so that extra leading/trailing text won’t block a match
        return [0.5 if pattern.search(r) else 0.0 for r in responses]

    # New xml count reward that only penalizes trailing text after </answer>
    def _xml_count_reward(self, completions) -> list[float]:
        """Reward for XML tag counting, with penalty only on text after </answer>."""
        def count_xml(text: str) -> float:
            count = 0.0
            # rewards for each tag appearing exactly once
            if text.count("<reasoning>") == 1: count += 0.125
            if text.count("</reasoning>") == 1: count += 0.125
            if text.count("<answer>") == 1: count += 0.125

            # only penalize trailing content if we actually saw exactly one closing </answer>
            if text.count("</answer>") == 1:
                count += 0.125
                # find the end of the tag, then measure only what's after it
                end_idx = text.find("</answer>") + len("</answer>")
                trailing = text[end_idx:]
                count -= len(trailing) * 0.001

            return count
        
        responses = [completion[0]['content'] for completion in completions]
        return [count_xml(r) for r in responses]

    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute all rewards for the given completions."""

        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)

        # Compute all reward functions
        all_scores = [
            self._correctness_reward(prompts, completions, answer),
            self._int_format_reward(completions),
            self._strict_format_reward(completions),
            self._soft_format_reward(completions),
            self._xml_count_reward(completions)
        ]
        
        # Fill rewards tensor
        for i, scores in enumerate(all_scores):
            rewards_per_func[:, i] = torch.tensor(scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        # My note: this is the average reward across the group, for each reward function
        reward_per_func = rewards_per_func.mean(0)
        
        # Calculate accuracy (perfect correctness score)
        correctness_scores = rewards_per_func[:, 0]  # First reward function is correctness
        num_perfect = (correctness_scores == 2.0).sum().item()
        accuracy = num_perfect / num_completions
        
        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/int_reward_func": reward_per_func[1].item(), 
            "rewards/strict_format_reward_func": reward_per_func[2].item(),
            "rewards/soft_format_reward_func": reward_per_func[3].item(),
            "rewards/xmlcount_reward_func": reward_per_func[4].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "accuracy": accuracy
        }
        
        return rewards_per_func, metrics

    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'correctness': reward_scores[0].item(),
            'integer_format': reward_scores[1].item(),
            'strict_format': reward_scores[2].item(),
            'soft_format': reward_scores[3].item(),
            'xml_count': reward_scores[4].item()
        }