# testing for new reward functions. To run, go to rl_takehome directory and run:
# python -m pytest -q

import pytest
from evaluator import GSM8kEvaluator

# Helper to wrap content into the expected completions format
def make_completion(text):
    return [{"role": "assistant", "content": text}]

@pytest.fixture
def evaluator():
    return GSM8kEvaluator()

def test_correctness_reward(evaluator):
    prompts = [None, None]  # unused by correctness_reward
    answers = ["42", "7"]
    comps = [
        make_completion("<reasoning>r</reasoning><answer>42</answer>"),
        make_completion("<reasoning>r</reasoning><answer>8</answer>")
    ]
    scores = evaluator._correctness_reward(prompts, comps, answers)
    assert scores == [2.0, 0.0]

def test_int_format_reward(evaluator):
    comps = [
        make_completion("<answer>123</answer>"),
        make_completion("<answer>12.3</answer>"),
        make_completion("<answer>abc</answer>")
    ]
    scores = evaluator._int_format_reward(comps)
    assert scores == [0.5, 0.0, 0.0]

@pytest.mark.parametrize("text,expected", [
    ("<reasoning>r</reasoning><answer>a</answer>", 0.5),
    ("<reasoning>r</reasoning> <answer>a</answer>", 0.5),
    ("<reasoning>r</reasoning>\n<answer>a</answer>", 0.5),
    ("<reasoning>r</reasoning><answer>a</answer> extra", 0.0),
    ("garbage<reasoning>r</reasoning><answer>a</answer>", 0.0),
])
def test_strict_format_reward(evaluator, text, expected):
    scores = evaluator._strict_format_reward([make_completion(text)])
    assert scores == [expected]

@pytest.mark.parametrize("text,expected", [
    ("prefix<reasoning>r</reasoning><answer>a</answer>suffix", 0.5),
    ("<reasoning>r</reasoning> middle <answer>a</answer>", 0.5),
    ("<reasoning>r</reasoning>", 0.0),
    ("<answer>a</answer>", 0.0),
])
def test_soft_format_reward(evaluator, text, expected):
    scores = evaluator._soft_format_reward([make_completion(text)])
    assert scores == [expected]

def test_xml_count_reward_exact(evaluator):
    text = "<reasoning>r</reasoning><answer>a</answer>"
    scores = evaluator._xml_count_reward([make_completion(text)])
    # 4 tags × 0.125 = 0.5, no penalty
    assert pytest.approx(scores[0], rel=1e-3) == 0.5

def test_xml_count_reward_with_extra(evaluator):
    text = "noise<reasoning>r</reasoning><answer>a</answer>tail"
    scores = evaluator._xml_count_reward([make_completion(text)])
    # penalty = (len("noise")+len("tail")) * 0.001
    expected = max(0.5 - (5 + 4) * 0.001, 0.0)
    assert pytest.approx(scores[0], rel=1e-3) == expected

def test_compute_rewards_and_metrics(evaluator):
    prompts = [None, None]
    answers = ["1", "2"]
    comps = [
        make_completion("<reasoning>r1</reasoning><answer>1</answer>"),
        make_completion("<reasoning>r2</reasoning><answer>wrong</answer>")
    ]
    rewards_tensor, metrics = evaluator.compute_rewards(prompts, comps, answers, device="cpu")
    
    # Tensor shape should be (2 completions × 5 reward functions)
    assert rewards_tensor.shape == (2, evaluator.num_reward_functions)
    
    # First completion: all sub-rewards positive
    assert rewards_tensor[0].tolist() == [2.0, 0.5, 0.5, 0.5, 0.5]
    # Second completion: only formatting rewards
    assert rewards_tensor[1].tolist() == [0.0, 0.0, 0.5, 0.5, 0.5]
    
    # Metrics sanity checks
    assert pytest.approx(metrics["rewards/correctness_reward_func"], rel=1e-3) == 1.0  # (2 + 0) / 2
    assert pytest.approx(metrics["accuracy"], rel=1e-3) == 0.5                         # 1 perfect out of 2
