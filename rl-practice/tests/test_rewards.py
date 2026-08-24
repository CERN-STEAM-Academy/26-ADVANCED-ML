"""Unit tests for the reward functions.

Students run these (``pytest tests/test_rewards.py``) after filling in the holes in the
notebook. The same battery is available inside the notebook as
``rewards.self_check(format_reward, correctness_reward)`` so that the notebook test cell
and the test suite cannot drift apart.
"""

import pytest

from rlpractice import rewards
from rlpractice.rewards import (
    CORRECT_FULL,
    FORMAT_FULL,
    FORMAT_PARTIAL,
    REWARD_CASES,
    as_conversational,
    completion_texts,
    correctness_reward,
    format_reward,
)


def test_self_check_passes_on_reference_implementation():
    rewards.self_check(format_reward, correctness_reward, verbose=False)


@pytest.mark.parametrize("text,want_format,answer,want_correct", REWARD_CASES)
def test_individual_cases(text, want_format, answer, want_correct):
    completions = as_conversational([text])
    assert format_reward(completions=completions) == [want_format]
    assert correctness_reward(completions=completions, answer=[answer]) == [want_correct]


def test_completion_texts_handles_both_conventions():
    assert completion_texts(["a", "b"]) == ["a", "b"]
    assert completion_texts(as_conversational(["a", "b"])) == ["a", "b"]


def test_returns_one_score_per_completion():
    texts = [case[0] for case in REWARD_CASES]
    answers = [case[2] for case in REWARD_CASES]
    assert len(format_reward(completions=as_conversational(texts))) == len(texts)
    assert len(correctness_reward(completions=as_conversational(texts), answer=answers)) == len(texts)


def test_empty_batch_is_an_empty_list():
    assert format_reward(completions=[]) == []
    assert correctness_reward(completions=[], answer=[]) == []


def test_extra_kwargs_are_tolerated():
    """TRL forwards every dataset column. Reward functions must ignore the ones they
    do not use, or adding a column to the dataset breaks training."""
    completions = as_conversational(["<think>t</think><answer>56</answer>"])
    assert format_reward(completions=completions, prompts=["p"], a=[7], b=[8], answer=["56"]) == [FORMAT_FULL]
    assert correctness_reward(completions=completions, prompts=["p"], a=[7], b=[8], answer=["56"]) == [CORRECT_FULL]


def test_format_and_correctness_are_independent():
    """The point of two reward functions is that they measure two different things."""
    right_shape_wrong_number = as_conversational(["<think>t</think><answer>99</answer>"])
    wrong_shape_right_number = as_conversational(["it is <answer>56</answer>"])

    assert format_reward(completions=right_shape_wrong_number) == [FORMAT_FULL]
    assert correctness_reward(completions=right_shape_wrong_number, answer=["56"]) == [0.0]

    assert format_reward(completions=wrong_shape_right_number) == [FORMAT_PARTIAL]
    assert correctness_reward(completions=wrong_shape_right_number, answer=["56"]) == [CORRECT_FULL]


def test_answers_may_be_ints_or_strings():
    """The dataset stores strings, but nothing should explode if an int arrives."""
    completions = as_conversational(["<think>t</think><answer>56</answer>"])
    assert correctness_reward(completions=completions, answer=["56"]) == [CORRECT_FULL]
    assert correctness_reward(completions=completions, answer=[56]) == [CORRECT_FULL]


def test_scores_are_finite_floats():
    texts = [case[0] for case in REWARD_CASES]
    answers = [case[2] for case in REWARD_CASES]
    for score in format_reward(completions=as_conversational(texts)):
        assert score == score and abs(score) < float("inf")
    for score in correctness_reward(completions=as_conversational(texts), answer=answers):
        assert score == score and abs(score) < float("inf")


def test_self_check_catches_a_broken_implementation():
    """Guard on the guard: self_check must actually fail on plausible student bugs."""

    def returns_a_scalar(completions, **kwargs):
        return 1.0

    def returns_wrong_length(completions, **kwargs):
        return [1.0]

    def ignores_ground_truth(completions, answer=None, **kwargs):
        return [CORRECT_FULL] * len(completions)

    for broken in (returns_a_scalar, returns_wrong_length):
        with pytest.raises(AssertionError):
            rewards.self_check(broken, correctness_reward, verbose=False)
    with pytest.raises(AssertionError):
        rewards.self_check(format_reward, ignores_ground_truth, verbose=False)
