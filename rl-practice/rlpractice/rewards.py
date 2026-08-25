"""Reward functions for GRPO, plus a self-check battery students can run.

The reward function *is* the objective
--------------------------------------
There is no learned reward model here and no human preference data. The entire training
signal is these two Python functions. That is worth pausing on, because it is also the
whole explanation for what goes wrong in Act 2: nothing in ``format_reward`` or
``correctness_reward`` mentions English prose, so nothing in the objective is defending
it. Optimisation pressure is applied to exactly what is measured, and to nothing else.

TRL's calling convention
------------------------
A reward function is called as ``fn(prompts=..., completions=..., **columns)`` where
``columns`` contains every dataset column other than ``prompt``. It must return a list of
floats, one per completion, in order. Get the length or the order wrong and GRPO will
happily train on nonsense - which is why ``self_check`` below exists and why the notebook
runs it before any training starts.

Because the dataset is conversational, ``completions`` arrives as a list of message
lists, not a list of strings. ``completion_texts`` normalises that; it is provided, not
an exercise, because unwrapping a list of dicts teaches nothing about RL.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from .arithmetic import FORMAT_RE, extract_answer_int, extract_answer_text

#: Reward for a completion that respects the ``<think>``/``<answer>`` contract exactly.
FORMAT_FULL = 1.0
#: Partial credit for producing an answer block without the full contract. Partial credit
#: matters more than it looks: it puts a gradient between "nearly right shape" and
#: "no shape at all", which is most of what the model has to learn in the first few steps.
FORMAT_PARTIAL = 0.5
#: Reward for the right number. Weighted above format on purpose - we want a
#: well-formatted wrong answer to score below a badly-formatted right one.
CORRECT_FULL = 2.0


def completion_texts(completions: Sequence[Any]) -> list[str]:
    """Normalise TRL's ``completions`` argument to a plain list of strings.

    Conversational datasets give ``[[{"role": "assistant", "content": "..."}], ...]``;
    plain-text datasets give ``["...", ...]``. Reward functions should not have to care.
    """
    texts: list[str] = []
    for completion in completions:
        if isinstance(completion, str):
            texts.append(completion)
        else:
            # A list of chat messages; the assistant turn is the last one.
            texts.append(completion[-1]["content"])
    return texts


def format_reward(completions, **kwargs) -> list[float]:
    """Reward the ``<think>...</think><answer>...</answer>`` output contract.

    Scoring, per completion:

    * ``FORMAT_FULL`` (1.0) if the whole completion is a think block followed by an
      answer block, ignoring surrounding whitespace. ``arithmetic.FORMAT_RE`` matches
      exactly this.
    * ``FORMAT_PARTIAL`` (0.5) if there is an ``<answer>...</answer>`` block somewhere
      but the strict contract is not met.
    * ``0.0`` otherwise.
    """
    # TODO(hint): normalise with completion_texts, then score each string against
    # FORMAT_RE (full credit), falling back to extract_answer_text (partial credit)
    # BEGIN SOLUTION
    scores = []
    for text in completion_texts(completions):
        if FORMAT_RE.match(text) is not None:
            scores.append(FORMAT_FULL)
        elif extract_answer_text(text) is not None:
            scores.append(FORMAT_PARTIAL)
        else:
            scores.append(0.0)
    return scores
    # END SOLUTION


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    """Reward an exactly correct product.

    ``answer`` is the ground-truth column of the dataset, forwarded by TRL as a list of
    strings, one per completion. Scoring is ``CORRECT_FULL`` (2.0) for an exact integer
    match and ``0.0`` for anything else, including a completion with no answer block.

    Note that this deliberately does *not* re-check the format. Extraction is lenient
    (``extract_answer_int`` takes the last answer block and tolerates commas and spaces)
    so that the two reward functions measure two different things rather than one thing
    twice.
    """
    # TODO(hint): normalise with completion_texts, extract each predicted integer with
    # extract_answer_int, and compare against int(expected)
    # BEGIN SOLUTION
    scores = []
    for text, expected in zip(completion_texts(completions), answer):
        predicted = extract_answer_int(text)
        scores.append(CORRECT_FULL if predicted is not None and predicted == int(expected) else 0.0)
    return scores
    # END SOLUTION


# ---------------------------------------------------------------------------------
# Self-check battery
#
# Shared by tests/test_rewards.py and by the notebook's test cell, so that a student's
# notebook implementation is held to byte-identical standards as the reference one.
# ---------------------------------------------------------------------------------

#: ``(completion_text, expected_format_reward, ground_truth, expected_correctness)``
REWARD_CASES: list[tuple[str, float, str, float]] = [
    # Perfect: right shape, right number.
    ("<think>7 times 8 is 56</think><answer>56</answer>", FORMAT_FULL, "56", CORRECT_FULL),
    # Right shape, wrong number. Format still pays out; correctness does not.
    ("<think>7 times 8 is 54</think><answer>54</answer>", FORMAT_FULL, "56", 0.0),
    # Whitespace and newlines around and inside the blocks are fine.
    ("\n<think>\n step one\n</think>\n<answer> 56 </answer>\n", FORMAT_FULL, "56", CORRECT_FULL),
    # No think block: partial format credit, full correctness credit.
    ("The answer is <answer>56</answer>", FORMAT_PARTIAL, "56", CORRECT_FULL),
    # Think block but no answer block: no credit anywhere.
    ("<think>7 times 8 is 56</think> so it is 56", 0.0, "56", 0.0),
    # Bare number, no tags at all. A model that has forgotten the contract entirely.
    ("56", 0.0, "56", 0.0),
    # Empty completion.
    ("", 0.0, "56", 0.0),
    # Prose before the think block breaks the strict contract but leaves the answer.
    ("Sure! <think>reasoning</think><answer>56</answer>", FORMAT_PARTIAL, "56", CORRECT_FULL),
    # Thousands separator inside the answer: lenient parsing accepts it.
    ("<think>work</think><answer>1,234</answer>", FORMAT_FULL, "1234", CORRECT_FULL),
    # Two answer blocks: the last one is the commitment.
    ("<think>t</think><answer>12</answer> wait <answer>56</answer>", FORMAT_PARTIAL, "56", CORRECT_FULL),
    # Answer block with no integer in it.
    ("<think>t</think><answer>fifty-six</answer>", FORMAT_FULL, "56", 0.0),
    # Negative numbers parse, and are wrong here.
    ("<think>t</think><answer>-56</answer>", FORMAT_FULL, "56", 0.0),
    # Multi-line reasoning, the common real case.
    ("<think>\n7 x 8\n= 56\n</think>\n<answer>56</answer>", FORMAT_FULL, "56", CORRECT_FULL),
]


def as_conversational(texts: Sequence[str]) -> list[list[dict[str, str]]]:
    """Wrap plain strings the way TRL hands conversational completions to reward funcs."""
    return [[{"role": "assistant", "content": t}] for t in texts]


def self_check(
    format_fn: Callable[..., list[float]],
    correctness_fn: Callable[..., list[float]],
    verbose: bool = True,
) -> None:
    """Assert that a pair of reward functions behaves. Raises AssertionError on failure.

    Checks the three things that actually break in practice:

    1. the returned value is a list of floats of exactly the right length,
    2. the scores are right for a battery of realistic completions,
    3. both calling conventions work (plain strings and conversational message lists),
       and batching does not reorder anything.

    A reward function that silently returns the wrong shape is a miserable thing to debug
    against a live training loop, so debug it here instead.
    """
    texts = [case[0] for case in REWARD_CASES]
    want_format = [case[1] for case in REWARD_CASES]
    answers = [case[2] for case in REWARD_CASES]
    want_correct = [case[3] for case in REWARD_CASES]

    for label, completions in (
        ("plain strings", list(texts)),
        ("conversational", as_conversational(texts)),
    ):
        got_format = format_fn(completions=completions, prompts=[None] * len(texts))
        got_correct = correctness_fn(
            completions=completions, prompts=[None] * len(texts), answer=list(answers)
        )

        for name, got in (("format_reward", got_format), ("correctness_reward", got_correct)):
            assert isinstance(got, list), f"{name} ({label}) returned {type(got).__name__}, expected list"
            assert len(got) == len(texts), (
                f"{name} ({label}) returned {len(got)} scores for {len(texts)} completions. "
                "One score per completion, in the same order."
            )
            for i, value in enumerate(got):
                assert isinstance(value, (int, float)) and not isinstance(value, bool), (
                    f"{name} ({label}) returned {value!r} of type {type(value).__name__} at "
                    f"index {i}; expected a float"
                )

        for i, (text, want, got) in enumerate(zip(texts, want_format, got_format)):
            assert abs(got - want) < 1e-9, (
                f"format_reward ({label}) case {i}: expected {want}, got {got}\n  completion: {text!r}"
            )
        for i, (text, want, got) in enumerate(zip(texts, want_correct, got_correct)):
            assert abs(got - want) < 1e-9, (
                f"correctness_reward ({label}) case {i}: expected {want}, got {got}\n"
                f"  completion: {text!r}\n  ground truth: {answers[i]!r}"
            )

    # Order sensitivity: reversing the batch must reverse the scores, nothing else.
    reversed_scores = correctness_fn(
        completions=as_conversational(texts[::-1]),
        prompts=[None] * len(texts),
        answer=list(answers[::-1]),
    )
    assert reversed_scores == want_correct[::-1], (
        "correctness_reward is order-sensitive in the wrong way: scoring the reversed "
        "batch did not give the reversed scores. Are you indexing `answer` correctly?"
    )

    if verbose:
        print(f"reward self-check passed: {len(REWARD_CASES)} cases x 2 calling conventions")
