"""Task data for the GRPO notebook: multiply two integers, answer inside tags.

Why generate rather than download
---------------------------------
Three reasons, in order of importance.

1. **No network dependency.** Student VMs have unverified network access. A dataset we
   can regenerate from a seed in milliseconds cannot fail at the worst possible moment.
2. **Difficulty is a dial.** Act 0 sweeps ``(digits_a, digits_b)`` and picks the setting
   where the base model sits in the 20-50% band. You cannot do that with a fixed
   downloaded benchmark. Getting this dial right is the difference between GRPO learning
   something and GRPO learning nothing, because a group of completions that are *all*
   right or *all* wrong has zero advantage and contributes zero gradient.
3. **Verifiable reward.** Multiplication has exactly one correct answer and it is cheap
   to check. That is what makes the task suitable for RL with a programmatic reward
   instead of a learned reward model.

Prompt format
-------------
DeepSeek-R1-Zero style: a system message that asks for reasoning inside ``<think>`` tags
and the final number inside ``<answer>`` tags, then a user message with the question.
Rows are *conversational* (a list of chat messages), which TRL renders through the
tokenizer's chat template for us.
"""

from __future__ import annotations

import random
import re
from typing import Iterable, Sequence

import datasets

#: Asks for the R1-Zero output contract. Kept deliberately short: the whole rendered
#: prompt has to fit inside ``max_prompt_length = 160`` tokens with room to spare.
SYSTEM_PROMPT = (
    "You are a careful calculator. Think step by step inside <think> </think> tags, "
    "then give the final number inside <answer> </answer> tags. "
    "Example: <think> your reasoning </think><answer> 42 </answer>"
)

#: Matches a full R1-style completion: a think block followed by an answer block, and
#: nothing else. ``re.DOTALL`` so that the reasoning may span lines.
#:
#: The answer body is ``[^<]*`` rather than ``.*?`` on purpose. With ``.*?`` and DOTALL,
#: the engine happily backtracks so that a completion carrying *two* answer blocks -
#: ``<think>t</think><answer>12</answer> wait <answer>56</answer>`` - still matches, by
#: letting the "answer" run from the first ``<answer>`` to the last ``</answer>``. That
#: would hand full format credit to a model that changed its mind mid-completion, which
#: is precisely the behaviour the format reward is supposed to discourage. Forbidding
#: ``<`` inside the answer makes the block exactly what it claims to be: a bare number.
FORMAT_RE = re.compile(r"^\s*<think>(.*?)</think>\s*<answer>([^<]*)</answer>\s*$", re.DOTALL)

#: Matches *any* answer block anywhere in the text. Used by the lenient extractor below.
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def question_text(a: int, b: int) -> str:
    """The user turn for one problem."""
    return f"What is {a} times {b}?"


def digit_range(digits: int) -> tuple[int, int]:
    """Inclusive integer range for a given number of digits.

    One-digit operands start at two rather than zero: multiplying by zero or one is not a
    multiplication problem, it is a copy, and including those cases inflates the measured
    pass rate for reasons that have nothing to do with the model's arithmetic.
    """
    if digits < 1:
        raise ValueError(f"digits must be >= 1, got {digits}")
    if digits == 1:
        return 2, 9
    return 10 ** (digits - 1), 10**digits - 1


def sample_pairs(
    n: int,
    digits_a: int,
    digits_b: int,
    seed: int,
    exclude: Iterable[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    """Sample ``n`` distinct ``(a, b)`` pairs, avoiding anything in ``exclude``.

    Distinctness and the exclusion set together are what let us promise that the eval
    split is genuinely held out. Without them the small difficulty settings leak badly:
    there are only sixty-four distinct one-by-one-digit problems, so an unfiltered train
    and eval split would overlap almost completely.
    """
    lo_a, hi_a = digit_range(digits_a)
    lo_b, hi_b = digit_range(digits_b)
    n_possible = (hi_a - lo_a + 1) * (hi_b - lo_b + 1)

    forbidden = set(exclude)
    n_available = n_possible - len(forbidden)
    if n > n_available:
        raise ValueError(
            f"asked for {n} distinct ({digits_a}x{digits_b})-digit problems but only "
            f"{n_available} are available (space is {n_possible}, excluded {len(forbidden)})"
        )

    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []
    # Rejection sampling. Cheap while n is small relative to the space, which the check
    # above guarantees it is for every setting the notebooks actually use.
    while len(pairs) < n:
        pair = (rng.randint(lo_a, hi_a), rng.randint(lo_b, hi_b))
        if pair in seen or pair in forbidden:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def make_dataset(
    n: int,
    digits_a: int,
    digits_b: int,
    seed: int,
    exclude: Iterable[tuple[int, int]] = (),
) -> datasets.Dataset:
    """Build a GRPO-ready dataset of ``n`` multiplication problems.

    Each row has:

    * ``prompt``  - a conversational prompt (list of chat messages). TRL applies the
      tokenizer's chat template to this automatically.
    * ``answer``  - the ground-truth product, as a string. TRL forwards every column that
      is not ``prompt`` to the reward functions as a keyword argument, which is how
      ``correctness_reward`` gets hold of it.
    * ``a``, ``b`` - the operands, kept for inspection and error analysis.
    """
    pairs = sample_pairs(n, digits_a, digits_b, seed, exclude=exclude)
    rows = {
        "prompt": [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question_text(a, b)},
            ]
            for a, b in pairs
        ],
        "answer": [str(a * b) for a, b in pairs],
        "a": [a for a, _ in pairs],
        "b": [b for _, b in pairs],
    }
    return datasets.Dataset.from_dict(rows)


def dataset_pairs(dataset: datasets.Dataset) -> list[tuple[int, int]]:
    """The ``(a, b)`` pairs in a dataset, for use as an ``exclude`` argument."""
    return list(zip(dataset["a"], dataset["b"]))


# ---------------------------------------------------------------------------------
# Parsing helpers.
#
# These live here, in the module with no student holes, rather than in ``rewards.py``,
# because ``evaluation.py`` needs them too. The frozen evaluation code must keep working
# in a student notebook where the reward functions are still ``NotImplementedError``.
# ---------------------------------------------------------------------------------


def extract_answer_text(completion: str) -> str | None:
    """Return the contents of the *last* ``<answer>`` block, or None if there is none.

    The last one, not the first: models that ramble sometimes emit a worked example
    before committing, and the final answer is the one they are standing behind. Being
    lenient here is deliberate - we want ``correctness_reward`` and ``format_reward`` to
    measure genuinely different things, so correctness should not silently punish a
    slightly malformed but correct answer.
    """
    matches = ANSWER_RE.findall(completion)
    if not matches:
        return None
    return matches[-1]


def parse_int(text: str | None) -> int | None:
    """Parse the integer a fragment of model output is committing to, tolerantly.

    Handles the things models actually do: thousands separators, spaces inside the
    number, a trailing full stop, an equals sign in front. Returns None if there is no
    integer in there at all.

    The **last** integer, not the first. Base Qwen2.5-0.5B routinely restates the
    problem inside the answer block - ``<answer> 10 * 3 = 30 </answer>`` - and taking the
    first integer scores that as 10, marking a correct answer wrong. That is not a
    harmless conservatism: it depresses measured accuracy for reasons unrelated to
    arithmetic, and the Act 0 difficulty sweep uses these numbers to choose what the
    session trains on.

    The trade-off is a completion like ``<answer> 30 (that is 10 times 3) </answer>``,
    which this scores as 3. That pattern is much rarer than the restate-then-solve one,
    and the format reward is what pushes the model away from both.
    """
    if text is None:
        return None
    cleaned = text.strip().replace(",", "").replace("_", "").replace(" ", "")
    # Match decimals as whole numbers, not as two integers separated by a dot. Without
    # this, "56.0" - an entirely ordinary thing for a model told it is a careful
    # calculator to write - tokenises as ["56", "0"], the last-match rule returns 0, and
    # a correct answer is scored wrong. That depresses training reward and the Act 0
    # difficulty measurement for reasons with nothing to do with arithmetic.
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    value = float(matches[-1])
    if value != int(value):
        return None          # a genuinely fractional answer is not an integer answer
    return int(value)


def extract_answer_int(completion: str) -> int | None:
    """Convenience: the integer the completion is committing to, or None."""
    return parse_int(extract_answer_text(completion))


def is_correct(completion: str, answer: str | int) -> bool:
    """Did this completion commit to the right number?"""
    predicted = extract_answer_int(completion)
    return predicted is not None and predicted == int(answer)


def describe(dataset: datasets.Dataset, name: str = "dataset") -> str:
    """A one-line human summary. Printed in the notebooks so nothing is silent."""
    pairs = dataset_pairs(dataset)
    a_vals = [a for a, _ in pairs]
    b_vals = [b for _, b in pairs]
    return (
        f"{name}: {len(dataset)} problems, "
        f"a in [{min(a_vals)}, {max(a_vals)}], b in [{min(b_vals)}, {max(b_vals)}], "
        f"columns={dataset.column_names}"
    )


def sweep_settings() -> Sequence[tuple[int, int]]:
    """The difficulty ladder used by the Act 0 sweep."""
    return [(1, 1), (2, 1), (2, 2), (3, 2)]
