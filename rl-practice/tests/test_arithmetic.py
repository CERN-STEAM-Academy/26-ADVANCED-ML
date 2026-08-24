"""Unit tests for task data generation and answer parsing.

The leakage tests are the ones that matter. If the train and eval splits overlap, every
before/after number in notebook 2 is meaningless, and the failure is completely silent.
"""

import pytest

from rlpractice import arithmetic
from rlpractice.arithmetic import (
    SYSTEM_PROMPT,
    dataset_pairs,
    digit_range,
    extract_answer_int,
    extract_answer_text,
    is_correct,
    make_dataset,
    parse_int,
    sample_pairs,
)


# --- data generation -------------------------------------------------------------


def test_digit_range_excludes_trivial_one_digit_operands():
    assert digit_range(1) == (2, 9)
    assert digit_range(2) == (10, 99)
    assert digit_range(3) == (100, 999)
    with pytest.raises(ValueError):
        digit_range(0)


def test_dataset_has_expected_schema():
    ds = make_dataset(n=8, digits_a=2, digits_b=2, seed=0)
    assert len(ds) == 8
    assert set(ds.column_names) == {"prompt", "answer", "a", "b"}
    row = ds[0]
    assert isinstance(row["prompt"], list) and len(row["prompt"]) == 2
    assert row["prompt"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert row["prompt"][1]["role"] == "user"
    assert isinstance(row["answer"], str)


def test_answers_are_correct_products():
    ds = make_dataset(n=64, digits_a=3, digits_b=2, seed=7)
    for row in ds:
        assert int(row["answer"]) == row["a"] * row["b"]
        assert str(row["a"]) in row["prompt"][1]["content"]
        assert str(row["b"]) in row["prompt"][1]["content"]


def test_operands_respect_the_digit_ranges():
    for digits_a, digits_b in arithmetic.sweep_settings():
        ds = make_dataset(n=32, digits_a=digits_a, digits_b=digits_b, seed=3)
        lo_a, hi_a = digit_range(digits_a)
        lo_b, hi_b = digit_range(digits_b)
        for row in ds:
            assert lo_a <= row["a"] <= hi_a
            assert lo_b <= row["b"] <= hi_b


def test_generation_is_deterministic_given_the_seed():
    a = make_dataset(n=16, digits_a=2, digits_b=2, seed=1234)
    b = make_dataset(n=16, digits_a=2, digits_b=2, seed=1234)
    assert dataset_pairs(a) == dataset_pairs(b)


def test_different_seeds_give_different_data():
    a = make_dataset(n=32, digits_a=2, digits_b=2, seed=1)
    b = make_dataset(n=32, digits_a=2, digits_b=2, seed=2)
    assert dataset_pairs(a) != dataset_pairs(b)


def test_pairs_within_a_split_are_distinct():
    pairs = sample_pairs(n=60, digits_a=1, digits_b=1, seed=0)
    assert len(set(pairs)) == 60


def test_exclude_guarantees_a_disjoint_split():
    """The whole point of the eval split. One-by-one digits is the adversarial case:
    the space is only 64 problems, so unfiltered splits would overlap heavily."""
    eval_ds = make_dataset(n=32, digits_a=1, digits_b=1, seed=1234)
    train_ds = make_dataset(n=32, digits_a=1, digits_b=1, seed=0, exclude=dataset_pairs(eval_ds))
    assert set(dataset_pairs(train_ds)).isdisjoint(dataset_pairs(eval_ds))


def test_asking_for_more_problems_than_exist_fails_loudly():
    with pytest.raises(ValueError, match="only"):
        sample_pairs(n=100, digits_a=1, digits_b=1, seed=0)  # space is 8 x 8 = 64


def test_exclusion_is_counted_against_availability():
    eval_pairs = sample_pairs(n=40, digits_a=1, digits_b=1, seed=0)
    with pytest.raises(ValueError, match="only"):
        sample_pairs(n=30, digits_a=1, digits_b=1, seed=1, exclude=eval_pairs)  # 64 - 40 = 24


# --- parsing ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("<answer>42</answer>", "42"),
        ("<answer> 42 </answer>", " 42 "),
        ("<answer>\n42\n</answer>", "\n42\n"),
        ("prefix <answer>1</answer> mid <answer>2</answer> suffix", "2"),
        ("no tags here", None),
        ("<answer>unclosed", None),
        ("", None),
    ],
)
def test_extract_answer_text(completion, expected):
    assert extract_answer_text(completion) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("42", 42),
        (" 42 ", 42),
        ("1,234", 1234),
        ("1 234", 1234),
        ("= 42", 42),
        ("42.", 42),
        ("-7", -7),
        ("the answer is 42", 42),
        # The last integer, not the first: models restate the problem before solving it.
        ("10 * 3 = 30", 30),
        ("The product of 896 and 66 is 59136", 59136),
        ("forty-two", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_int(text, expected):
    assert parse_int(text) == expected


def test_extract_and_compare():
    assert extract_answer_int("<think>x</think><answer>4,830</answer>") == 4830
    assert is_correct("<think>x</think><answer>4830</answer>", "4830")
    assert is_correct("<think>x</think><answer>4830</answer>", 4830)
    assert not is_correct("<think>x</think><answer>4831</answer>", "4830")
    assert not is_correct("no answer at all", "4830")


def test_format_regex_matches_only_the_strict_contract():
    assert arithmetic.FORMAT_RE.match("<think>a</think><answer>b</answer>")
    assert arithmetic.FORMAT_RE.match("  <think>a\nb</think>\n<answer>c</answer>  ")
    assert not arithmetic.FORMAT_RE.match("hello <think>a</think><answer>b</answer>")
    assert not arithmetic.FORMAT_RE.match("<think>a</think>")
    assert not arithmetic.FORMAT_RE.match("<answer>b</answer>")
    # Two answer blocks must not earn full format credit.
    assert not arithmetic.FORMAT_RE.match("<think>t</think><answer>12</answer> x <answer>56</answer>")


# --- regression: decimals in the answer block ------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # "56.0" is an entirely ordinary thing for a model told it is a careful
        # calculator to write. Splitting it into ["56", "0"] and taking the last match
        # scored a correct answer as 0, depressing both training reward and the Act 0
        # difficulty measurement for reasons unrelated to arithmetic.
        ("56.0", 56),
        ("56.00", 56),
        ("204.0", 204),
        ("1,234.0", 1234),
        ("-56.0", -56),
        # A genuinely fractional answer is not an integer answer.
        ("3.5", None),
        ("0.5", None),
    ],
)
def test_parse_int_handles_decimals(text, expected):
    assert parse_int(text) == expected


def test_correct_answer_written_as_a_decimal_is_scored_correct():
    assert is_correct("<think>x</think><answer>56.0</answer>", "56")
    assert is_correct("<think>x</think><answer>4830.00</answer>", 4830)
    assert not is_correct("<think>x</think><answer>56.5</answer>", "56")
