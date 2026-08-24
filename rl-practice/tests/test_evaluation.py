"""Regression tests for the evaluation module's invariants.

These cover defects found in review that were, in every case, silent: nothing crashed,
nothing looked wrong, and the numbers were subtly incorrect. That is the category of bug
worth a permanent test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rlpractice import evaluation  # noqa: E402


def test_fixed_seed_restores_the_callers_rng_stream():
    """The frozen evaluation must not disturb the RNG of a training run that calls it.

    ``eval_task_accuracy`` runs from a training callback every ``acc_every`` steps, and
    TRL's GRPO samples its completions from the *global* torch RNG. Seeding without
    restoring resets the exploration noise of the live run to the same state every
    twentieth step, so the policy keeps redrawing the same sample sequence. Nothing about
    that failure is visible in a training log.
    """
    torch.manual_seed(999)
    expected = [torch.randn(4) for _ in range(3)]

    torch.manual_seed(999)
    first = torch.randn(4)
    with evaluation._fixed_seed(evaluation.EVAL_SEED):
        # Consume draws inside the block; the caller's stream must be unaffected.
        torch.randn(100)
    rest = [torch.randn(4) for _ in range(2)]

    assert torch.equal(first, expected[0])
    for got, want in zip(rest, expected[1:]):
        assert torch.equal(got, want), "the RNG stream was not restored"


def test_fixed_seed_restores_even_on_exception():
    torch.manual_seed(7)
    before = torch.random.get_rng_state()
    with pytest.raises(RuntimeError):
        with evaluation._fixed_seed(1234):
            torch.randn(10)
            raise RuntimeError("boom")
    assert torch.equal(torch.random.get_rng_state(), before)


def test_fixed_seed_is_actually_deterministic_inside_the_block():
    with evaluation._fixed_seed(4321):
        a = torch.randn(5)
    with evaluation._fixed_seed(4321):
        b = torch.randn(5)
    assert torch.equal(a, b)


def test_sweep_mirrors_trl_sampling_configuration():
    """The Act 0 sweep chooses what the session trains on, so it must measure the
    distribution GRPO actually draws from. TRL 0.15.2 inherits top_k=50 from the
    transformers GenerationConfig default rather than sampling from the policy."""
    from transformers import GenerationConfig

    assert evaluation.SWEEP_TEMPERATURE == 1.0
    assert evaluation.SWEEP_TOP_K == GenerationConfig().top_k == 50


def test_sweep_does_not_depend_on_student_implemented_rewards():
    """The frozen evaluation must keep working in a distribution where the reward
    functions are still NotImplementedError."""
    import inspect

    source = inspect.getsource(evaluation.eval_sampled_pass_rate)
    assert "from .rewards import" not in source, (
        "eval_sampled_pass_rate must not import the student-implemented reward functions"
    )
    assert "from .arithmetic import" in source


def test_general_prompts_are_frozen_and_exactly_five():
    assert len(evaluation.GENERAL_PROMPTS) == 5
    assert len(set(evaluation.GENERAL_PROMPTS)) == 5
    # No digits and no arithmetic: a perplexity rise on these cannot be explained away as
    # the model having learned the task's output format.
    joined = " ".join(evaluation.GENERAL_PROMPTS)
    assert not any(character.isdigit() for character in joined)


def test_frozen_constants_have_not_drifted():
    """Acts 0, 2 and 4 must call byte-identical evaluation code. If any of these change,
    previously saved snapshots stop being comparable with new ones."""
    assert evaluation.EVAL_SEED == 1234
    assert evaluation.TASK_MAX_NEW_TOKENS == 128
    assert evaluation.GENERAL_MAX_NEW_TOKENS == 96
    assert evaluation.TASK_BATCH_SIZE == 16
    assert evaluation.GENERAL_BATCH_SIZE == 5
    assert evaluation.EVAL_AUTOCAST_DTYPE is torch.float16


def test_eval_functions_take_no_tuning_knobs():
    """The freeze is enforced by signature: no decoding or corpus parameters."""
    import inspect

    forbidden = {"temperature", "top_k", "top_p", "do_sample", "seed", "corpus", "prompts"}
    for name in ("eval_task_accuracy", "eval_general_perplexity", "sample_general_generations"):
        parameters = set(inspect.signature(getattr(evaluation, name)).parameters)
        assert not (parameters & forbidden), f"{name} exposes a knob that could drift: {parameters}"


def test_empty_eval_dataset_fails_loudly():
    import datasets

    empty = datasets.Dataset.from_dict({"prompt": [], "answer": [], "a": [], "b": []})
    with pytest.raises(ValueError, match="non-empty"):
        evaluation.eval_task_accuracy(None, None, empty, n=8)
