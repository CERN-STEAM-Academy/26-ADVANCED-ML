"""A real 3-step GRPO run on the GPU.

This is the test that would have caught every problem worth catching before a live
session: an fp16 NaN, an out-of-memory at the log-prob forward pass, a reward function
that returns the wrong shape, a TRL version whose internals moved under our subclass.

It is slow by unit-test standards (about a minute, including loading a 0.5B model) and it
skips cleanly when there is no GPU, so it can live in the same suite as the fast tests.
"""

from __future__ import annotations

import math
import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device; the GRPO path is GPU-only"
)

SMOKE_STEPS = 3


@pytest.fixture(scope="module")
def model_and_tokenizer():
    from rlpractice.grpo import load_model_and_tokenizer

    return load_model_and_tokenizer(verbose=False)


def test_device_is_turing_and_torch_did_not_move():
    """The environment gate, as an assertion. See tools/check_env.py for the full version."""
    assert torch.__version__.startswith("2.3.1"), (
        f"torch is {torch.__version__}; pip moved it away from the version the image ships"
    )
    capability = torch.cuda.get_device_capability()
    if capability != (7, 5):
        pytest.skip(f"not a T4 (capability {capability}); memory budgets are untested here")


def test_model_loads_in_fp32_with_sdpa(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    assert next(model.parameters()).dtype is torch.float32, (
        "the model must be loaded in fp32. Loading fp16 weights and also setting fp16=True "
        "in the training config is the classic NaN generator."
    )
    assert model.config._attn_implementation == "sdpa", "flash-attention has no sm75 kernels"
    assert tokenizer.pad_token_id is not None


def test_frozen_evaluation_runs(model_and_tokenizer):
    from rlpractice import evaluation
    from rlpractice.arithmetic import make_dataset

    model, tokenizer = model_and_tokenizer
    eval_dataset = make_dataset(n=8, digits_a=2, digits_b=1, seed=evaluation.EVAL_SEED)

    perplexity = evaluation.eval_general_perplexity(model, tokenizer)
    assert math.isfinite(perplexity) and 1.0 < perplexity < 1000.0, (
        f"general perplexity is {perplexity}, which is not a plausible value for a "
        "working instruction-tuned model on ordinary English prose"
    )

    accuracy = evaluation.eval_task_accuracy(model, tokenizer, eval_dataset, n=8)
    assert 0.0 <= accuracy <= 1.0

    generations = evaluation.sample_general_generations(model, tokenizer)
    assert len(generations) == len(evaluation.GENERAL_PROMPTS) == 5
    assert all(isinstance(text, str) for text in generations)


def test_evaluation_is_deterministic(model_and_tokenizer):
    """Frozen means frozen. If two identical calls disagree, every before/after number
    in notebook 2 is noise."""
    from rlpractice import evaluation

    model, tokenizer = model_and_tokenizer
    first = evaluation.eval_general_perplexity(model, tokenizer)
    second = evaluation.eval_general_perplexity(model, tokenizer)
    assert abs(first - second) < 1e-4, f"perplexity is not reproducible: {first} vs {second}"

    generations_a = evaluation.sample_general_generations(model, tokenizer)
    generations_b = evaluation.sample_general_generations(model, tokenizer)
    assert generations_a == generations_b, "greedy generation is not reproducible"


def test_three_grpo_steps_produce_finite_losses(tmp_path, model_and_tokenizer):
    """The whole pipeline: LoRA, generation, rewards, advantages, fp16 backward.

    Asserts three separate things that all fail differently in practice:

    1. the loss stayed finite (fp16 stability),
    2. our subclass really recorded entropy and zero-std-group metrics (TRL internals
       have not moved under us),
    3. peak memory is inside the T4 budget.
    """
    from rlpractice import grpo
    from rlpractice.arithmetic import make_dataset
    from rlpractice.callbacks import InstrumentedGRPOTrainer, MetricsCSVCallback, NaNGuardCallback
    from rlpractice.rewards import correctness_reward, format_reward

    model, tokenizer = model_and_tokenizer
    torch.cuda.reset_peak_memory_stats()

    train_dataset = make_dataset(n=16, digits_a=2, digits_b=1, seed=0)
    eval_dataset = make_dataset(n=8, digits_a=2, digits_b=1, seed=99)

    config = grpo.build_grpo_config(
        beta=0.0,
        max_steps=SMOKE_STEPS,
        output_dir=str(tmp_path / "run"),
        disable_tqdm=True,
    )
    csv_path = tmp_path / "log.csv"
    metrics = MetricsCSVCallback(
        csv_path=csv_path,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        ppl_every=2,
        acc_every=0,          # generation-based eval is too slow for a smoke test
        dashboard_every=None,
        verbose=False,
    )

    trainer = InstrumentedGRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward],
        args=config,
        train_dataset=train_dataset,
        peft_config=grpo.lora_config(),
        callbacks=[NaNGuardCallback(), metrics],
    )
    trainer.train()

    # 1. finite losses, one per step
    losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
    assert len(losses) == SMOKE_STEPS, f"expected {SMOKE_STEPS} logged steps, got {len(losses)}"
    for step, loss in enumerate(losses, start=1):
        assert math.isfinite(loss), f"loss at step {step} is {loss}"

    # 2. our instrumentation fired
    rows = [row for row in metrics.rows if row["step"] != 0]
    assert len(rows) == SMOKE_STEPS
    for row in rows:
        for column in (
            "reward_mean",
            "reward_std",
            "kl",
            "entropy",
            "frac_zero_std_groups",
            "completion_length_mean",
        ):
            assert row[column] != "", f"{column} was never recorded; did TRL's internals move?"
        assert float(row["entropy"]) > 0.0, "entropy estimate must be positive"
        assert 0.0 <= float(row["frac_zero_std_groups"]) <= 1.0
        # beta = 0.0, so KL is measured but not penalised. It must still be recorded.
        assert math.isfinite(float(row["kl"]))

    assert os.path.exists(csv_path)

    # 3. memory
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    assert peak_gb < 14.0, f"peak allocated memory {peak_gb:.2f} GB exceeds the 14 GB budget"


def test_nan_guard_raises_with_a_useful_message():
    """The guard is worthless if it does not fire, and a live session is the wrong place
    to find that out."""
    from transformers import TrainerState

    from rlpractice.callbacks import NaNGuardCallback

    guard = NaNGuardCallback(check_parameters=False)
    state = TrainerState()
    state.global_step = 42

    guard.on_log(None, state, None, logs={"loss": 0.5})  # finite: no exception

    with pytest.raises(FloatingPointError, match="step 42"):
        guard.on_log(None, state, None, logs={"loss": float("nan")})
    with pytest.raises(FloatingPointError, match="fp32"):
        guard.on_log(None, state, None, logs={"loss": float("inf")})
