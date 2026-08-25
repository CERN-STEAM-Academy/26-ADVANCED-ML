"""Model loading and the shared GRPO geometry for notebook 2.

Two things live here, and both are here for the same reason: Act 1 and Act 4 must differ
in **exactly one** hyperparameter, and the only way to be sure of that is for both to be
built by the same function.

Model loading
-------------
fp32 weights, SDPA attention, offline-first. Each of those is a decision:

* **fp32 weights, ``fp16=True`` in the training config.** Not fp16 weights. The training
  config's ``fp16=True`` turns on autocast plus a ``GradScaler``, which keeps a master
  copy of the weights in fp32 and does the unsafe operations in fp16 with loss scaling.
  Loading fp16 weights *and* enabling fp16 gives you fp16 master weights with fp16
  gradients and no scaler headroom, which is the classic recipe for a NaN loss twenty
  steps in. A 0.5B model in fp32 is about 2 GB, which is nothing on a 16 GB card.
* **SDPA, not flash-attention.** Turing (sm75) is not supported by flash-attn. SDPA gives
  most of the memory benefit through PyTorch's own fused kernels and works everywhere.
* **Offline-first.** Student VM network access is unverified, so the loader prefers a
  pre-staged local snapshot and only falls back to the hub.

The memory arithmetic
---------------------
The bottleneck is not the model. Qwen2.5 has a vocabulary of about 152k tokens, so the
logits tensor in the log-prob forward pass is ``batch x seq_len x 152k``. Sixty-four
sequences of 250 tokens in fp32 is roughly 11 GB of logits alone on a 16 GB card.

The fix is that **generation batch and forward batch are different things**. We generate
8 completions per prompt because GRPO needs a group to normalise within, but we only ever
push 8 sequences through the log-prob forward pass at a time, and we reach an effective
batch of 2 prompts by accumulating gradients rather than by widening the batch.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from . import paths

#: The base model. 0.5B is small enough to train on a T4 inside a lecture slot, and
#: instruction-tuned so that Act 0 has some general ability to lose.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

#: Kept for backwards compatibility. Real resolution lives in ``rlpractice.paths``, which
#: also searches a shared read-only volume - on a CERN Kubeflow server, an /eos path.
DEFAULT_LOCAL_DIR = os.environ.get(paths.MODEL_ENV, "assets/base_model")


def local_model_path(local_dir: str | os.PathLike | None = None) -> str | None:
    """The pre-staged model directory, or None if the weights must come from the hub.

    Searches, in order: ``$RLPRACTICE_MODEL_DIR``, ``assets/base_model`` in the checkout,
    then ``$RLPRACTICE_SHARED_DIR/base_model``. Pass ``local_dir`` to bypass all of that
    and check exactly one place.
    """
    if local_dir is not None:
        path = os.fspath(local_dir)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, paths.MODEL_SENTINEL)):
            return path
        return None
    return paths.model_dir()


def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    local_dir: str | os.PathLike | None = None,
    device: str = "cuda",
    verbose: bool = True,
):
    """Load the base model in fp32 with SDPA attention, preferring pre-staged weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = local_model_path(local_dir)
    source = path or model_id
    kwargs: dict[str, Any] = {"local_files_only": True} if path else {}

    if path is None and verbose:
        print(
            f"[grpo] no local copy of {model_id} found, so it will be fetched from the "
            f"HuggingFace hub (about 1 GB, and it needs network access).\n"
            f"       Looked in: {', '.join(paths.candidates('base_model'))}\n"
            f"       To use a shared copy instead, set {paths.SHARED_ENV} (an /eos path at CERN) "
            f"or {paths.MODEL_ENV}."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.float32,   # fp32 master weights; fp16=True does the rest
            attn_implementation="sdpa",  # flash-attn has no sm75 kernels
            **kwargs,
        )
    except Exception as error:
        if path is not None:
            raise
        # The hub was the only option left and it did not work. Say what to do about it,
        # because "OSError: We couldn't connect to huggingface.co" tells a student nothing.
        raise RuntimeError(
            f"Could not load {model_id}. There is no local copy and the hub is not "
            f"reachable ({type(error).__name__}: {error}).\n\n"
            f"Fix it in one of these ways:\n"
            f"  1. python tools/prestage.py --model      (downloads it into assets/base_model)\n"
            f"  2. export {paths.SHARED_ENV}=/eos/.../rl-practice   (a shared copy laid out\n"
            f"     like assets/, so the weights are at $" + paths.SHARED_ENV + "/base_model)\n"
            f"  3. export {paths.MODEL_ENV}=/path/to/the/weights     (points straight at them)\n\n"
            f"Note that TRAIN_FROM_SCRATCH = False does NOT avoid this: the reference "
            f"adapters are LoRA deltas and are useless without the base weights underneath."
        ) from error
    if device:
        model = model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        bytes_per_param = next(model.parameters()).element_size()
        print(f"loaded {source}")
        print(f"  parameters      {n_params / 1e6:.1f} M")
        print(f"  dtype           {next(model.parameters()).dtype}")
        print(f"  weight memory   {n_params * bytes_per_param / 1024**3:.2f} GB")
        print(f"  attention       {model.config._attn_implementation}")
        print(f"  vocabulary      {model.config.vocab_size:,} tokens")
        print(f"  loaded from     {'pre-staged snapshot' if path else 'the hub'}")
    return model, tokenizer


# ---------------------------------------------------------------------------------
# The shared GRPO geometry
# ---------------------------------------------------------------------------------

#: Every setting that Act 1 and Act 4 share. Act 4 changes ``beta`` and nothing else.
#:
#: The four numbers at the top are the memory budget, and they are load-bearing:
#: ``per_device_train_batch_size`` must be divisible by ``num_generations``, and it is the
#: forward-pass batch, not the generation batch.
GRPO_COMMON: dict[str, Any] = {
    "num_generations": 8,               # G: completions per prompt, the "group" in GRPO
    "per_device_train_batch_size": 8,   # forward-pass batch; must be a multiple of G
    "gradient_accumulation_steps": 2,   # -> 2 prompts per optimiser step
    "max_prompt_length": 160,           # measured prompts are ~74 tokens; ample headroom
    "max_completion_length": 128,       # the single biggest lever on peak memory
    # Deliberately hot. The spec proposed 5e-5; that was measured and rejected, and the
    # reasoning is worth recording because "raise the learning rate" has a narrow window
    # here:
    #   5e-5  reward climbs, then the policy goes deterministic and training stalls with
    #         every group at zero advantage. General perplexity does not move at all
    #         (x1.00 over 53 steps). No forgetting to show.
    #   1e-4  task accuracy 0.594 -> 0.875 in 40 steps. Still x1.00 perplexity.
    #   2e-4  task accuracy 0.594 -> ~0.94 AND general perplexity rises monotonically and
    #         at an accelerating rate (+0.015/step early, +0.03/step by step 80). This is
    #         the setting that makes Act 2 unambiguous.
    #   3e-4  the policy leaves the reference distribution entirely and emits token soup
    #         within 120 steps: reward 0.59 -> 0.00, accuracy 0.000, perplexity x1000.
    #         Spectacular, but it destroys Act 1's "the training worked" premise.
    "learning_rate": 2e-4,
    "temperature": 1.0,                 # entropy estimate in callbacks.py assumes 1.0
    "fp16": True,                       # autocast + GradScaler over fp32 master weights
    "bf16": False,                      # Turing has no bf16 tensor cores
    "use_vllm": False,                  # vLLM would force a torch upgrade
    "lr_scheduler_type": "constant_with_warmup",
    "warmup_steps": 3,
    "logging_steps": 1,                 # MetricsCSVCallback wants one row per step
    "save_strategy": "no",
    "report_to": [],
    "log_completions": False,
    "disable_tqdm": False,
    "gradient_checkpointing": False,
    "seed": 0,
}

#: LoRA. ``all-linear`` rather than just the attention projections: the MLP matrices are
#: where a lot of a small model's factual and stylistic behaviour lives, and Act 2 needs
#: the forgetting to be visible, not subtle.
LORA_KWARGS: dict[str, Any] = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "target_modules": "all-linear",
    "task_type": "CAUSAL_LM",
}


def lora_config(**overrides):
    from peft import LoraConfig

    return LoraConfig(**{**LORA_KWARGS, **overrides})


# ---------------------------------------------------------------------------------
# Session-level constants
#
# The notebook and tools/prestage.py both import these. They are here rather than
# duplicated in each, because a reference adapter trained on one difficulty and a notebook
# that evaluates on another would produce a comparison that is wrong in a way no test
# would catch.
# ---------------------------------------------------------------------------------

#: The difficulty the session trains on: a **mixture**, not a single setting. This is the
#: one place where the measured behaviour of the model overruled the obvious design, so
#: the reasoning is worth writing down.
#:
#: Training on two-by-one digits alone works, briefly, and then stops. The base model
#: passes about 28% of those when sampled at temperature 1.0, comfortably inside the band
#: the Act 0 exercise asks for. GRPO learns the task quickly - held-out accuracy goes from
#: 0.62 to about 0.94 within forty steps - and then three things happen in lockstep:
#: reward saturates at its maximum, policy entropy collapses to ~0.000, and because the
#: policy is now deterministic all eight completions in a group become *identical*. Their
#: rewards are identical, so the advantage is exactly zero, so the gradient is exactly
#: zero. Measured: ``frac_zero_std_groups`` sits at 1.00 for essentially every step after
#: about step 120, and general perplexity plateaus at x1.07 and stays there no matter how
#: many more steps you buy. The run looks busy and is doing nothing.
#:
#: Mixing in two-by-two digits - which the model does *not* saturate - keeps a healthy
#: fraction of groups varying, so entropy survives (measured 0.02-0.49 rather than 0.000)
#: and the gradient keeps flowing. Same learning rate, same everything else, 200 steps:
#: held-out accuracy 0.62 -> 0.97 at its peak, and general perplexity 27.6 -> 33.1, which
#: is the difference between a demonstration and an anecdote.
#:
#: The transferable point, and it is worth making to students: task difficulty is not a
#: scalar you tune, it is a *distribution* you shape, and what you are shaping it for is
#: non-degenerate advantage.
#
#: MEASURED OUTCOME, and the reason this is a single setting again: the mixture keeps the
#: gradient alive and does raise general perplexity further, but it costs the premise of
#: the whole session. On a mixed held-out split, Act 1 held-out accuracy went *down*
#: (0.578 -> 0.328 over 100 steps) because the two-by-two half degrades faster than the
#: two-by-one half improves, and "we trained it and it got worse at the task" is not a
#: foundation Acts 2 to 4 can stand on. The mixture was also not reproducible: two runs of
#: the identical configuration gave perplexity x1.20 and x1.03 at the same step count,
#: which is not something to put in front of a room.
#:
#: Single-setting two-by-one digits behaved consistently across three runs (perplexity
#: x1.04 to x1.07, accuracy 0.625 -> 0.84 to 0.94) and is what ships. The mixture finding
#: is kept in the notebook narrative because the *mechanism* it demonstrates - entropy
#: collapse producing zero advantage - is the real lesson of Act 3.
TRAIN_MIX = ((2, 1),)

#: Kept as the headline setting for the Act 0 narrative, which picks one difficulty from
#: the sweep before discovering why a mixture is better.
TRAIN_DIGITS = TRAIN_MIX[0]

#: Split seeds. The eval split is drawn first and the training split excludes it, so the
#: two are disjoint by construction rather than by luck.
EVAL_SEED_SPLIT = 1234
TRAIN_SEED_SPLIT = 0
N_EVAL = 64
N_TRAIN = 512

#: Wall-clock budgets, set by the length of the session rather than by what would be
#: ideal. The whole practice - both notebooks - has to fit in 105 minutes including the
#: talking, so the two training runs together get about sixteen of those minutes.
#:
#: 100 steps is enough. Measured on the reference runs, the interesting part of the
#: trajectory is over well before then: reward reaches its ceiling by step 15, held-out
#: accuracy peaks between steps 40 and 120, and general perplexity has done most of its
#: rise by step 100 (27.6 -> about 28.7 at step 100 against 29.7 at step 150). Going to
#: 150 buys a slightly larger number and costs four minutes of a session that does not
#: have four minutes.
ACT1_TIME_BUDGET_SECONDS = 480
ACT4_TIME_BUDGET_SECONDS = 420
#: An upper bound on steps regardless of how fast the hardware turns out to be. On a T4
#: (~4.7 s/step) the cap binds at about eight minutes and the time budgets are the safety
#: net for slower hardware. That ordering is deliberate: a cap that binds gives a
#: reproducible run, and a time budget that binds gives a session that still ends on time.
MAX_STEPS_CAP = 100

#: Evaluation cadence during training. Perplexity is one cheap forward pass, so it runs
#: often; accuracy needs generation, so it runs rarely and on a fixed prefix of the eval
#: set, which keeps it comparable with the full endpoint measurements.
PPL_EVERY = 5
ACC_EVERY = 20
#: Deliberately equal to N_EVAL. Using a smaller n during training and the full n at the
#: endpoints makes the plotted curve and the before/after table two different estimators of
#: the same quantity, and they disagree: on one run the step-100 curve point read 0.906
#: (29/32) while the snapshot of the *same model* read 0.844 (54/64). A 0.06 step of pure
#: estimator artefact, on a curve whose entire claim is a rise of about 0.17, is not
#: acceptable - the binomial standard error at n=32 is already about 0.09. Evaluating 64
#: problems rather than 32 costs roughly four extra seconds per probe, seven times a run.
ACC_N = N_EVAL


def build_splits(mix: tuple[tuple[int, int], ...] = TRAIN_MIX):
    """The train and eval datasets, guaranteed disjoint. One definition, used everywhere.

    ``mix`` is a tuple of ``(digits_a, digits_b)`` settings. Each contributes an equal
    share of both splits, and each setting's training rows exclude that setting's eval
    rows, so the two splits are disjoint by construction rather than by luck.

    The eval split is mixed in the same proportions as the training split on purpose. An
    eval set drawn only from the easy setting would report an accuracy that says nothing
    about the harder half of what we trained on, and would flatter the run.
    """
    import datasets as hf_datasets

    from .arithmetic import dataset_pairs, make_dataset

    per_setting_eval = max(1, N_EVAL // len(mix))
    per_setting_train = max(1, N_TRAIN // len(mix))

    eval_parts, train_parts = [], []
    for digits_a, digits_b in mix:
        eval_part = make_dataset(
            n=per_setting_eval, digits_a=digits_a, digits_b=digits_b, seed=EVAL_SEED_SPLIT
        )
        train_part = make_dataset(
            n=per_setting_train,
            digits_a=digits_a,
            digits_b=digits_b,
            seed=TRAIN_SEED_SPLIT,
            exclude=dataset_pairs(eval_part),
        )
        eval_parts.append(eval_part)
        train_parts.append(train_part)

    # Interleave rather than concatenate for the eval split, so that a truncated
    # evaluation (``n=32`` during training) still sees both difficulties instead of only
    # the first one. This matters: ``eval_task_accuracy`` takes the first n rows.
    eval_dataset = hf_datasets.concatenate_datasets(eval_parts).shuffle(seed=EVAL_SEED_SPLIT)
    train_dataset = hf_datasets.concatenate_datasets(train_parts).shuffle(seed=TRAIN_SEED_SPLIT)
    return train_dataset, eval_dataset


def build_grpo_config(beta: float, max_steps: int, output_dir: str, **overrides):
    """A ``trl.GRPOConfig`` built from the shared geometry, with ``beta`` set explicitly.

    ``beta`` is a required positional argument rather than something with a default,
    because it is the entire experiment. Act 1 passes 0.0 and Act 4 passes 0.04, and
    everything else comes from ``GRPO_COMMON``, which is how the single-variable
    comparison is enforced rather than merely intended.
    """
    from trl import GRPOConfig

    settings = {**GRPO_COMMON, "beta": beta, "max_steps": max_steps, "output_dir": output_dir}
    settings.update(overrides)
    return GRPOConfig(**settings)


def describe_config(config, extra_keys: tuple[str, ...] = ()) -> str:
    """The hyperparameters that matter, as a printable block. Nothing silent."""
    keys = (
        "beta",
        "learning_rate",
        "temperature",
        "num_generations",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_prompt_length",
        "max_completion_length",
        "max_steps",
        "fp16",
        "bf16",
        "use_vllm",
        "lr_scheduler_type",
        "warmup_steps",
        "seed",
    ) + extra_keys
    width = max(len(key) for key in keys)
    lines = [f"  {key:<{width}}  {getattr(config, key, '(unset)')}" for key in keys]
    effective = config.per_device_train_batch_size * config.gradient_accumulation_steps
    lines.append(f"  {'prompts / optimiser step':<{width}}  {effective // config.num_generations}")
    lines.append(f"  {'completions / optimiser step':<{width}}  {effective}")
    return "\n".join(lines)


def diff_configs(a, b, keys: tuple[str, ...] = ()) -> dict[str, tuple[Any, Any]]:
    """Fields where two GRPOConfigs disagree. Act 4 prints this to prove it changed one line."""
    keys = keys or tuple(GRPO_COMMON) + ("beta", "max_steps")
    return {
        key: (getattr(a, key, None), getattr(b, key, None))
        for key in keys
        if getattr(a, key, None) != getattr(b, key, None)
    }
