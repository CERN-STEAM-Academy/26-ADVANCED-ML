"""FROZEN evaluation code. Write once, never parameterise, never let a later act drift.

Why this module is frozen
-------------------------
Acts 0, 2 and 4 all make the same three measurements, and the entire argument of notebook
2 rests on those measurements being comparable. If Act 2 quietly used a different decoding
temperature, or a different number of eval problems, or a different prose corpus, the
before/after comparison would measure the change in the evaluator rather than the change
in the model - and it would do so silently, which is the worst kind of wrong.

So: every knob that could drift is a module-level constant, the decoding is greedy, the
seeds are fixed, and the functions take no configuration arguments beyond the model and
the data. If you find yourself wanting to add a parameter here, add a new function
instead.

The three measurements
----------------------
* ``eval_task_accuracy``       - did it learn the thing we rewarded?
* ``eval_general_perplexity``  - what did that cost, measured cheaply?
* ``sample_general_generations`` - what did that cost, measured legibly?

The perplexity probe is one forward pass over an embedded corpus. No generation, no
download. That is what makes it cheap enough to call every few steps *during* training,
which is what makes the scissors plot possible at all.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any

import torch

from .arithmetic import extract_answer_int
from .general_text import PASSAGES

# --- frozen configuration --------------------------------------------------------

#: Every seed in this module. Greedy decoding makes generation deterministic anyway; the
#: seed is belt and braces against any sampling that sneaks in later.
EVAL_SEED = 1234

#: Exactly five general prompts, fixed forever. Instruction-following prose, deliberately
#: as far from tagged integer multiplication as the model's repertoire allows: no digits,
#: no arithmetic, no output contract to satisfy. If these degrade, it is not because the
#: model got confused about the task format.
GENERAL_PROMPTS: list[str] = [
    "Write a short paragraph explaining why sea cliffs retreat over time.",
    "Give three pieces of practical advice to someone starting a vegetable garden.",
    "Explain the difference between a simile and a metaphor, with one example of each.",
    "Describe what the daily routine of a lighthouse keeper might have looked like.",
    "In plain language, explain why bread dough needs to rest before it is baked.",
]

#: Decoding budget for the task-accuracy probe. Matches ``max_completion_length`` in
#: training, so eval and training see the model under the same length constraint.
TASK_MAX_NEW_TOKENS = 128
#: Decoding budget for the qualitative general-prose samples. Long enough for degradation
#: to be visible, short enough that five of them cost a couple of seconds.
GENERAL_MAX_NEW_TOKENS = 96

#: Fixed batch sizes. Changing a batch size changes fp16 reduction order and therefore
#: changes the numbers in the last decimal place, so these are frozen too.
TASK_BATCH_SIZE = 16
GENERAL_BATCH_SIZE = 5
PPL_BATCH_SIZE = 1

#: Autocast dtype for all evaluation. fp16, not bf16: Turing (sm75) has no bf16.
EVAL_AUTOCAST_DTYPE = torch.float16


@contextlib.contextmanager
def _eval_mode(model):
    """Put a model into a safe, fast, side-effect-free state for evaluation.

    This has to work when called from a training callback halfway through a run, so it
    restores everything it touches:

    * ``model.eval()`` - disables dropout, which would otherwise add noise to a
      measurement we are about to plot as a trend.
    * ``use_cache = True`` - gradient checkpointing turns the KV cache off, and
      generating without a cache is slow enough to blow the time budget.
    * ``torch.no_grad()`` - no graph, and a large amount of memory not spent.
    """
    was_training = model.training
    config = getattr(model, "config", None)
    prev_use_cache = getattr(config, "use_cache", None) if config is not None else None
    model.eval()
    if config is not None:
        config.use_cache = True
    try:
        with torch.no_grad():
            yield model
    finally:
        if config is not None and prev_use_cache is not None:
            config.use_cache = prev_use_cache
        if was_training:
            model.train()


@contextlib.contextmanager
def _fixed_seed(seed: int):
    """Seed the RNG for a measurement, then put the caller's RNG stream back.

    This matters far more than it looks. ``eval_task_accuracy`` is called from a training
    callback every ``acc_every`` steps, and TRL's GRPO samples its completions with
    ``torch.multinomial`` off the *global* RNG. Seeding without restoring therefore resets
    the exploration noise of the live training run to the same state every twentieth step,
    so the policy keeps re-drawing the same sequence of samples - a subtle, silent and
    entirely avoidable corruption of the run being measured. Greedy decoding consumes no
    random draws at all, so without this the caller's stream is left sitting exactly where
    the seed put it.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextlib.contextmanager
def _left_padding(tokenizer):
    """Batched generation needs left padding, or the model continues from pad tokens."""
    previous = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        yield tokenizer
    finally:
        tokenizer.padding_side = previous


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=EVAL_AUTOCAST_DTYPE)
    return contextlib.nullcontext()


def _generate_greedy(model, tokenizer, chats, max_new_tokens, batch_size) -> list[str]:
    """Greedy-decode a list of chat conversations, returning only the new text."""
    device = _device_of(model)
    outputs: list[str] = []

    with _fixed_seed(EVAL_SEED), _eval_mode(model), _left_padding(tokenizer):
        for start in range(0, len(chats), batch_size):
            batch = chats[start : start + batch_size]
            rendered = [
                tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                for chat in batch
            ]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            prompt_len = encoded["input_ids"].shape[1]
            with _autocast(device):
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,          # greedy: the eval must not be stochastic
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = generated[:, prompt_len:]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


def generate_greedy(model, tokenizer, chats, max_new_tokens: int = TASK_MAX_NEW_TOKENS, batch_size: int = TASK_BATCH_SIZE) -> list[str]:
    """Public wrapper around the greedy decoder, for looking at raw model output.

    The notebooks use this in Act 0 to show what the base model actually writes, before
    any metric is computed. Defaults match the task-accuracy probe so that what you read
    is what that probe scored.
    """
    return _generate_greedy(model, tokenizer, list(chats), max_new_tokens, batch_size)


def eval_task_accuracy(model, tokenizer, eval_dataset, n: int = 64) -> float:
    """Greedy pass@1 on the first ``n`` held-out problems.

    Greedy rather than sampled, so the number is a property of the model and not of the
    random seed. The first ``n`` rather than a random ``n``, so that a smaller ``n`` used
    during training is a strict prefix of the larger ``n`` used at the endpoints and the
    two are still comparable.
    """
    n = min(n, len(eval_dataset))
    if n == 0:
        raise ValueError("eval_task_accuracy needs a non-empty eval_dataset")
    subset = eval_dataset.select(range(n))
    completions = _generate_greedy(
        model, tokenizer, list(subset["prompt"]), TASK_MAX_NEW_TOKENS, TASK_BATCH_SIZE
    )
    correct = sum(
        1
        for completion, answer in zip(completions, subset["answer"])
        if extract_answer_int(completion) == int(answer)
    )
    return correct / n


#: Group size and temperature for the sampled sweep below. These deliberately mirror the
#: training configuration (``num_generations``, ``temperature``): the whole point is to
#: measure the distribution GRPO will actually sample from.
SWEEP_GROUP_SIZE = 8
SWEEP_TEMPERATURE = 1.0
SWEEP_SEED = 4321
#: TRL 0.15.2 builds its ``GenerationConfig`` without setting ``top_k``, so it inherits the
#: transformers class default of 50: GRPO samples from the top-50 renormalised
#: distribution, not from the policy itself. The sweep mirrors that, because its entire
#: purpose is to measure the distribution training will actually draw from.
SWEEP_TOP_K = 50


def eval_sampled_pass_rate(model, tokenizer, eval_dataset, n: int = 16) -> dict[str, float]:
    r"""Pass rate and group variance under GRPO's *sampling* distribution.

    Greedy pass@1 is the number everyone quotes, and for choosing a GRPO training
    difficulty it is the wrong one.

    GRPO does not see greedy decodes. It draws :math:`G` completions per prompt at
    ``temperature=1.0`` and normalises the rewards within that group:

    .. math:: A_i = \frac{r_i - \mathrm{mean}(r_{1..G})}{\mathrm{std}(r_{1..G}) + \epsilon}

    If all :math:`G` completions in a group score the same - all right, or all wrong - the
    numerator is zero for every member, the advantage is *exactly* zero, and that prompt
    contributes no gradient whatsoever. A task where the base model is at 5% or at 95%
    produces such groups almost every time, and training silently becomes an expensive
    no-op: the loss still looks fine, the run still takes an hour, and the model does not
    move.

    So this returns three numbers:

    * ``sampled_pass_rate`` - fraction of sampled completions that are correct;
    * ``zero_variance_groups`` - fraction of groups whose rewards are all identical, which
      is the fraction of the batch that is doing nothing;
    * ``mean_reward`` - format plus correctness, for reference.
    """
    # Scored here from the parsing primitives rather than by importing the reward
    # functions. Those are a student exercise, and in a distribution where they still
    # raise NotImplementedError - or where a student has them subtly wrong - this Act 0
    # measurement must still be correct, because it is what chooses the task to train on.
    from .arithmetic import FORMAT_RE, extract_answer_int, extract_answer_text

    def _score(text: str, answer: str) -> tuple[float, float]:
        if FORMAT_RE.match(text) is not None:
            fmt = 1.0
        elif extract_answer_text(text) is not None:
            fmt = 0.5
        else:
            fmt = 0.0
        predicted = extract_answer_int(text)
        return fmt, (2.0 if predicted is not None and predicted == int(answer) else 0.0)

    n = min(n, len(eval_dataset))
    subset = eval_dataset.select(range(n))
    device = _device_of(model)
    group = SWEEP_GROUP_SIZE

    completions: list[str] = []
    with _fixed_seed(SWEEP_SEED), _eval_mode(model), _left_padding(tokenizer):
        for start in range(0, n, TASK_BATCH_SIZE):
            batch = list(subset["prompt"])[start : start + TASK_BATCH_SIZE]
            rendered = [
                tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                for chat in batch
            ]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            prompt_len = encoded["input_ids"].shape[1]
            with _autocast(device):
                generated = model.generate(
                    **encoded,
                    max_new_tokens=TASK_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=SWEEP_TEMPERATURE,
                    top_p=1.0,
                    top_k=SWEEP_TOP_K,
                    num_return_sequences=group,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completions.extend(tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True))

    # generate() with num_return_sequences returns the group for prompt i contiguously.
    answers = [answer for answer in subset["answer"] for _ in range(group)]
    scored = [_score(text, answer) for text, answer in zip(completions, answers)]
    corrects = [c for _, c in scored]
    totals = [f + c for f, c in scored]

    n_zero_variance = 0
    for i in range(n):
        rewards = totals[i * group : (i + 1) * group]
        if max(rewards) == min(rewards):
            n_zero_variance += 1

    return {
        "sampled_pass_rate": sum(1 for c in corrects if c > 0) / len(corrects),
        "zero_variance_groups": n_zero_variance / n,
        "mean_reward": sum(totals) / len(totals),
        "n_prompts": n,
        "group_size": group,
    }


def eval_general_perplexity(model, tokenizer) -> float:
    """Token-level perplexity on the embedded general-prose corpus.

    One forward pass per passage, no generation, no download. Returns
    ``exp(total NLL / total tokens)`` over the whole corpus - pooled rather than averaged
    per passage, so that a long passage counts for more than a short one, which is what
    "perplexity of the corpus" conventionally means.
    """
    device = _device_of(model)
    total_nll = 0.0
    total_tokens = 0

    with _eval_mode(model):
        for passage in PASSAGES:
            input_ids = tokenizer(passage, return_tensors="pt").input_ids.to(device)
            with _autocast(device):
                logits = model(input_ids=input_ids).logits
            # Standard causal-LM shift: predict token t+1 from everything up to t.
            shift_logits = logits[:, :-1, :].float()
            shift_labels = input_ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += shift_labels.numel()

    return float(torch.exp(torch.tensor(total_nll / total_tokens)))


def sample_general_generations(model, tokenizer) -> list[str]:
    """Greedy completions of the five frozen general prompts.

    No judge model. The degradation this is meant to expose should be obvious to a naive
    reader printed side by side, and if it is not obvious then the run has not
    demonstrated what the session claims it demonstrates.
    """
    chats = [[{"role": "user", "content": prompt}] for prompt in GENERAL_PROMPTS]
    return _generate_greedy(model, tokenizer, chats, GENERAL_MAX_NEW_TOKENS, GENERAL_BATCH_SIZE)


def snapshot(
    model,
    tokenizer,
    eval_dataset,
    tag: str,
    out_dir: str | os.PathLike = "assets/snapshots",
    n_task: int = 64,
) -> dict[str, Any]:
    """Run all three frozen evals, save to ``{out_dir}/{tag}.json``, and return the dict.

    ``tag`` is what makes a snapshot comparable to another one: "before", "after_act1",
    "after_act4". Saving to disk means a kernel restart does not destroy the baseline
    the rest of the notebook is comparing against.
    """
    os.makedirs(out_dir, exist_ok=True)
    result = {
        "tag": tag,
        "task_accuracy": eval_task_accuracy(model, tokenizer, eval_dataset, n=n_task),
        "general_perplexity": eval_general_perplexity(model, tokenizer),
        "general_generations": sample_general_generations(model, tokenizer),
        "general_prompts": GENERAL_PROMPTS,
        "n_task": min(n_task, len(eval_dataset)),
        "eval_seed": EVAL_SEED,
    }
    path = os.path.join(out_dir, f"{tag}.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=2)
    result["path"] = path
    return result


def load_snapshot(tag: str, out_dir: str | os.PathLike = "assets/snapshots") -> dict[str, Any]:
    """Read a snapshot back. Used by the ``TRAIN_FROM_SCRATCH = False`` path."""
    with open(os.path.join(out_dir, f"{tag}.json")) as handle:
        return json.load(handle)


def print_snapshot(result: dict[str, Any], show_generations: bool = True) -> None:
    """Human-readable dump of a snapshot. Silent tensors teach nothing."""
    print(f"=== snapshot: {result['tag']} ===")
    print(f"  task accuracy (n={result['n_task']}, greedy) : {result['task_accuracy']:.3f}")
    print(f"  general-text perplexity                      : {result['general_perplexity']:.3f}")
    if show_generations:
        for prompt, generation in zip(result["general_prompts"], result["general_generations"]):
            print(f"\n  prompt: {prompt}")
            print(f"  ----> {generation.strip()[:400]}")


def compare_snapshots(before: dict[str, Any], after: dict[str, Any], width: int = 78) -> None:
    """Print the five general generations before and after, one prompt at a time.

    Stacked rather than in two columns: two narrow columns of wrapped prose in a terminal
    are unreadable, and the point here is that a naive reader can see the difference.
    """
    import textwrap

    print("=" * width)
    print(f"BEFORE ({before['tag']})  vs  AFTER ({after['tag']})")
    print(
        f"  task accuracy : {before['task_accuracy']:.3f}  ->  {after['task_accuracy']:.3f}"
        f"   ({after['task_accuracy'] - before['task_accuracy']:+.3f})"
    )
    print(
        f"  general ppl   : {before['general_perplexity']:.3f}  ->  {after['general_perplexity']:.3f}"
        f"   (x{after['general_perplexity'] / before['general_perplexity']:.2f})"
    )
    print("=" * width)

    for i, prompt in enumerate(before["general_prompts"]):
        print(f"\n[{i + 1}/5] {prompt}")
        for label, snap in (("BEFORE", before), ("AFTER ", after)):
            text = snap["general_generations"][i].strip() or "(empty)"
            wrapped = textwrap.fill(text, width=width - 10, initial_indent="", subsequent_indent=" " * 10)
            print(f"  {label}  {wrapped}")
        print("-" * width)
