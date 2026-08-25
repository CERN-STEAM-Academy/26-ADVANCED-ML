"""Talk to the model, before and after training, in the same cell.

Every other measurement in notebook 2 is a number. This is the one place where you type
whatever you like and read what comes back, which is a different and necessary kind of
evidence: a perplexity of 29.6 against 27.6 is an argument, and watching a model that used
to write English answer your question with ``<answer> 42 </answer>`` is a demonstration.

The nice trick is that comparing before and after costs no extra memory. The trained model
is a LoRA adapter sitting on top of frozen base weights, so ``disable_adapter()`` gives you
the original model back exactly, in the same process, with no second copy and no reloading.
The same mechanism TRL uses internally to compute the reference log-probabilities for the
KL penalty is the one that lets you A/B the model against its own past self here - and it
is also the answer to the second discussion question in Act 5.

Greedy decoding by default, so that running a cell twice gives the same answer and a
difference you see is a difference in the model rather than in the sampling.
"""

from __future__ import annotations

import textwrap

import torch

from .evaluation import _autocast, _device_of, _eval_mode, _left_padding

#: Long enough for a real answer, short enough that a cell returns in a couple of seconds.
DEFAULT_MAX_NEW_TOKENS = 192


def chat(
    model,
    tokenizer,
    message: str,
    system: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    seed: int = 0,
) -> str:
    """Send one message and return the reply.

    ``temperature=0`` means greedy decoding, which is the default because a reproducible
    answer is worth more here than a varied one. Pass a positive temperature if you want to
    see the spread of what the model might say - that is the distribution GRPO samples from,
    and after training it is often startlingly narrow.
    """
    device = _device_of(model)
    conversation = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": message}
    ]

    with _eval_mode(model), _left_padding(tokenizer):
        rendered = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]

        sampling = (
            {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
            if temperature <= 0
            else {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 50}
        )
        if temperature > 0:
            torch.manual_seed(seed)

        with _autocast(device):
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                **sampling,
            )

    return tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()


def _without_adapter(model):
    """Context manager giving the base model back, or a no-op if this is not a PEFT model."""
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    import contextlib

    return contextlib.nullcontext()


def compare(
    model,
    tokenizer,
    message: str,
    system: str | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.0,
    width: int = 78,
    show: bool = True,
) -> tuple[str, str]:
    """Answer the same message with the adapter off and on, and print both.

    ``model`` should be the trained (PEFT) model. Returns ``(before, after)``.

    If you pass a plain model with no adapter, both halves are the same model and the
    comparison is trivially empty - which is itself a useful thing to see happen once.
    """
    with _without_adapter(model):
        before = chat(model, tokenizer, message, system, max_new_tokens, temperature)
    after = chat(model, tokenizer, message, system, max_new_tokens, temperature)

    if show:
        print("=" * width)
        print(textwrap.fill(f"YOU: {message}", width=width))
        print("=" * width)
        for label, reply in (("BEFORE training", before), ("AFTER training ", after)):
            body = reply or "(the model returned nothing at all)"
            wrapped = textwrap.fill(
                body, width=width, initial_indent="", subsequent_indent=" " * 17
            )
            print(f"\n{label}: {wrapped}")
        print()

    return before, after


def ask_both(model, tokenizer, messages, **kwargs) -> list[tuple[str, str]]:
    """``compare`` over several messages. Returns the list of (before, after) pairs."""
    return [compare(model, tokenizer, message, **kwargs) for message in messages]


#: Suggested things to try. Deliberately a mix: two arithmetic questions the training
#: rewarded, and several that it never mentioned. The interesting ones are the latter.
SUGGESTED_PROMPTS: list[str] = [
    "What is 47 times 6?",
    "What is 1234 times 5678?",
    "What is the capital of Portugal?",
    "Write a two-sentence bedtime story about a fox.",
    "I have a headache and a deadline. Any advice?",
    "Translate 'good morning' into French.",
]
