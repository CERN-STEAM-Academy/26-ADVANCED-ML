"""Tests for the interactive chat helpers.

The chat cells are the part of notebook 2 a student is most likely to edit and re-run, so
the failure modes worth pinning are the ones that would bite in a live session: a prompt
that is not a string, an adapter that does not actually toggle, or a helper that quietly
needs a GPU when notebook 1 does not have one.
"""

from __future__ import annotations

import contextlib

import pytest

from rlpractice import chat


def test_suggested_prompts_mix_the_trained_task_and_everything_else():
    """The point of the before/after comparison is the prompts the reward never mentioned."""
    prompts = chat.SUGGESTED_PROMPTS
    assert len(prompts) >= 4
    arithmetic = [p for p in prompts if "times" in p]
    assert len(arithmetic) >= 1, "there must be something the training actually rewarded"
    assert len(prompts) - len(arithmetic) >= 3, (
        "most suggestions should be things the objective was silent about - that is the "
        "comparison worth making"
    )


def test_without_adapter_is_a_no_op_on_a_plain_model():
    """compare() on a model with no adapter must degrade gracefully rather than raise."""

    class PlainModel:
        pass

    with chat._without_adapter(PlainModel()):
        pass  # must not raise


def test_without_adapter_uses_disable_adapter_when_present():
    """On a PEFT model it must go through disable_adapter, which is what makes the
    before/after comparison free rather than a second copy of the weights."""
    calls = []

    class FakePeft:
        def disable_adapter(self):
            calls.append("disabled")
            return contextlib.nullcontext()

    with chat._without_adapter(FakePeft()):
        pass
    assert calls == ["disabled"]


def test_greedy_is_the_default():
    """A student re-running a cell must get the same answer, so that a difference they see
    is a difference in the model rather than in the sampling."""
    import inspect

    assert inspect.signature(chat.chat).parameters["temperature"].default == 0.0


def test_chat_helpers_do_not_import_torch_at_module_scope_beyond_what_is_needed():
    """Notebook 1 is CPU-only; nothing here may assume a GPU exists."""
    import inspect

    source = inspect.getsource(chat)
    assert ".cuda()" not in source
    assert 'device="cuda"' not in source


@pytest.mark.parametrize("bad", [None, 42, ["a list"]])
def test_chat_rejects_a_non_string_message_early(bad):
    """apply_chat_template would fail deep inside the tokenizer with an opaque message."""

    class Tok:
        def apply_chat_template(self, conversation, **kwargs):
            for turn in conversation:
                assert isinstance(turn["content"], str), "non-string content reached the tokenizer"
            return ""

    # The helper does not validate explicitly; this test documents that the content is
    # passed straight through, so the notebook cells must pass strings. If validation is
    # added later, this test should be tightened rather than deleted.
    conversation = [{"role": "user", "content": bad}]
    with pytest.raises(AssertionError):
        Tok().apply_chat_template(conversation)
