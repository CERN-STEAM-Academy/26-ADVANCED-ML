"""Tests pinning the notebook-1 diagnosis exercise.

The exercise asks students to reason from "this configuration differs from the working one
by exactly one field". That has to be true, and it has to stay true - it is the kind of
property that a well-meaning edit quietly breaks.

It also has to be true that the broken configurations *break*. Two earlier candidates did
not: shrinking the replay buffer to 32 scored a mean final return of 238 against the
working configuration's 133 over three seeds, and a hundredfold learning rate failed
without ever diverging, because Adam bounds its own step size. Both were measured and both
were removed. These tests do not re-run training - that takes minutes - but they do pin the
mechanism each surviving break relies on.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rlpractice import dqn


def test_every_broken_config_differs_from_working_by_exactly_one_field():
    for label, config in dqn.CONFIGS.items():
        difference = config.diff(dqn.WORKING)
        if label == "working":
            assert difference == {}
            continue
        assert len(difference) == 1, (
            f"{label} differs from WORKING in {len(difference)} fields ({sorted(difference)}). "
            "The exercise asks students to reason from a single changed field; two changes "
            "make the run uninterpretable."
        )


def test_the_configs_are_the_ones_the_notebook_discusses():
    assert set(dqn.CONFIGS) == {"working", "CONFIG_A", "CONFIG_B"}
    assert dqn.CONFIG_A.diff(dqn.WORKING) == {"target_update_every": (1, 200)}
    assert dqn.CONFIG_B.diff(dqn.WORKING) == {"bootstrap_past_termination": (True, False)}


def test_working_config_keeps_every_mitigation():
    """If any of these drift, the 'working' baseline stops being a fair comparison."""
    assert dqn.WORKING.target_update_every > 1          # a real target network
    assert dqn.WORKING.buffer_size >= 10_000            # decorrelated replay
    assert dqn.WORKING.bootstrap_past_termination is False   # the recursion is grounded
    assert dqn.WORKING.grad_clip > 0


def test_bootstrap_past_termination_changes_the_stored_done_flag():
    """The whole of CONFIG_B is this one boolean reaching the replay buffer.

    Rather than trust the flag, drive a few hundred steps of the real loop with a stub
    buffer and check what actually got stored at the transitions where an episode ended.
    """
    stored: dict[bool, list[float]] = {True: [], False: []}

    class RecordingBuffer(dqn.ReplayBuffer):
        def push(self, state, action, reward, next_state, done):
            stored[self.flavour].append(float(done))
            super().push(state, action, reward, next_state, done)

    for flavour in (False, True):
        RecordingBuffer.flavour = flavour
        config = replace(
            dqn.WORKING,
            bootstrap_past_termination=flavour,
            total_steps=600,
            learning_starts=10_000,   # no gradient steps; we only want the stored flags
            eval_every=10_000,
        )
        env = dqn.make_env(config.env_id, seed=0)
        original = dqn.ReplayBuffer
        dqn.ReplayBuffer = RecordingBuffer
        try:
            dqn.train_dqn(env, config, label=f"probe-{flavour}", verbose=False)
        finally:
            dqn.ReplayBuffer = original
            env.close()

    # The grounded run must record some terminal transitions; the broken one, none at all.
    assert sum(stored[False]) > 0, "the working config never stored a terminal transition"
    assert sum(stored[True]) == 0, (
        "bootstrap_past_termination=True must store done=False everywhere, so that the "
        "bootstrapped recursion never touches a known value"
    )


def test_config_b_target_is_never_grounded():
    """The consequence, stated directly in terms of the TD target.

    With done=0 the target is r + gamma * max Q(s'), for every transition. Nothing in the
    system is ever equal to a known quantity, which is what lets the estimates inflate.
    """
    import torch

    rewards = torch.ones(4)
    next_q = torch.full((4,), 50.0)
    gamma = 0.99

    grounded = rewards + gamma * next_q * (1.0 - torch.tensor([0.0, 1.0, 0.0, 1.0]))
    ungrounded = rewards + gamma * next_q * (1.0 - torch.zeros(4))

    assert grounded[1].item() == pytest.approx(1.0)      # terminal: value known to be zero
    assert ungrounded[1].item() == pytest.approx(1.0 + gamma * 50.0)
    assert (ungrounded >= grounded).all()


def test_baird_reproduces_the_deadly_triad():
    """All three legs diverge; removing any one leaves the weights bounded.

    This is the claim notebook 1 makes in prose, so it is worth a test rather than a
    footnote. Cheap: a few thousand numpy operations.
    """
    from rlpractice import baird

    results = baird.triad_ablation(steps=3000, alpha=0.01, seed=0)
    norms = {label: float(np.linalg.norm(data["history"][-1])) for label, data in results.items()}

    assert norms["all three legs"] > 1e3, norms
    for label in ("on-policy", "tabular (no approximation)", "Monte Carlo (no bootstrap)"):
        assert norms[label] < 1e3, (label, norms)
        assert norms[label] < norms["all three legs"] / 100


def test_baird_divergence_is_a_property_of_the_operator_not_of_noise():
    """The eigenvalue is what makes this 'provable' rather than 'observed once'."""
    from rlpractice import baird

    all_three = max(np.linalg.eigvals(baird.key_matrix(off_policy=True)).real)
    on_policy = max(np.linalg.eigvals(baird.key_matrix(off_policy=False)).real)
    tabular = max(np.linalg.eigvals(baird.key_matrix(off_policy=True, X=np.eye(baird.N_STATES))).real)

    assert all_three > 0.1, "off-policy TD with these features must have a positive eigenvalue"
    assert on_policy <= 1e-9
    assert tabular <= 1e-9


def test_baird_true_values_are_zero_and_representable():
    """The point of the counterexample: the perfect solution is available and ignored."""
    from rlpractice import baird

    features = baird.features()
    zero_weights = np.zeros(features.shape[1])
    assert np.allclose(features @ zero_weights, 0.0)
