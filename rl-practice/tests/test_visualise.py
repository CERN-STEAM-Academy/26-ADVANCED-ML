"""Tests for the CartPole episode viewer in :mod:`rlpractice.dqn`.

The viewer exists so that students can *watch* a policy rather than only read its return,
which means its failure mode is a cell that renders nothing and says nothing. These tests
are therefore mostly about the boring guarantees: a random rollout produces frames without
needing a network, an extreme observation does not blow up the drawing code, and the
animation object really does carry the ``to_jshtml`` method the notebook embeds it with.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display on the training machines, and none needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from rlpractice import dqn  # noqa: E402


@pytest.fixture
def env():
    environment = dqn.make_env("CartPole-v1", seed=0)
    yield environment
    environment.close()


def test_random_rollout_needs_no_network(env):
    """``q_network=None`` is the untrained baseline and must not require any weights."""
    observations, episode_return = dqn.rollout_frames(env, None, max_steps=500, seed=0)

    assert len(observations) > 1
    assert all(np.asarray(o).shape == (4,) for o in observations)
    assert episode_return > 0
    # One reward of +1 per step survived, plus the terminal frame that earned nothing.
    assert len(observations) == int(episode_return) + 1


def test_rollout_with_a_network_runs(env):
    """An untrained ``QNetwork`` is still a policy: greedy, deterministic, and short-lived."""
    network = dqn.QNetwork(4, 2, hidden=16)
    observations, episode_return = dqn.rollout_frames(env, network, max_steps=50, seed=0)

    assert len(observations) > 1
    assert episode_return > 0
    assert len(observations) <= 51


def test_rollout_is_reproducible_for_a_fixed_seed(env):
    """Same seed, same episode - otherwise "before and after" compares two different worlds."""
    first, first_return = dqn.rollout_frames(env, None, seed=7)
    second, second_return = dqn.rollout_frames(env, None, seed=7)

    assert first_return == second_return
    assert np.allclose(np.asarray(first), np.asarray(second))


def test_draw_cartpole_survives_an_extreme_observation():
    """Well past both failure thresholds, with velocities that never occur in practice."""
    figure, ax = plt.subplots()
    try:
        dqn.draw_cartpole(ax, np.array([-9.0, 42.0, 1.4, -37.0], dtype=np.float32), title="way off")
        x_low, x_high = ax.get_xlim()
        threshold, _, _ = dqn.cartpole_geometry()
        # The limits stay pinned to the track, so an off-track cart is visibly off-track
        # rather than quietly rescaling the picture until it looks fine.
        assert x_low > -2.0 * threshold and x_high < 2.0 * threshold
    finally:
        plt.close(figure)


def test_cartpole_geometry_matches_gymnasium():
    """Read from ``CartPoleEnv``, not guessed: 2.4 metres and 12 degrees."""
    x_threshold, theta_threshold, half_length = dqn.cartpole_geometry()

    assert x_threshold == pytest.approx(2.4)
    assert np.degrees(theta_threshold) == pytest.approx(12.0)
    assert half_length == pytest.approx(0.5)


def test_filmstrip_has_one_axis_per_requested_frame(env):
    observations, _ = dqn.rollout_frames(env, None, seed=0)
    figure = dqn.filmstrip(observations, n_frames=6, title="random policy")
    try:
        assert len(figure.axes) == 6
    finally:
        plt.close(figure)


def test_filmstrip_copes_with_an_episode_shorter_than_the_strip(env):
    """Fewer states than frames still yields the requested number of axes, with repeats."""
    observations, _ = dqn.rollout_frames(env, None, max_steps=3, seed=0)
    figure = dqn.filmstrip(observations, n_frames=6)
    try:
        assert len(figure.axes) == 6
    finally:
        plt.close(figure)


def test_animate_episode_returns_something_embeddable(env):
    """The notebook embeds the animation as ``HTML(anim.to_jshtml())`` and nothing else."""
    observations, _ = dqn.rollout_frames(env, None, seed=0)
    anim = dqn.animate_episode(observations, title="random", max_frames=4)

    assert hasattr(anim, "to_jshtml")
    html = anim.to_jshtml()
    assert "<img" in html or "base64" in html
    plt.close(anim._fig)


def test_compare_episodes_animates_two_agents_in_lockstep():
    """``None`` and a ``DQNResult`` side by side is the before-and-after the notebook shows."""
    result = dqn.DQNResult(label="toy", config=dqn.WORKING)
    result.state_dict = {k: v.cpu() for k, v in dqn.QNetwork(4, 2, hidden=8).state_dict().items()}

    anim = dqn.compare_episodes(
        [None, result], ["random", "toy"], max_steps=25, max_frames=3
    )
    try:
        assert hasattr(anim, "to_jshtml")
        assert len(anim._fig.axes) == 2
    finally:
        plt.close(anim._fig)


def test_compare_episodes_can_fall_back_to_stills():
    """The filmstrip form is the fallback for anywhere the player will not embed."""
    figure = dqn.compare_episodes(
        [None, None], ["random", "random again"], max_steps=25, as_filmstrip=True, n_frames=4
    )
    try:
        assert len(figure.axes) == 8
    finally:
        plt.close(figure)


def test_a_result_without_weights_is_a_loud_error():
    """A ``DQNResult`` loaded without a ``state_dict`` must not silently become random."""
    empty = dqn.DQNResult(label="no weights", config=dqn.WORKING)
    with pytest.raises(ValueError, match="no weights"):
        dqn.compare_episodes([empty], ["nothing"], max_steps=5)
