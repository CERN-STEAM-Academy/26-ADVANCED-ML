"""DQN for notebook 1, provided complete so that it can be *diagnosed* rather than written.

Why you are given this code
---------------------------
Writing DQN from scratch is a pleasant afternoon and teaches you mostly about PyTorch.
Reading a working DQN, breaking one thing, and predicting the shape of the resulting
failure teaches you about the deadly triad, which is the part that transfers to every
other value-based method you will ever meet.

The deadly triad is the claim that combining

1. **bootstrapping** - updating an estimate towards another estimate,
2. **off-policy learning** - learning about a policy other than the one collecting data,
3. **function approximation** - representing values with a parametric model,

can diverge, even though any two of the three are safe.

Notebook 1 demonstrates that claim twice, in two different registers. Baird's
counterexample (:mod:`rlpractice.baird`) shows it *provably*, in a seven-state problem with
an eigenvalue you can compute. This file shows it *happening*, to a real agent, in a way
that is messier and more like the thing you will actually debug.

The broken configurations here each remove one safeguard that keeps the bootstrapped
target grounded, and each differs from ``WORKING`` by exactly one field. Every mitigation
in this file is labelled in the code, because the exercise is to find the one that was
removed.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------


@dataclass
class DQNConfig:
    """Every knob of the agent in one object, so that a broken run differs by one field.

    The three diagnosis configs in notebook 1 are built with ``dataclasses.replace`` from
    ``WORKING`` below, which is what makes "exactly one change" a fact rather than a
    claim: you can print the diff.
    """

    env_id: str = "CartPole-v1"
    total_steps: int = 12_000
    seed: int = 0

    # --- optimisation
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    hidden: int = 128
    grad_clip: float = 10.0

    # --- mitigation 1: a target network, updated slowly, so the bootstrap target is
    #     approximately fixed while the online network chases it.
    target_update_every: int = 200

    # --- mitigation 2: a large replay buffer, so that minibatches are approximately
    #     i.i.d. rather than a correlated slice of one trajectory.
    #
    #     Worth knowing, because notebook 1 used to have an exercise about it: on CartPole
    #     this mitigation cannot be shown to matter. Shrinking the buffer to 32, and
    #     separately sampling it in strict temporal order, were both measured over three
    #     seeds and neither underperformed the working configuration - the run-to-run
    #     spread of the final return is simply larger than the effect. A benchmark that
    #     cannot resolve an effect is not evidence that the effect is absent, and it is
    #     also not a good exercise, so the notebook demonstrates this leg with Baird's
    #     counterexample (rlpractice/baird.py) instead, where it is provable.
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    train_every: int = 1

    # --- mitigation 3: the bootstrap is grounded. When the pole falls, the episode is
    #     over and the value of the next state is zero by definition, not estimated. That
    #     is the base case of the whole recursion. Set this True and there is no base
    #     case: every value is defined in terms of another estimated value, and the
    #     estimates inflate without limit.
    bootstrap_past_termination: bool = False

    # --- exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 5_000

    # --- how many CPU threads torch may use for this run.
    #
    # This is a performance setting that turned out to matter enormously, so it is a
    # config field rather than something left to chance. The networks here are
    # 4 -> 128 -> 128 -> 2, and a minibatch is 64 rows. Operations that small are pure
    # overhead to parallelise: the synchronisation costs more than the arithmetic saves.
    #
    # torch defaults to one thread per core. Measured on a 28-core machine, one
    # configuration took 100 seconds at the default and 35 seconds pinned to four
    # threads - the same work, three times faster, because the other 24 threads were
    # spending their time coordinating. A student on a large Kubeflow node would hit
    # exactly this. Set to 0 to leave torch alone.
    torch_threads: int = 4

    # --- bookkeeping
    #
    # Evaluation, not training, is what makes a DQN run slow here. Twelve thousand
    # gradient steps on a two-layer MLP cost about 30 seconds; ten greedy episodes of a
    # *working* CartPole agent are 5000 environment steps each time, and at fifteen
    # evaluations that dominated everything else (measured: 343 s per run, of which
    # roughly 300 s was evaluation). Eight evaluations of five episodes is enough to draw
    # the curve and brings a run back under a minute and a half, which is what makes
    # running every configuration inside the exercise slot possible.
    eval_every: int = 1_500
    eval_episodes: int = 5
    device: str = "cpu"

    def diff(self, other: "DQNConfig") -> dict[str, tuple[Any, Any]]:
        """Fields where ``self`` and ``other`` disagree. Printed in the notebook."""
        return {
            name: (getattr(self, name), getattr(other, name))
            for name in self.__dataclass_fields__
            if getattr(self, name) != getattr(other, name)
        }


#: The configuration that works. Each broken one below is this, with exactly one field
#: changed - which you can print, rather than take on trust, with ``config.diff(WORKING)``.
WORKING = DQNConfig()

#: The target network follows the online network immediately, which is to say there is no
#: target network. Every time the online network moves, the thing it is being regressed
#: towards moves with it, so it chases its own tail. Measured over 60k steps on three
#: seeds: mean |Q| reaches about 6e9 and the return sits at the floor.
CONFIG_A = replace(WORKING, target_update_every=1)

#: The agent bootstraps past the end of the episode: the pole has fallen, but the target
#: is still ``r + gamma * max_a' Q(s', a')`` as though the episode continued. The Bellman
#: recursion now has no base case anywhere - every value is defined in terms of another
#: value and nothing is ever grounded in an observed outcome - so the estimates inflate
#: without limit. Measured over 60k steps on three seeds: mean |Q| around 2.5e8.
CONFIG_B = replace(WORKING, bootstrap_past_termination=True)

CONFIGS: dict[str, DQNConfig] = {
    "working": WORKING,
    "CONFIG_A": CONFIG_A,
    "CONFIG_B": CONFIG_B,
}


# ---------------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------------


class QNetwork(nn.Module):
    """A two-hidden-layer MLP mapping a state to one Q-value per action.

    Small on purpose. CartPole has a four-dimensional state and two actions, and a bigger
    network would only hide the instabilities we are trying to expose.
    """

    def __init__(self, n_observations: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class ReplayBuffer:
    """A fixed-capacity ring buffer of transitions, sampled uniformly at random.

    Two jobs, and it is worth being explicit that they are separate:

    * **Decorrelation.** Consecutive transitions in an episode are highly dependent.
      Gradient descent assumes something closer to i.i.d. samples, and a buffer that
      holds many episodes approximates that.
    * **Sample reuse.** Every transition is trained on many times, which is the whole
      economic argument for off-policy learning.

    Shrink the capacity to roughly one minibatch (``CONFIG_B``) and both jobs vanish at
    once: you are back to online, correlated, single-pass updates, with a bootstrapped
    target and a neural network still in the loop.
    """

    def __init__(self, capacity: int, n_observations: int, seed: int = 0):
        self.capacity = int(capacity)
        self.states = np.zeros((self.capacity, n_observations), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, n_observations), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def push(self, state, action, reward, next_state, done) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, ...]:
        """Uniform sample without replacement. Never larger than what we actually hold."""
        n = min(batch_size, self.size)
        idx = self.rng.choice(self.size, size=n, replace=False)
        to = lambda array, dtype: torch.as_tensor(array[idx], dtype=dtype, device=device)
        return (
            to(self.states, torch.float32),
            to(self.actions, torch.int64),
            to(self.rewards, torch.float32),
            to(self.next_states, torch.float32),
            to(self.dones, torch.float32),
        )


def epsilon_greedy(
    q_network: QNetwork,
    state: np.ndarray,
    epsilon: float,
    n_actions: int,
    rng: random.Random,
    device: str = "cpu",
) -> int:
    """Pick a uniformly random action with probability epsilon, else the greedy one.

    The simplest possible answer to exploration versus exploitation, and the reason DQN
    is *off-policy*: the data is collected by this epsilon-greedy behaviour policy, but
    the values being learned are those of the greedy target policy.
    """
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(q_network(state_tensor).argmax(dim=1).item())


def linear_epsilon(step: int, config: DQNConfig) -> float:
    """Linear decay from ``eps_start`` to ``eps_end`` over ``eps_decay_steps``."""
    fraction = min(1.0, step / max(1, config.eps_decay_steps))
    return config.eps_start + fraction * (config.eps_end - config.eps_start)


# ---------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------


@dataclass
class DQNResult:
    """Everything notebook 1 needs to plot and to diagnose a run."""

    label: str
    config: DQNConfig
    episode_steps: list[int] = field(default_factory=list)
    episode_returns: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    eval_returns: list[float] = field(default_factory=list)
    loss_steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    mean_abs_q: list[float] = field(default_factory=list)
    state_dict: dict | None = None
    wall_seconds: float = 0.0

    def final_eval(self) -> float:
        return self.eval_returns[-1] if self.eval_returns else float("nan")

    def best_eval(self) -> float:
        return max(self.eval_returns) if self.eval_returns else float("nan")

    def summary(self) -> str:
        return (
            f"{self.label:>9}: final eval return {self.final_eval():6.1f}, "
            f"best {self.best_eval():6.1f}, "
            f"max mean|Q| {max(self.mean_abs_q) if self.mean_abs_q else float('nan'):.3g}, "
            f"{self.wall_seconds:.0f} s"
        )


def make_env(env_id: str = "CartPole-v1", seed: int | None = None):
    """Construct a Gymnasium environment, seeded. Imported lazily so that importing
    ``rlpractice`` does not require Gymnasium for the GRPO notebook."""
    import gymnasium as gym

    env = gym.make(env_id)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def evaluate_policy(env, q_network: QNetwork, n_episodes: int = 10, seed: int = 12345, device: str = "cpu") -> float:
    """Mean return of the *greedy* policy. No exploration, fixed seeds.

    Reported separately from training returns because training returns are contaminated
    by epsilon-greedy exploration, and a curve that mixes the two is hard to read.

    ``env`` must **not** be the environment being trained in. This function resets it,
    and a training loop that shares one environment with its evaluator will resume
    stepping from a state the environment is no longer in - quietly writing garbage
    transitions into the replay buffer every time it evaluates.
    """
    total = 0.0
    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        while not done:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(q_network(state_tensor).argmax(dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            done = terminated or truncated
        total += episode_return
    return total / n_episodes


def train_dqn(env, config: DQNConfig, label: str = "dqn", verbose: bool = True) -> DQNResult:
    """Train a DQN agent. Fully implemented; the exercise is to break it and explain why.

    The loop is the textbook one:

    1. act epsilon-greedily in the environment and store the transition,
    2. sample a minibatch from the replay buffer,          <- mitigation: decorrelation
    3. form the bootstrapped target  r + gamma * max_a' Q_target(s', a') * (1 - done),
                                                            <- mitigation: target network
    4. regress the online Q(s, a) onto that target with a Huber loss,
    5. periodically copy the online weights into the target network.

    Read step 3 carefully. The target contains the network's own output. That is
    bootstrapping, and it is what makes all three mitigations necessary rather than
    merely nice.
    """
    import time

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    action_rng = random.Random(config.seed)

    previous_threads = torch.get_num_threads()
    if config.torch_threads:
        torch.set_num_threads(min(config.torch_threads, previous_threads))

    n_observations = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    device = config.device

    online = QNetwork(n_observations, n_actions, config.hidden).to(device)
    # The target network starts as an exact copy and is updated only every
    # `target_update_every` steps. With target_update_every = 1 it is not a target
    # network at all, it is the online network under a different name.
    target = copy.deepcopy(online).to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(online.parameters(), lr=config.lr)
    buffer = ReplayBuffer(config.buffer_size, n_observations, seed=config.seed)
    result = DQNResult(label=label, config=config)

    # A separate environment for evaluation. Sharing one environment between the
    # training loop and the evaluator means every evaluation resets the environment out
    # from under the loop, which then keeps stepping from a stale `state` and writes
    # transitions that never happened into the replay buffer.
    eval_env = make_env(config.env_id, seed=config.seed + 10_000)

    state, _ = env.reset(seed=config.seed)
    episode_return = 0.0
    started = time.time()

    for step in range(1, config.total_steps + 1):
        epsilon = linear_epsilon(step, config)
        action = epsilon_greedy(online, state, epsilon, n_actions, action_rng, device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)

        # `terminated` (the pole fell) bootstraps to zero; `truncated` (we hit the time
        # limit) does not, because the episode would have continued. Conflating the two
        # teaches the agent that surviving to the time limit is worth nothing.
        #
        # Storing done=False unconditionally (CONFIG_B) removes the only place where the
        # bootstrapped recursion touches something that is true by definition rather than
        # estimated.
        done_flag = False if config.bootstrap_past_termination else terminated
        buffer.push(state, action, reward, next_state, done_flag)
        state = next_state

        if terminated or truncated:
            result.episode_steps.append(step)
            result.episode_returns.append(episode_return)
            state, _ = env.reset()
            episode_return = 0.0

        if step >= config.learning_starts and step % config.train_every == 0:
            states, actions, rewards, next_states, dones = buffer.sample(config.batch_size, device)

            with torch.no_grad():
                next_q = target(next_states).max(dim=1).values
                td_target = rewards + config.gamma * next_q * (1.0 - dones)

            q_values = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_values, td_target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), config.grad_clip)
            optimizer.step()

            if step % 100 == 0:
                result.loss_steps.append(step)
                result.losses.append(float(loss.item()))
                result.mean_abs_q.append(float(q_values.abs().mean().item()))

        if step % config.target_update_every == 0:
            target.load_state_dict(online.state_dict())

        if step % config.eval_every == 0:
            mean_return = evaluate_policy(eval_env, online, config.eval_episodes, device=device)
            result.eval_steps.append(step)
            result.eval_returns.append(mean_return)
            if verbose:
                print(
                    f"  [{label}] step {step:6d}  eps {epsilon:.2f}  "
                    f"greedy eval return {mean_return:6.1f}  "
                    f"mean|Q| {result.mean_abs_q[-1] if result.mean_abs_q else float('nan'):8.3g}"
                )

    eval_env.close()
    torch.set_num_threads(previous_threads)
    result.wall_seconds = time.time() - started
    result.state_dict = {k: v.cpu() for k, v in online.state_dict().items()}
    return result


# ---------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------


def smooth(values: Sequence[float], window: int = 20) -> np.ndarray:
    """Trailing moving average. Raw episode returns on CartPole are unreadably noisy."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def plot_runs(results: Sequence[DQNResult], title: str = "DQN: one change each", figsize=(13, 4.5)):
    """The four-curve figure: smoothed training return and greedy eval return.

    Both panels, not one. The training-return panel shows what the agent experienced;
    the greedy-eval panel shows what it actually learned, and for at least one of the
    broken configs those two tell noticeably different stories.
    """
    import matplotlib.pyplot as plt

    fig, (ax_train, ax_eval) = plt.subplots(1, 2, figsize=figsize)
    for i, result in enumerate(results):
        color = f"C{i}"
        ax_train.plot(
            result.episode_steps, smooth(result.episode_returns), color=color, label=result.label
        )
        ax_eval.plot(
            result.eval_steps, result.eval_returns, color=color, marker="o", ms=3, label=result.label
        )

    ax_train.set_title("Training episode return (smoothed, epsilon-greedy)", fontsize=10)
    ax_eval.set_title("Greedy evaluation return", fontsize=10)
    for ax in (ax_train, ax_eval):
        ax.set_xlabel("environment step")
        ax.set_ylabel("return")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_q_divergence(results: Sequence[DQNResult], figsize=(7, 4)):
    """Mean |Q| over training, on a log axis. Divergence is obvious here and nowhere else."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    for i, result in enumerate(results):
        ax.plot(result.loss_steps, result.mean_abs_q, color=f"C{i}", label=result.label)
    ax.set_yscale("log")
    ax.set_xlabel("environment step")
    ax.set_ylabel("mean |Q(s, a)|  (log scale)")
    ax.set_title("Are the value estimates staying bounded?", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------
# Watching an episode
# ---------------------------------------------------------------------------------
#
# A learning curve says how much return an agent collected. It cannot say what the agent
# *did* to collect it, and on CartPole what it did is most of the diagnosis: a policy that
# lets the pole topple, a policy that balances the pole beautifully while walking the cart
# off the end of the track, and a policy that oscillates until the time limit rescues it
# can all end their episodes at the same step count.
#
# Gymnasium can render CartPole itself, but only through pygame, which is an extra
# dependency and an unhappy one on a headless machine. It is also unnecessary here,
# because the observation *is* the scene: CartPole's four numbers are the cart position x,
# the cart velocity, the pole angle theta and the pole's angular velocity, so a rectangle
# at x with a line leaning at theta is a faithful picture of the state, drawn with the
# matplotlib we already have.

#: Geometry and failure thresholds, copied from Gymnasium's ``CartPoleEnv.__init__``: the
#: episode ends when ``|x| > 2.4`` or ``|theta| > 12 degrees``, and ``length`` is *half*
#: the pole, which is why the pole drawn below is ``2 * length`` long. Used only as the
#: fallback for :func:`cartpole_geometry`, which prefers to read the live values.
X_THRESHOLD = 2.4
THETA_THRESHOLD_RADIANS = 12 * 2 * np.pi / 360
POLE_HALF_LENGTH = 0.5


@lru_cache(maxsize=1)
def cartpole_geometry() -> tuple[float, float, float]:
    """``(x_threshold, theta_threshold_radians, pole_half_length)`` from the installed Gymnasium.

    Drawing the failure boundary in the wrong place would be a quiet lie - the whole point
    of the picture is that you can see how close to the edge the agent is - so the numbers
    come from ``CartPoleEnv`` rather than from memory, and the module constants above are
    only the fallback for an environment where Gymnasium cannot be imported.
    """
    try:
        from gymnasium.envs.classic_control.cartpole import CartPoleEnv

        env = CartPoleEnv()
        return (
            float(env.x_threshold),
            float(env.theta_threshold_radians),
            float(env.length),
        )
    except Exception:  # pragma: no cover - only reached without Gymnasium
        return X_THRESHOLD, THETA_THRESHOLD_RADIANS, POLE_HALF_LENGTH


def rollout_frames(
    env,
    q_network: QNetwork | None = None,
    max_steps: int = 500,
    seed: int = 12345,
    device: str = "cpu",
) -> tuple[list[np.ndarray], float]:
    """One greedy episode, returned as the states visited and the return earned.

    ``q_network=None`` acts uniformly at random, and that path deliberately needs no
    network at all: "what does an agent that has learned nothing look like" is a question
    worth answering *before* any training has happened, and it is the baseline against
    which every later episode is read.

    Greedy rather than epsilon-greedy, for the same reason :func:`evaluate_policy` is:
    what we want to watch is the policy the agent learned, not the exploration noise
    sitting on top of it.
    """
    observations: list[np.ndarray] = []
    state, _ = env.reset(seed=seed)
    rng = random.Random(seed)
    n_actions = int(env.action_space.n)
    episode_return = 0.0

    for _ in range(max_steps):
        observations.append(np.asarray(state, dtype=np.float32).copy())
        if q_network is None:
            action = rng.randrange(n_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(q_network(state_tensor).argmax(dim=1).item())
        state, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        if terminated or truncated:
            # Keep the final state: the frame in which the pole is already over is the one
            # that tells you which way it fell.
            observations.append(np.asarray(state, dtype=np.float32).copy())
            break

    return observations, episode_return


def draw_cartpole(
    ax,
    observation,
    title: str = "",
    annotate: bool = True,
    cart_color: str = "C0",
    pole_color: str = "C1",
) -> None:
    """Draw one CartPole state on ``ax``: track, cart, pole, and the failure boundaries.

    The x-axis is pinned to the track's real extent, so the cart's distance from the two
    dashed lines at ``|x| = 2.4`` is the agent's remaining margin, at a glance and on the
    same scale in every frame. The two faint lines leaning away from the cart are the
    ``+/- 12 degrees`` beyond which the pole counts as fallen; the pole turns red once it
    or the cart is outside its limit.

    Does not clear ``ax`` - an animation redrawing into one axis should call ``ax.clear()``
    first, which is what :func:`animate_episode` does.
    """
    from matplotlib.patches import Rectangle

    x, _, theta, _ = (float(value) for value in np.asarray(observation, dtype=np.float64).reshape(-1)[:4])
    x_threshold, theta_threshold, half_length = cartpole_geometry()
    pole_length = 2.0 * half_length
    cart_width, cart_height = 0.5, 0.3
    failed = abs(x) > x_threshold or abs(theta) > theta_threshold

    ax.plot([-x_threshold, x_threshold], [0.0, 0.0], color="0.55", lw=1.5, zorder=0)
    for edge in (-x_threshold, x_threshold):
        ax.axvline(edge, color="0.75", ls="--", lw=1.0, zorder=0)

    for sign in (-1.0, 1.0):
        ax.plot(
            [x, x + pole_length * np.sin(sign * theta_threshold)],
            [cart_height, cart_height + pole_length * np.cos(theta_threshold)],
            color="0.85",
            ls="--",
            lw=1.0,
            zorder=1,
        )

    ax.add_patch(
        Rectangle(
            (x - cart_width / 2, 0.0),
            cart_width,
            cart_height,
            facecolor=cart_color,
            edgecolor="black",
            lw=0.8,
            zorder=2,
        )
    )
    ax.plot(
        [x, x + pole_length * np.sin(theta)],
        [cart_height, cart_height + pole_length * np.cos(theta)],
        color="C3" if failed else pole_color,
        lw=4.0,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot([x], [cart_height], marker="o", ms=4, color="black", zorder=4)

    ax.set_xlim(-x_threshold * 1.15, x_threshold * 1.15)
    ax.set_ylim(-0.18, cart_height + pole_length + 0.22)
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.set_xticks([-x_threshold, 0.0, x_threshold])
    ax.tick_params(labelsize=7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    if annotate:
        ax.text(
            0.01,
            0.97,
            f"x = {x:+.2f}    theta = {np.degrees(theta):+5.1f} deg",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            color="0.35",
        )
    if title:
        ax.set_title(title, fontsize=9)


def _frame_indices(n_observations: int, max_frames: int | None) -> list[int]:
    """Evenly spaced frame indices, at most ``max_frames`` of them, ends included."""
    n_observations = max(1, int(n_observations))
    if max_frames is None or n_observations <= max_frames:
        return list(range(n_observations))
    spaced = np.linspace(0, n_observations - 1, int(max_frames)).round().astype(int)
    return sorted(set(int(index) for index in spaced))


def animate_episode(
    observations: Sequence[np.ndarray],
    title: str = "",
    interval_ms: int = 40,
    max_frames: int | None = 150,
    figsize=(6.5, 2.6),
):
    """A ``FuncAnimation`` over one episode, embedded with ``HTML(anim.to_jshtml())``.

    ``to_jshtml`` writes every frame into the notebook as a base64 PNG and ships a small
    JavaScript player alongside them, which is what makes this work in a plain Jupyter
    kernel with no ffmpeg installed and no pygame available. It also means the notebook
    file grows with the frame count, hence ``max_frames``: a 500-step episode is
    subsampled to an evenly spaced 150 frames, which plays a little fast and costs a few
    megabytes rather than twenty.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if len(observations) == 0:
        raise ValueError("nothing to animate: the episode has no observations")

    indices = _frame_indices(len(observations), max_frames)
    fig, ax = plt.subplots(figsize=figsize)

    def draw(frame: int) -> None:
        ax.clear()
        step = indices[frame]
        label = f"{title}    t = {step}" if title else f"t = {step}"
        draw_cartpole(ax, observations[step], title=label)

    anim = FuncAnimation(fig, draw, frames=len(indices), interval=interval_ms, blit=False)
    # Close the figure, or the inline backend also emits the last frame as a still PNG
    # underneath the player, which looks like a bug to everyone who sees it.
    plt.close(fig)
    return anim


def _as_q_network(item, n_observations: int, n_actions: int, device: str = "cpu") -> QNetwork | None:
    """Accept ``None`` (random), a ``DQNResult``, a state dict, or a ready ``QNetwork``.

    Notebook 1 holds a ``results`` dictionary of ``DQNResult`` and nothing else, so the
    viewer takes that directly rather than making every caller rebuild the network by
    hand. The hidden width is read back out of the weights instead of assumed, so a run
    saved with a different ``hidden`` still loads.
    """
    if item is None:
        return None
    if isinstance(item, nn.Module):
        return item.to(device).eval()

    state_dict = item.state_dict if isinstance(item, DQNResult) else item
    if state_dict is None:
        raise ValueError("this run carries no weights: DQNResult.state_dict is None")
    hidden = int(state_dict["net.0.weight"].shape[0])
    network = QNetwork(n_observations, n_actions, hidden)
    network.load_state_dict(state_dict)
    return network.to(device).eval()


def compare_episodes(
    results_or_networks: Sequence[Any],
    labels: Sequence[str],
    env_id: str = "CartPole-v1",
    max_steps: int = 500,
    seed: int = 12345,
    device: str = "cpu",
    interval_ms: int = 40,
    max_frames: int | None = 150,
    as_filmstrip: bool = False,
    n_frames: int = 6,
    figsize=None,
):
    """Roll out one greedy episode per agent and show them side by side, in lockstep.

    Each entry of ``results_or_networks`` is ``None`` for a uniformly random agent, or a
    ``DQNResult``, state dict or ``QNetwork`` for a trained one, so the notebook can pass
    ``[None, results["working"]]`` and get the before-and-after picture in one call. Every
    agent starts from the same seeded initial state, which is what makes the comparison a
    comparison rather than two anecdotes.

    The episodes end at different times, and that is the point: a finished one holds its
    final frame, labelled and dimmed, while the others carry on. Each panel is titled with
    the return that episode actually earned.

    Returns a ``FuncAnimation``, or a ``Figure`` of stills when ``as_filmstrip=True``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    episodes: list[tuple[list[np.ndarray], float]] = []
    for item in results_or_networks:
        env = make_env(env_id, seed=seed)
        try:
            n_observations = int(np.prod(env.observation_space.shape))
            n_actions = int(env.action_space.n)
            network = _as_q_network(item, n_observations, n_actions, device)
            episodes.append(rollout_frames(env, network, max_steps=max_steps, seed=seed, device=device))
        finally:
            env.close()

    titles = [
        f"{label} - return {episode_return:.0f}"
        for label, (_, episode_return) in zip(labels, episodes)
    ]
    length = max(len(observations) for observations, _ in episodes)

    if as_filmstrip:
        steps = np.linspace(0, length - 1, max(1, int(n_frames))).round().astype(int)
        fig, axes = plt.subplots(
            len(episodes),
            len(steps),
            figsize=figsize or (2.2 * len(steps), 1.75 * len(episodes)),
            squeeze=False,
        )
        for row, ((observations, _), title) in enumerate(zip(episodes, titles)):
            for ax, step in zip(axes[row], steps):
                index = min(int(step), len(observations) - 1)
                over = index < int(step)
                draw_cartpole(
                    ax,
                    observations[index],
                    title=f"t = {index}" + (" (over)" if over else ""),
                    annotate=False,
                    pole_color="0.7" if over else "C1",
                )
            # Horizontal, to the left of the row: a rotated label is taller than these
            # short axes and would collide with the row below it.
            axes[row][0].set_ylabel(
                title.replace(" - ", "\n"), fontsize=8, rotation=0, ha="right", va="center"
            )
        fig.tight_layout()
        return fig

    indices = _frame_indices(length, max_frames)
    fig, axes = plt.subplots(
        1,
        len(episodes),
        figsize=figsize or (4.9 * len(episodes), 2.7),
        squeeze=False,
    )
    panels = list(axes[0])

    def draw(frame: int) -> None:
        step = indices[frame]
        for ax, (observations, _), title in zip(panels, episodes, titles):
            ax.clear()
            index = min(step, len(observations) - 1)
            over = index < step
            draw_cartpole(
                ax,
                observations[index],
                title=f"{title}\nt = {index}" + ("  (episode over)" if over else ""),
                pole_color="0.7" if over else "C1",
            )

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=len(indices), interval=interval_ms, blit=False)
    plt.close(fig)
    return anim


def filmstrip(
    observations: Sequence[np.ndarray],
    n_frames: int = 6,
    title: str = "",
    figsize=None,
    annotate: bool = False,
):
    """``n_frames`` evenly spaced stills of one episode in a row, labelled by timestep.

    The fallback wherever an embedded player is unwelcome - a kernel that will not render
    ``to_jshtml``, a printed handout, a diff of the notebook you would like to stay small.
    It is also better than the animation for one specific job: six frames side by side can
    be compared to each other all at once, which is how an oscillating policy gives itself
    away.
    """
    import matplotlib.pyplot as plt

    if len(observations) == 0:
        raise ValueError("nothing to draw: the episode has no observations")

    n_frames = max(1, int(n_frames))
    steps = np.linspace(0, len(observations) - 1, n_frames).round().astype(int)
    fig, axes = plt.subplots(
        1, n_frames, figsize=figsize or (2.2 * n_frames, 1.7), squeeze=False
    )
    for ax, step in zip(axes[0], steps):
        draw_cartpole(ax, observations[int(step)], title=f"t = {int(step)}", annotate=annotate)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90) if title else None)
    return fig


# ---------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------


def save_result(result: DQNResult, path: str) -> None:
    """Persist a run so notebook 1 can ship pre-trained weights and reference curves."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "label": result.label,
            "config": result.config.__dict__,
            "episode_steps": result.episode_steps,
            "episode_returns": result.episode_returns,
            "eval_steps": result.eval_steps,
            "eval_returns": result.eval_returns,
            "loss_steps": result.loss_steps,
            "losses": result.losses,
            "mean_abs_q": result.mean_abs_q,
            "state_dict": result.state_dict,
            "wall_seconds": result.wall_seconds,
        },
        path,
    )


def load_result(path: str) -> DQNResult:
    """Read back a saved run. Used by the pre-staged fallback path in notebook 1."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return DQNResult(
        label=blob["label"],
        config=DQNConfig(**blob["config"]),
        episode_steps=blob["episode_steps"],
        episode_returns=blob["episode_returns"],
        eval_steps=blob["eval_steps"],
        eval_returns=blob["eval_returns"],
        loss_steps=blob["loss_steps"],
        losses=blob["losses"],
        mean_abs_q=blob["mean_abs_q"],
        state_dict=blob["state_dict"],
        wall_seconds=blob["wall_seconds"],
    )
