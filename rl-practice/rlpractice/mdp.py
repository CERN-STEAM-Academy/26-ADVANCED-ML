"""A small tabular MDP and the two dynamic-programming algorithms, for notebook 1.

Why start with dynamic programming at all
-----------------------------------------
Because it is the only part of RL where you can see the whole answer. When the transition
model :math:`p(s', r \\mid s, a)` is known and the state space is small enough to
enumerate, the Bellman equations are just a linear system, and value iteration solves it
by repeated application of a contraction mapping. There is no sampling, no exploration,
no function approximation, and therefore nothing that can diverge.

Everything that follows in reinforcement learning is an attempt to keep doing this after
one of those luxuries is taken away. Notebook 1 takes them away one at a time:

* DP knows the model, sweeps every state, and uses exact values.
* DQN knows no model, samples states, and approximates values with a network - and the
  interaction of those three concessions is exactly the deadly triad.

The reference implementations live here so that the pre-staging tools, the test suite and
the fallback path in the student notebook have something to call. Do the exercise in the
notebook first.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

#: Action indices. Chosen so that `ACTION_DELTAS[a]` reads as (row change, column change).
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ["up", "right", "down", "left"]
ACTION_ARROWS = ["^", ">", "v", "<"]
ACTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GridWorld:
    """A windy, slippery gridworld with a goal, a pit, and walls.

    The layout is given as a list of strings, one per row:

    ``S`` start, ``G`` goal (terminal, +1), ``P`` pit (terminal, -1), ``#`` wall,
    ``.`` ordinary cell.

    Transitions are **stochastic**: the agent moves in the intended direction with
    probability ``slip_prob`` complement, and perpendicular to it with probability
    ``slip_prob / 2`` on each side. It never moves backwards. Bumping into a wall or the
    grid edge leaves it where it was.

    Stochasticity is not decoration. With deterministic transitions, policy evaluation
    degenerates into following a single path and the expectation in the Bellman operator
    is invisible. Here you have to sum over outcomes, which is the thing worth learning.

    The dynamics are exposed as ``P[state][action] -> [(probability, next_state, reward,
    terminal), ...]``, the same shape as the classic Gym ``env.P`` table, so the DP code
    written here transfers unchanged to those environments.
    """

    def __init__(
        self,
        # Two routes from S to G. The short one runs along the top and down the right
        # column, and passes the pit; the long one goes down the left and along the
        # bottom. Which one is optimal depends on `slip_prob`, which makes the sweep in
        # notebook 1 worth doing rather than decorative.
        layout: Iterable[str] = (
            "S....",
            ".##..",
            ".....",
            "..#.P",
            "....G",
        ),
        step_reward: float = -0.04,
        goal_reward: float = 1.0,
        pit_reward: float = -1.0,
        slip_prob: float = 0.2,
    ):
        self.grid = [list(row) for row in layout]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        if any(len(row) != self.n_cols for row in self.grid):
            raise ValueError("all layout rows must have the same length")

        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.pit_reward = pit_reward
        self.slip_prob = slip_prob

        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4
        self.start_state = self._find("S")
        self.terminal_states = {s for s in range(self.n_states) if self.cell(s) in "GP"}
        self.wall_states = {s for s in range(self.n_states) if self.cell(s) == "#"}
        self.P = self._build_transition_table()

    # --- geometry ------------------------------------------------------------------

    def state_of(self, row: int, col: int) -> int:
        return row * self.n_cols + col

    def coords_of(self, state: int) -> tuple[int, int]:
        return divmod(state, self.n_cols)

    def cell(self, state: int) -> str:
        row, col = self.coords_of(state)
        return self.grid[row][col]

    def _find(self, marker: str) -> int:
        for state in range(self.n_states):
            if self.cell(state) == marker:
                return state
        raise ValueError(f"layout contains no '{marker}' cell")

    def _move(self, state: int, action: int) -> int:
        """Where does the agent end up if it actually moves in direction ``action``?"""
        row, col = self.coords_of(state)
        d_row, d_col = ACTION_DELTAS[action]
        new_row, new_col = row + d_row, col + d_col
        if not (0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols):
            return state  # walked into the edge
        if self.grid[new_row][new_col] == "#":
            return state  # walked into a wall
        return self.state_of(new_row, new_col)

    def _reward_for(self, next_state: int) -> float:
        cell = self.cell(next_state)
        if cell == "G":
            return self.goal_reward
        if cell == "P":
            return self.pit_reward
        return self.step_reward

    def _build_transition_table(self):
        """``P[s][a] = [(prob, s_next, reward, terminal), ...]``, outcomes merged."""
        table: dict[int, dict[int, list[tuple[float, int, float, bool]]]] = {}
        for state in range(self.n_states):
            table[state] = {}
            for action in range(self.n_actions):
                if state in self.terminal_states or state in self.wall_states:
                    # Absorbing: a terminal state loops to itself with zero reward, so
                    # that V(terminal) = 0 falls out of the Bellman equation rather than
                    # having to be special-cased in the algorithms.
                    table[state][action] = [(1.0, state, 0.0, True)]
                    continue

                intended = 1.0 - self.slip_prob
                sideways = self.slip_prob / 2.0
                outcomes = {
                    action: intended,
                    (action - 1) % 4: sideways,
                    (action + 1) % 4: sideways,
                }

                merged: dict[int, float] = {}
                for direction, probability in outcomes.items():
                    next_state = self._move(state, direction)
                    merged[next_state] = merged.get(next_state, 0.0) + probability

                table[state][action] = [
                    (probability, next_state, self._reward_for(next_state), next_state in self.terminal_states)
                    for next_state, probability in sorted(merged.items())
                ]
        return table

    # --- rendering -----------------------------------------------------------------

    def render_values(self, values: np.ndarray, precision: int = 2) -> str:
        """The value function as a grid of numbers. Walls blank, terminals marked."""
        lines = []
        for row in range(self.n_rows):
            cells = []
            for col in range(self.n_cols):
                state = self.state_of(row, col)
                cell = self.grid[row][col]
                if cell == "#":
                    cells.append("  ####")
                else:
                    cells.append(f"{values[state]:6.{precision}f}")
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def render_policy(self, policy: np.ndarray) -> str:
        """The greedy policy as a grid of arrows. ``G``/``P`` for terminals."""
        lines = []
        for row in range(self.n_rows):
            cells = []
            for col in range(self.n_cols):
                state = self.state_of(row, col)
                cell = self.grid[row][col]
                if cell == "#":
                    cells.append("#")
                elif cell in "GP":
                    cells.append(cell)
                else:
                    action = int(np.argmax(policy[state]))
                    cells.append(ACTION_ARROWS[action])
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def uniform_random_policy(self) -> np.ndarray:
        """The policy that picks uniformly among the four actions in every state."""
        return np.full((self.n_states, self.n_actions), 1.0 / self.n_actions)


# ---------------------------------------------------------------------------------
# Dynamic programming
# ---------------------------------------------------------------------------------


def policy_evaluation(
    mdp: GridWorld,
    policy: np.ndarray,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> np.ndarray:
    r"""Iterative policy evaluation: compute :math:`v_\pi` for a fixed policy.

    Repeatedly apply the Bellman *expectation* operator until the values stop moving:

    .. math::
        v_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)
                     \bigl[ r + \gamma\, v_k(s') \bigr]

    The operator is a :math:`\gamma`-contraction in the max norm, so the iteration
    converges to the unique fixed point from any starting values. ``theta`` is the
    stopping tolerance on the largest single-state change.
    """
    values = np.zeros(mdp.n_states)
    for _ in range(max_iterations):
        delta = 0.0
        for state in range(mdp.n_states):
            old_value = values[state]
            new_value = 0.0
            for action in range(mdp.n_actions):
                action_probability = policy[state][action]
                if action_probability == 0.0:
                    continue
                action_value = 0.0
                for probability, next_state, reward, _ in mdp.P[state][action]:
                    action_value += probability * (reward + gamma * values[next_state])
                new_value += action_probability * action_value
            values[state] = new_value
            delta = max(delta, abs(new_value - old_value))
        if delta < theta:
            break
    return values


def action_values(mdp: GridWorld, values: np.ndarray, state: int, gamma: float) -> np.ndarray:
    r"""One-step lookahead: :math:`q(s, a)` for every action, given :math:`v`."""
    q = np.zeros(mdp.n_actions)
    for action in range(mdp.n_actions):
        for probability, next_state, reward, _ in mdp.P[state][action]:
            q[action] += probability * (reward + gamma * values[next_state])
    return q


def greedy_policy(mdp: GridWorld, values: np.ndarray, gamma: float = 0.99) -> np.ndarray:
    """The deterministic policy that is greedy with respect to ``values``.

    Ties are split evenly rather than broken by index. That matters in a symmetric
    gridworld, where an arbitrary tie-break makes the arrow plot look like a decision the
    algorithm did not actually make.
    """
    policy = np.zeros((mdp.n_states, mdp.n_actions))
    for state in range(mdp.n_states):
        q = action_values(mdp, values, state, gamma)
        best = np.flatnonzero(q == q.max())
        policy[state][best] = 1.0 / len(best)
    return policy


def value_iteration(
    mdp: GridWorld,
    gamma: float = 0.99,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    r"""Value iteration: compute :math:`v_*` and a greedy optimal policy.

    Identical to policy evaluation except that the expectation over actions is replaced
    by a maximum - the Bellman *optimality* operator:

    .. math::
        v_{k+1}(s) = \max_a \sum_{s', r} p(s', r \mid s, a)
                     \bigl[ r + \gamma\, v_k(s') \bigr]

    Still a :math:`\gamma`-contraction, so it still converges, and it does policy
    improvement and policy evaluation in the same sweep. Returns the optimal values, the
    greedy policy, and the history of max-norm deltas, which is worth plotting: the
    geometric decay at rate :math:`\gamma` is the contraction made visible.
    """
    values = np.zeros(mdp.n_states)
    deltas: list[float] = []
    for _ in range(max_iterations):
        delta = 0.0
        for state in range(mdp.n_states):
            old_value = values[state]
            values[state] = action_values(mdp, values, state, gamma).max()
            delta = max(delta, abs(values[state] - old_value))
        deltas.append(delta)
        if delta < theta:
            break
    return values, greedy_policy(mdp, values, gamma), deltas


def policy_return(
    mdp: GridWorld,
    policy: np.ndarray,
    gamma: float = 0.99,
    n_episodes: int = 2000,
    max_steps: int = 200,
    seed: int = 0,
) -> float:
    """Monte-Carlo estimate of the discounted return from the start state.

    A sanity check on the DP answer that uses no DP at all: if
    ``policy_evaluation(...)[start]`` and this disagree, one of them is wrong.
    """
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_episodes):
        state = mdp.start_state
        discount = 1.0
        episode_return = 0.0
        for _ in range(max_steps):
            if state in mdp.terminal_states:
                break
            action = rng.choice(mdp.n_actions, p=policy[state])
            outcomes = mdp.P[state][action]
            probabilities = [outcome[0] for outcome in outcomes]
            index = rng.choice(len(outcomes), p=probabilities)
            _, next_state, reward, _ = outcomes[index]
            episode_return += discount * reward
            discount *= gamma
            state = next_state
        total += episode_return
    return total / n_episodes
