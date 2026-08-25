#!/usr/bin/env python3
"""Build ``notebooks/01_classics_solutions.ipynb`` from a list of cells.

The notebook is authored here, as ordinary Python, and rendered by :mod:`tools.nbbuild`.
Hand-writing notebook JSON is a bad use of anyone's time and produces diffs nobody can
read; a builder script gives us version control that shows what actually changed, and one
obvious place to fix a typo that appears in three cells.

Run it, then execute the notebook once to fill in outputs, then generate the student
version with ``tools/make_student.py``. Never hand-edit the notebook: the next build
silently discards the edit.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from nbbuild import code, md, validate, write  # noqa: E402

OUTPUT = os.path.join(ROOT, "notebooks", "01_classics_solutions.ipynb")


CELLS = [
    # =============================================================================
    # Front matter
    # =============================================================================
    md(
        r"""
# Notebook 1: the classics

Thirty minutes, two parts, and one idea running through both of them: a value function is
defined by a self-referential equation, and every reinforcement learning algorithm is a way
of solving that equation under progressively worse conditions.

**Part 1, dynamic programming (10 min).** The model is known and the state space is
small, so the Bellman equations can be solved exactly. You implement policy evaluation
and value iteration by hand. Nothing here can diverge, and it is worth seeing what that
feels like before it stops being true.

**Part 2, the deadly triad (20 min).** The model is unknown, the states are sampled, and
the value function is a neural network. Not an implementation exercise: you are given a
working DQN and two configurations that each break it in exactly one way, and the job of
explaining which failure is which. First, though, a seven-state problem where divergence
can be proved rather than observed.

The two exercises in part 1 are the only code you write. Everything in part 2 is reading,
running and explaining.

Notation used throughout: $s$ is a state, $a$ an action, $r$ a reward,
$p(s', r \mid s, a)$ the transition model, $\gamma \in [0, 1)$ the discount factor,
$\pi(a \mid s)$ a policy, and

$$v_\pi(s) = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\middle|\; s_t = s\right]$$

the value of a state under a policy: the expected discounted return from here on.
"""
    ),
    md(
        r"""
## Setup

The notebooks live in `notebooks/` and the support package in `rlpractice/`, one
directory up, so the repository root goes on `sys.path` before anything is imported.

Two flags are defined here and used much later:

* `USE_REFERENCE_SOLUTIONS` is the instructor escape hatch. It is off by default. When
  the environment variable `RLPRACTICE_REFERENCE=1` is set, the cells after each exercise
  overwrite your function with the reference implementation, so the notebook runs end to
  end even with the exercises unanswered. That is how continuous integration checks this
  file; it is not how you should spend the next ten minutes.
* `TRAIN_FROM_SCRATCH` decides whether part 2 trains its three agents or loads
  pre-staged runs from `assets/dqn/`. Training all three takes roughly five minutes on a
  CPU.
"""
    ),
    code(
        r'''
import sys, os; sys.path.insert(0, os.path.abspath(".."))

# Where the big files live. Notebook 1 needs nothing large - it is CPU-only and generates
# everything it uses - so this is here purely so that both notebooks are configured the
# same way. Set it to a shared read-only path (at CERN, an /eos path) if you were given
# one; None means "use this repository".
from rlpractice import paths
SHARED_DIR = None
paths.use_shared(SHARED_DIR, verbose=False)

# Configure the inline backend explicitly rather than relying on it happening by itself.
# Matplotlib resolves its backend lazily, on the first figure, so importing pyplot does
# not register the PNG formatter for figures; that registration happens at the end of
# whichever cell first draws something. The consequence is that without this line the
# *first* plot of the notebook is recorded as the string "<Figure size 650x400 with 1
# Axes>" and its image is silently dropped, while every later plot renders normally.
%matplotlib inline

import inspect

import numpy as np
import matplotlib.pyplot as plt
import torch

from rlpractice import dqn
from rlpractice import mdp as mdp_module

# The instructor escape hatch. Leave it alone unless you are running this file in CI.
USE_REFERENCE_SOLUTIONS = os.environ.get("RLPRACTICE_REFERENCE", "0") == "1"

TRAIN_FROM_SCRATCH = True   # set False to load pre-staged runs from assets/dqn/

print("numpy      ", np.__version__)
print("torch      ", torch.__version__)
print("reference solutions:", USE_REFERENCE_SOLUTIONS)
print("train DQN from scratch:", TRAIN_FROM_SCRATCH)
'''
    ),
    # =============================================================================
    # Part 1: dynamic programming
    # =============================================================================
    md(
        r"""
# Part 1: dynamic programming

Dynamic programming is the only corner of reinforcement learning where the whole answer
is computable. Three luxuries make that possible:

1. the transition model $p(s', r \mid s, a)$ is **known**, so expectations are sums we
   can evaluate rather than averages we have to sample;
2. the state space is **small enough to enumerate**, so every state can be visited on
   every sweep;
3. values are stored **exactly**, one number per state, so updating one state does not
   disturb any other.

Under those conditions the Bellman equations are a fixed-point problem, the Bellman
operator is a $\gamma$-contraction in the max norm, and iterating it converges to the
unique fixed point from any starting values. There is no exploration problem, no variance,
and nothing that can diverge.

Everything else in reinforcement learning is an attempt to keep doing this once one of
the three luxuries is removed. Part 2 removes all three at once, which is the point of
part 2. Removing them one at a time is the entire history of the field.
"""
    ),
    md(
        r"""
## 1.1 The environment

`rlpractice.mdp.GridWorld` is a five-by-five grid with a start `S`, a goal `G` worth
$+1$, a pit `P` worth $-1$, walls `#`, and a step cost of $-0.04$ everywhere else. Goal
and pit are terminal.

The transitions are **stochastic**, and that is not decoration. The agent moves in the
intended direction with probability $1 - \texttt{slip\_prob} = 0.8$ and perpendicular to
it with probability $0.1$ on each side. It never moves backwards. Walking into a wall or
off the edge leaves it where it was.

With deterministic transitions, policy evaluation degenerates into following a single
path and the expectation in the Bellman operator becomes invisible. With slip, you have
to sum over outcomes, which is the part worth learning. It also changes the answer: a
route that squeezes past the pit is fine for a deterministic agent and a bad idea for one
that slips sideways one time in ten.

The dynamics are exposed as `P[state][action] -> [(probability, next_state, reward,
terminal), ...]`, the same shape as the classic Gym `env.P` table. Code written against
this interface transfers unchanged.
"""
    ),
    code(
        r'''
grid = mdp_module.GridWorld()

print("layout")
for row in grid.grid:
    print("   ", " ".join(row))

print()
print("states       ", grid.n_states, "(5 x 5, walls included as unreachable states)")
print("actions      ", grid.n_actions, mdp_module.ACTION_NAMES)
print("start state  ", grid.start_state, "at", grid.coords_of(grid.start_state))
print("terminals    ", sorted(grid.terminal_states))
print("walls        ", sorted(grid.wall_states))
print("slip prob    ", grid.slip_prob)


def show_transitions(mdp, state):
    """Print the transition table of a single state, one line per outcome."""
    row, col = mdp.coords_of(state)
    print(f"state {state} at (row {row}, col {col}), cell {mdp.cell(state)!r}")
    for action in range(mdp.n_actions):
        name = mdp_module.ACTION_NAMES[action]
        print(f"  action {action} ({name:>5}):")
        for probability, next_state, reward, terminal in mdp.P[state][action]:
            next_row, next_col = mdp.coords_of(next_state)
            print(
                f"      p={probability:.2f} -> state {next_state:2d} "
                f"(row {next_row}, col {next_col}, cell {mdp.cell(next_state)!r})"
                f"  reward {reward:+.2f}  terminal {terminal}"
            )


print()
show_transitions(grid, grid.start_state)

# The state directly above the pit. Look at the "down" action: four fifths of the
# probability mass lands in a terminal state worth -1. This single table is the reason
# the optimal policy will refuse to come near this column.
print()
show_transitions(grid, grid.state_of(2, 4))
'''
    ),
    md(
        r"""
## 1.2 Exercise 1: policy evaluation

*Given a policy, what is it worth?* Not "which policy is best" - that comes next. Fix
$\pi$ and compute $v_\pi$, the expected discounted return from every state.

The definition is self-referential, because the return from $s$ is the immediate reward
plus the discounted return from wherever you land:

$$v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \bigl[ r + \gamma\, v_\pi(s') \bigr]$$

That is one equation per state, and with 25 states you could solve the linear system
directly. Nobody does, because iterating is simpler and generalises: start from
$v_0 \equiv 0$, treat the right-hand side as an operator, and apply it repeatedly,

$$v_{k+1}(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \bigl[ r + \gamma\, v_k(s') \bigr].$$

The operator is a $\gamma$-contraction in the max norm: applying it to any two value
functions brings them at least a factor $\gamma$ closer together. So the iteration
converges to the unique fixed point, from any starting point, and the largest single-state
change $\Delta = \max_s |v_{k+1}(s) - v_k(s)|$ is a usable stopping signal. Stop when it
drops below `theta`.

Two implementation notes:

* Update `values` **in place**, so later states in a sweep already see the newer values of
  earlier ones. This is Gauss-Seidel rather than Jacobi. It is still a contraction, and it
  converges faster because information propagates within a sweep instead of waiting for
  the next one.
* Terminal states need no special case. `GridWorld` makes them absorbing with zero
  reward, so $v(\text{terminal}) = 0$ falls straight out of the equation.
"""
    ),
    code(
        r'''
def policy_evaluation(mdp, policy, gamma=0.99, theta=1e-8, max_iterations=10_000):
    """Compute the value function of a fixed policy by iterating the Bellman operator.

    Parameters
    ----------
    mdp
        A ``GridWorld``. Use ``mdp.n_states``, ``mdp.n_actions`` and the transition table
        ``mdp.P[state][action] -> [(probability, next_state, reward, terminal), ...]``.
    policy
        Array of shape ``(n_states, n_actions)``. ``policy[s][a]`` is the probability of
        taking action ``a`` in state ``s``, so each row sums to one.
    gamma
        Discount factor.
    theta
        Stopping tolerance on the largest single-state change in a sweep.
    max_iterations
        Safety valve, so a bug costs you a second rather than a kernel restart.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_states,)`` holding the value of every state under ``policy``.
    """
    values = np.zeros(mdp.n_states)

    # TODO(hint): sweep every state, updating values[state] in place with the Bellman expectation backup, and stop when the largest change in a sweep is below theta
    # BEGIN SOLUTION
    for _ in range(max_iterations):
        delta = 0.0
        for state in range(mdp.n_states):
            old_value = values[state]

            # The expectation over actions, then over outcomes of each action. Two
            # separate sums, written as two separate loops, because they mean two
            # different things: the agent's choice, and the environment's response.
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
    # END SOLUTION

    return values
'''
    ),
    code(
        r'''
if USE_REFERENCE_SOLUTIONS:
    from rlpractice.mdp import policy_evaluation
    print("using the reference implementation")
'''
    ),
    md(
        r"""
### Test: three ways of asking the same question

The first check is against the reference implementation in `rlpractice.mdp`, which is the
same algorithm and therefore only catches coding mistakes.

The second is more interesting, because it uses no dynamic programming at all:
`mdp.policy_return` rolls out two thousand episodes under the same policy and averages the
discounted return from the start state. If the fixed point of the Bellman operator and a
Monte-Carlo average of actual episodes disagree, one of them is wrong. They agree here to
within sampling noise, and the residual gap is dominated by episodes truncated at 200
steps.
"""
    ),
    code(
        r'''
GAMMA = 0.99

uniform_policy = grid.uniform_random_policy()
print("uniform policy shape", uniform_policy.shape, "row sums", uniform_policy.sum(axis=1)[:3])

values_mine = policy_evaluation(grid, uniform_policy, gamma=GAMMA)
values_reference = mdp_module.policy_evaluation(grid, uniform_policy, gamma=GAMMA)

largest_difference = np.abs(values_mine - values_reference).max()
print(f"largest disagreement with the reference: {largest_difference:.3e}")
assert largest_difference < 1e-6, "policy_evaluation disagrees with the reference"

print()
print("v_pi for the uniform random policy (gamma = 0.99)")
print(grid.render_values(values_mine))

start = grid.start_state
monte_carlo = mdp_module.policy_return(grid, uniform_policy, gamma=GAMMA, n_episodes=2000, seed=0)
print()
print(f"v_pi(start) from dynamic programming : {values_mine[start]:.4f}")
print(f"v_pi(start) from 2000 episodes       : {monte_carlo:.4f}")
print(f"difference                           : {abs(values_mine[start] - monte_carlo):.4f}")
assert abs(values_mine[start] - monte_carlo) < 0.1, "DP and Monte Carlo disagree"

# Every value is negative: a policy that flips a coin four ways wanders for a long time,
# pays -0.04 per step, and falls into the pit about as often as it reaches the goal.
print()
print("value range:", f"{values_mine.min():.3f} to {values_mine.max():.3f}")
'''
    ),
    md(
        r"""
## 1.3 Exercise 2: value iteration

Now the useful question: *what is the best a policy could be worth, and which policy
achieves it?*

The optimal value function satisfies the Bellman **optimality** equation, which differs
from the expectation equation in exactly one place - the expectation over actions becomes
a maximum:

$$v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a) \bigl[ r + \gamma\, v_*(s') \bigr]$$

The reason that one change is enough is worth a moment. Policy evaluation averages over
whatever the policy does; the optimality operator assumes the agent will take the best
action available given the current estimate of what follows. Iterating it therefore
performs policy improvement and policy evaluation in the same sweep, without ever
representing a policy explicitly. The policy only appears at the end, read off by acting
greedily with respect to $v_*$:

$$\pi_*(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) \bigl[ r + \gamma\, v_*(s') \bigr]$$

The optimality operator is also a $\gamma$-contraction, so the same convergence argument
and the same stopping rule apply. Return the history of $\Delta$ values as well: plotted
on a log axis it is the contraction made visible.

Extracting the greedy policy at the end is mechanical, so use the provided
`mdp_module.greedy_policy`, which splits ties evenly between equally good actions. That
matters in a symmetric gridworld: breaking ties by index draws an arrow that looks like a
decision the algorithm never made.
"""
    ),
    code(
        r'''
def value_iteration(mdp, gamma=0.99, theta=1e-8, max_iterations=10_000):
    """Compute the optimal value function and a greedy optimal policy.

    Identical to policy evaluation except that the expectation over actions is replaced
    by a maximum, and that the size of each sweep's largest change is recorded.

    Parameters
    ----------
    mdp
        A ``GridWorld``.
    gamma
        Discount factor.
    theta
        Stopping tolerance on the largest single-state change in a sweep.
    max_iterations
        Safety valve.

    Returns
    -------
    tuple
        ``(values, policy, deltas)`` where ``values`` has shape ``(n_states,)``,
        ``policy`` has shape ``(n_states, n_actions)`` and is greedy with respect to
        ``values``, and ``deltas`` is the list of max-norm changes, one per sweep.
    """
    values = np.zeros(mdp.n_states)
    deltas = []

    # TODO(hint): same sweep as policy evaluation, but take the maximum over actions instead of the policy-weighted average, and append the sweep's largest change to deltas
    # BEGIN SOLUTION
    for _ in range(max_iterations):
        delta = 0.0
        for state in range(mdp.n_states):
            old_value = values[state]

            best_action_value = -np.inf
            for action in range(mdp.n_actions):
                action_value = 0.0
                for probability, next_state, reward, _ in mdp.P[state][action]:
                    action_value += probability * (reward + gamma * values[next_state])
                best_action_value = max(best_action_value, action_value)

            values[state] = best_action_value
            delta = max(delta, abs(values[state] - old_value))

        deltas.append(delta)
        if delta < theta:
            break
    # END SOLUTION

    policy = mdp_module.greedy_policy(mdp, values, gamma)
    return values, policy, deltas
'''
    ),
    code(
        r'''
if USE_REFERENCE_SOLUTIONS:
    from rlpractice.mdp import value_iteration
    print("using the reference implementation")
'''
    ),
    md(
        r"""
### Test: the answer, the policy, and the contraction

The delta plot is the part to look at. On a log axis the curve is a straight line, which
is what geometric decay looks like, and the dashed line is the worst case the contraction
property guarantees: $\Delta_k \le \Delta_1 \gamma^{k-1}$. The real curve falls faster
than the bound because in-place sweeps propagate a value change to its neighbours within
the same sweep rather than in the next one. The guarantee is a floor on the rate of
progress, not a prediction of it.
"""
    ),
    code(
        r'''
optimal_values, optimal_policy, deltas = value_iteration(grid, gamma=GAMMA)
reference_values, reference_policy, reference_deltas = mdp_module.value_iteration(grid, gamma=GAMMA)

largest_difference = np.abs(optimal_values - reference_values).max()
print(f"sweeps to convergence: {len(deltas)}")
print(f"largest disagreement with the reference: {largest_difference:.3e}")
assert largest_difference < 1e-6, "value_iteration disagrees with the reference"
assert np.array_equal(optimal_policy.argmax(axis=1), reference_policy.argmax(axis=1))

print()
print("v_* (gamma = 0.99)")
print(grid.render_values(optimal_values))
print()
print("greedy policy with respect to v_*")
print(grid.render_policy(optimal_policy))

start_return = mdp_module.policy_return(grid, optimal_policy, gamma=GAMMA, n_episodes=2000, seed=0)
print()
print(f"v_*(start) from dynamic programming : {optimal_values[start]:.4f}")
print(f"v_*(start) from 2000 episodes       : {start_return:.4f}")

# Optimality dominates: no state is worse off under the optimal policy than under the
# uniform random one. This is true by construction rather than by luck, but it is cheap
# to check, and a violation would mean one of the two exercises is wrong.
assert np.all(optimal_values >= values_mine - 1e-9), "v_* is below v_pi somewhere"
print()
print(f"largest improvement of v_* over v_pi: {(optimal_values - values_mine).max():.3f}")
print(f"smallest improvement (terminal states, both zero): {(optimal_values - values_mine).min():.3f}")

# --- the contraction, made visible -----------------------------------------------
# This lives in the same cell as the test above rather than in its own, so that an
# unimplemented value_iteration produces exactly one failing cell instead of a failing
# cell followed by a cascade of NameErrors on `deltas`.
sweeps = np.arange(1, len(deltas) + 1)
contraction_bound = deltas[0] * GAMMA ** (sweeps - 1)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogy(sweeps, deltas, marker="o", ms=3, label="observed max-norm change")
ax.semilogy(sweeps, contraction_bound, "--", color="grey", label=r"guaranteed bound $\Delta_1\,\gamma^{k-1}$")
ax.set_xlabel("sweep $k$")
ax.set_ylabel(r"$\Delta_k = \max_s |v_k(s) - v_{k-1}(s)|$")
ax.set_title("Value iteration is a contraction", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

observed_ratios = [deltas[i + 1] / deltas[i] for i in range(len(deltas) - 1) if deltas[i] > 0]
print(f"first five per-sweep contraction ratios: {[round(r, 3) for r in observed_ratios[:5]]}")
print(f"last five                              : {[round(r, 3) for r in observed_ratios[-5:]]}")
print(f"gamma                                  : {GAMMA}")
'''
    ),
    md(
        r"""
## 1.4 What the two answers say

**Optimality dominates, everywhere.** Every entry of $v_*$ is at least the corresponding
entry of $v_\pi$ for the uniform random policy - the test cell above asserted it - and in
every non-terminal state it is better by more than two units of return. That is not a
happy accident of this gridworld: $v_*(s) \ge v_\pi(s)$ holds for every state and every
policy, directly from the definition of the maximum, and it is the reason policy
improvement can never make things worse.

**The optimal policy routes around the pit rather than past it.** The geometrically
obvious way from the top-right corner to `G` is straight down the last column, and the pit
sits in it. Read the arrows in that column: at (row 1, column 4) the policy moves *left*
and at (row 2, column 4) it moves *up*, in both cases away from the goal it is trying to
reach. The transition table printed in 1.1 explains why - from (row 2, column 4), "down"
puts four fifths of its probability mass into a terminal state worth $-1$, and the sideways
slips offer no comfort either. The grid therefore splits into two safe descents: the left
half goes down column 0 and along the bottom row, the right half goes down column 3, one
column clear of the pit.

The values record the same judgement as numbers. The cell at (row 2, column 4) is four
steps from the goal and worth $0.44$; the cell at (row 2, column 1) is five steps away and
worth $0.69$. Being nearer the goal is worth less than being far from the pit, and the
size of that inversion is exactly what a slip probability of $0.2$ costs.

None of this is hard-coded. Rebuild the grid with `GridWorld(slip_prob=0.0)` and re-run the
two cells above: with deterministic movement the ranking flips, (row 2, column 4) becomes
the more valuable of the two at $0.85$ against $0.80$, and the policy at the start state
turns right along the top row to take the short path it just refused. Nobody told the
algorithm that pits are dangerous. It read the transition probabilities.

That trade-off is the entire content of the value function, and it was computed without a
single episode being simulated. The rest of reinforcement learning is about what to do when
you cannot do that.
"""
    ),
    # =============================================================================
    # Part 2: DQN diagnosis
    # =============================================================================
    md(
        r"""
# Part 2: DQN and the deadly triad

Take the three luxuries away.

CartPole has a continuous four-dimensional state, so there is no table to sweep and no
transition model to sum over. What is left is Q-learning with a neural network: act,
store the transition, and regress the network's prediction onto a bootstrapped target,

$$y = r + \gamma \max_{a'} Q_{\text{target}}(s', a'), \qquad
\mathcal{L}(\theta) = \bigl( Q_\theta(s, a) - y \bigr)^2 .$$

Look at what $y$ contains: the network's own output. That is **bootstrapping** - updating
an estimate towards another estimate rather than towards observed returns.

Sutton and Barto call the following combination the **deadly triad**:

1. **bootstrapping** - the target contains the current estimate;
2. **off-policy learning** - the data comes from a behaviour policy (epsilon-greedy) while
   the values being learned belong to a different policy (greedy);
3. **function approximation** - values are represented by a parametric model, so an update
   at one state moves the value of every state that resembles it.

Any two of the three are safe. Tabular Q-learning bootstraps and is off-policy, but each
state has its own number, so an update cannot corrupt a neighbour. Monte-Carlo control with
a network is off-policy and approximate, but its targets are observed returns and do not
move when the weights move. On-policy TD with a network bootstraps and approximates, but
the distribution it trains on is the distribution it acts under.

All three together can diverge - not "converge slowly", diverge, with value estimates
growing without bound. The counterexamples are small and famous (Baird's star, Tsitsiklis
and Van Roy). DQN does not remove any leg of the triad; it adds engineering that keeps two
of them in check:

* a **target network**, refreshed only every few hundred steps, so the bootstrap target is
  approximately fixed while the online network chases it;
* a **replay buffer**, large enough to hold many episodes, so a minibatch is roughly i.i.d.
  rather than a correlated slice of the last two seconds of one trajectory.

The third leg, function approximation, has no mitigation at all beyond a small learning
rate and gradient clipping.

We are going to check that "any two of the three are safe, all three can diverge" claim
twice, because one check on its own would not be convincing. First on a problem small
enough to settle the question exactly, and then on a real agent, where it is messier and
much more like the thing you will actually be debugging.
"""
    ),

    # -------------------------------------------------------------------------
    # 2.0 Baird's counterexample
    # -------------------------------------------------------------------------
    md(
        r"""
## 2.0 First, the claim, settled exactly

Before touching a neural network, here is the deadly triad in a form small enough to
reason about completely. Baird's counterexample (Baird, 1995) is seven states, and every
reward is zero.

Because every reward is zero, the true value function is zero everywhere - and it is
*exactly representable* by the linear approximator we are about to use. The perfect
solution $w = 0$ is sitting right there. This is not a capacity problem, an optimisation
problem or a tuning problem.

Semi-gradient off-policy TD(0) walks away from it anyway, to infinity.

What makes this worth five minutes is that each leg of the triad can be switched off
independently:

* replace the target policy with the behaviour policy, and it is **on-policy**;
* replace the features with an identity matrix, and it is **tabular** - no approximation;
* regress on the observed return instead of on $r + \gamma v(s')$, and there is **no
  bootstrapping**.

Any one of those three changes should be enough to make it safe, if the claim is true.
"""
    ),
    code(
        r'''
from rlpractice import baird

ablation = baird.triad_ablation(steps=5000, alpha=0.01, seed=0)

print(f"{'configuration':<30}{'final |w|':>14}{'max Re(eig)':>14}")
for label, data in ablation.items():
    norm = np.linalg.norm(data["history"][-1])
    eigenvalue = data["eigenvalue"]
    shown = "     n/a" if np.isnan(eigenvalue) else f"{eigenvalue:+8.4f}"
    print(f"{label:<30}{norm:>14.4g}{shown:>14}")

print()
print("max Re(eig) is the largest real part of an eigenvalue of the matrix governing the")
print("EXPECTED update. Positive means the iteration diverges for every step size, however")
print("small - so this is not a matter of tuning alpha, and not sampling noise either.")
'''
    ),
    code(
        r'''
baird.plot_triad(ablation)
plt.show()
'''
    ),
    md(
        r"""
That is the claim settled: all three legs together diverge, and removing any single one of
them leaves the weights bounded. Note the eigenvalue column in particular - the divergence
is a property of the update operator itself, not of unlucky sampling, so no amount of
smaller step sizes or longer training would rescue it.

Now the messy version.
"""
    ),
    md(
        r"""
## 2.1 Read the code before you break it

`rlpractice.dqn.train_dqn` is complete and working. Read it now. It is forty lines of loop
and there is nothing hidden in it - if you have written a supervised training loop, the
only unfamiliar parts are where the labels come from.

Three things to find while reading:

* the target network being **read** (`target(next_states)`) and being **refreshed**
  (`target.load_state_dict`) in two different places, `target_update_every` steps apart;
* the replay buffer being written to on every environment step and sampled from on every
  training step;
* the distinction between `terminated` and `truncated`. The pole falling means the return
  really is over and the target bootstraps to zero; hitting the time limit does not, and
  conflating the two teaches the agent that surviving to the end of the episode is worth
  nothing.
"""
    ),
    code(
        r'''
print(inspect.getsource(dqn.train_dqn))
'''
    ),
    md(
        r"""
The three mitigations are labelled in the source you just printed:

| Where | What it does | Which leg it protects |
|---|---|---|
| `target = copy.deepcopy(online)` and the periodic `target.load_state_dict(...)` | freezes the bootstrap target for `target_update_every` steps | bootstrapping |
| `buffer.sample(config.batch_size, ...)` over a 50 000-transition buffer | decorrelates and reuses experience | off-policy data distribution |
| `lr=1e-3` with `clip_grad_norm_(..., 10.0)` and a Huber loss | keeps each parameter step inside the region where the network's local linearisation is a reasonable description of it | function approximation |

Each of the three broken configurations below removes exactly one of these.
"""
    ),
    md(
        r"""
## 2.2 Two broken configurations, one change each

`DQNConfig` is a dataclass, and each broken configuration is built from the working one
with `dataclasses.replace`. That makes "differs by exactly one field" a fact you can print
rather than a claim you have to trust, which matters: if a run misbehaves for two reasons
at once, the exercise teaches nothing.

Both breaks attack the **bootstrapping** leg, from opposite ends. One removes the thing
that holds the target still; the other removes the thing that grounds the recursion in an
observed fact. Neither touches the replay buffer, and that omission is deliberate - see
the note after the curves.
"""
    ),
    code(
        r'''
print("the working configuration")
for name, value in dqn.WORKING.__dict__.items():
    print(f"  {name:22s} {value!r}")

print()
for label, config in dqn.CONFIGS.items():
    if label == "working":
        continue
    difference = config.diff(dqn.WORKING)
    assert len(difference) == 1, f"{label} differs from WORKING in {len(difference)} fields"
    for field_name, (broken_value, working_value) in difference.items():
        print(f"{label}: {field_name} = {working_value!r} -> {broken_value!r}")

print()
print("total environment steps per run:", dqn.WORKING.total_steps)
print("evaluations per run:            ", dqn.WORKING.total_steps // dqn.WORKING.eval_every)
'''
    ),
    md(
        r"""
## 2.3 Run all three

Roughly ninety seconds each on an unloaded CPU, so about five minutes for the set; a busy
machine can double that. Evaluation, not training, is most of the cost: twelve thousand
gradient steps on a two-layer MLP are cheap, whereas five greedy episodes of an agent that
has learned to balance the pole are up to 2500 environment steps every time.

If you would rather not wait, or if a run misbehaves, set `TRAIN_FROM_SCRATCH = False` in
the setup cell and re-run from here. Both paths produce the same `results` dictionary,
keyed by label, so every cell below this one works either way.
"""
    ),
    code(
        r'''
ASSET_DIR = os.path.abspath(os.path.join("..", "assets", "dqn"))


def get_result(label, config):
    """Return a finished run of ``config``, either loaded from disk or trained now.

    The loaded and the trained object are the same ``DQNResult`` type carrying the same
    fields, so nothing downstream needs to know which branch was taken.
    """
    asset_path = os.path.join(ASSET_DIR, f"{label}.pt")

    if not TRAIN_FROM_SCRATCH:
        if os.path.exists(asset_path):
            print(f"[{label}] loading pre-staged run from {asset_path}")
            return dqn.load_result(asset_path)
        print(f"[{label}] WARNING: no pre-staged run at {asset_path}, training instead")

    env = dqn.make_env(config.env_id, seed=config.seed)
    try:
        return dqn.train_dqn(env, config, label=label, verbose=True)
    finally:
        env.close()


results = {}
for label, config in dqn.CONFIGS.items():
    print(f"=== {label} " + "=" * (40 - len(label)))
    results[label] = get_result(label, config)

print()
print("results:", list(results))
'''
    ),
    code(
        r'''
# final eval return, best eval return, the largest mean |Q| seen during training, wall time
for label, result in results.items():
    print(result.summary())
'''
    ),
    md(
        r"""
## 2.4 The curves

Two figures. The first has two panels because training return and greedy evaluation return
tell different stories: training return is contaminated by epsilon-greedy exploration, and
for at least one of these configurations the agent's experience and what it actually
learned diverge visibly.

The second figure is the one that settles arguments. It plots the mean absolute Q-value of
each training minibatch on a log axis. There is a hard ceiling on what those numbers are
allowed to be: CartPole pays $+1$ per surviving step, so with $\gamma = 0.99$ no true value
exceeds $\sum_{t \ge 0} \gamma^t = 1/(1 - \gamma) = 100$, whatever the policy. A run whose
estimates sit far above that is no longer approximating a value function, it is diverging,
and the log axis is where you can see the difference between "worse" and "diverging".

Reward curves tell you that something went wrong. This plot tells you whether the value
function itself came apart, and the two failures below are distinguishable only here.
"""
    ),
    code(
        r'''
figure = dqn.plot_runs(list(results.values()))
plt.show()
'''
    ),
    code(
        r'''
figure = dqn.plot_q_divergence(list(results.values()))
plt.show()

# The same evidence as a table, with the late-training TD loss beside it. The median and
# the maximum over the last forty recorded losses, not the mean, because one of these runs
# is quiet almost all the time and violently spiky occasionally, and a mean hides exactly
# that. Read the columns together: a loss that never settles means the regression is not
# converging, and a loss far below the working run's means the network is fitting whatever
# it was shown - which is good news only if it was shown a representative sample.
header = f"{'config':>9}  {'|Q| start':>10}  {'|Q| end':>10}  {'|Q| max':>10}  {'loss median':>11}  {'loss max':>10}"
print(header)
print("-" * len(header))
for label, result in results.items():
    q_values = np.asarray(result.mean_abs_q)
    late_losses = np.asarray(result.losses[-40:])
    print(
        f"{label:>9}  {q_values[0]:10.3g}  {q_values[-1]:10.3g}  {q_values.max():10.3g}  "
        f"{np.median(late_losses):11.3g}  {late_losses.max():10.3g}"
    )

print()
print(f"ceiling on any true CartPole value under gamma = {dqn.WORKING.gamma}: {1.0 / (1.0 - dqn.WORKING.gamma):.0f}")
'''
    ),
    md(
        r"""
## 2.5 Watch an episode

A return of 500 and a return of 60 are two numbers, and numbers are where a diagnosis
starts rather than where it ends. What a learning curve cannot tell you is *how* the
episode ended: whether the pole toppled, whether the cart balanced it beautifully and then
walked off the end of the track, or whether the policy is oscillating and surviving by
luck. Those are three different faults with one summary statistic between them, and thirty
seconds of watching separates them.

CartPole's observation is the entire scene - cart position $x$, cart velocity, pole angle
$\theta$, pole angular velocity - so the pictures below are drawn straight from the
observations with matplotlib, with no rendering backend involved. The dashed vertical lines
at $|x| = 2.4$ and the faint wedge at $\pm 12^\circ$ are the two conditions that end an
episode, so the distance to them is the agent's remaining margin.
"""
    ),
    code(
        r'''
from IPython.display import HTML

# Both agents start from the same seeded initial state, so this is a comparison rather than
# two anecdotes. `None` is the untrained baseline: it acts uniformly at random and needs no
# network at all, which is the point of having it.
animation = dqn.compare_episodes(
    [None, results["working"]],
    ["random (untrained)", "working DQN"],
    seed=12345,
    max_steps=500,
    max_frames=90,   # every frame is embedded in this file as a PNG, so they are rationed
)
HTML(animation.to_jshtml())
'''
    ),
    md(
        r"""
The random agent lasts about twenty steps and never had a plan. The trained one holds the
pole up for as long as you are willing to watch - and when it does finally end, look at
*which* limit it hits. An agent that runs out of episode at $x = 2.4$ with the pole still
near vertical has not failed at balancing at all: it has found that a steady push in one
direction is a perfectly good way to keep the pole up, because nothing in the reward
mentions where the cart is until the track runs out. That is a reward specification
problem, not a learning problem, and it is invisible in every figure above.
"""
    ),
    code(
        r'''
# The same view of a broken run, as stills rather than a player: six frames side by side
# can be compared to one another at a glance, which is what an episode that is over in ten
# steps needs. `as_filmstrip=True` - and `dqn.filmstrip` for a single episode - is also the
# fallback anywhere the embedded player will not render.
figure = dqn.compare_episodes(
    [None, results["CONFIG_A"]],
    ["random (untrained)", "CONFIG_A (trained)"],
    seed=12345,
    as_filmstrip=True,
    n_frames=6,
)
plt.show()
'''
    ),
    md(
        r"""
This is the difference between learning nothing and learning something wrong. The random
agent survives roughly twice as long as `CONFIG_A`, which is a peculiar thing for twelve
thousand gradient steps to have bought. `CONFIG_A` is neither undertrained nor unlucky: its
greedy policy is confident and deterministic, and what it confidently does is put the pole
on the floor as fast as the physics allow. Hold that against its mean $|Q|$ from the
previous figure, in the tens of millions, and the next exercise already has most of its
answer.
"""
    ),
    md(
        r"""
## 2.6 Exercise 3: diagnose the failures

This exercise is prose, not code, and it is the reason part 2 exists.

For each of `CONFIG_A` and `CONFIG_B`: name what the single changed field removed, and
account for the curve you actually plotted. Use both figures and the mean $|Q|$ panel -
the two runs finish at almost exactly the same evaluation return and could hardly be more
different inside.

Then one further question, which is the one worth carrying away: both of these break the
same leg of the triad. Look back at section 2.0. Why did we demonstrate the *other* two
legs with Baird's counterexample rather than with a DQN configuration here?

Answer in this cell.

<!-- TODO(hint): what did each changed field remove, why does that produce the curve you saw, and why is the triad demonstrated on Baird rather than on CartPole? -->

<!-- BEGIN SOLUTION -->
Both runs diverge, and they diverge for reasons that are worth keeping separate.

**`CONFIG_A`, `target_update_every: 200 -> 1`. Removed: the frozen bootstrap target.**

Refreshing the target every step means there is no target network, only the online network
under a second name. The regression target $y = r + \gamma \max_{a'} Q(s', a')$ therefore
moves every time the weights move - and it moves *because* they moved, since the step that
raises $Q(s, a)$ also raises $Q(s', a')$ for every $s'$ the network treats as similar to
$s$, which in CartPole is nearly all of them. The $\max$ then selects whichever action is
currently most over-estimated, so the feedback has a systematic upward bias rather than a
symmetric one. Nothing in the loop damps it.

The numbers say the rest. Mean $|Q|$ is about $0.04$ when training begins at step 1000 and
around $2.7 \times 10^7$ by the end, with the TD loss climbing with it into the millions.
No true CartPole value can exceed $100$, so five orders of magnitude past that ceiling is
not a bad estimate, it is not an estimate. Greedy evaluation reads 8.8 at the first
measurement and 8.8 at every one after: the policy was destroyed within a few hundred
gradient steps, and the episodes of roughly nine steps each are the pole falling over
immediately, again and again.

**`CONFIG_B`, `bootstrap_past_termination: False -> True`. Removed: the base case.**

This one is a single boolean and it is easy to miss what it does. When the pole falls, the
episode is over, and the value of what comes next is zero *by definition* - not estimated,
known. That single fact is the only place in the entire algorithm where the recursion
touches something that is true rather than predicted. Every other value in the table is
defined in terms of another predicted value.

Store `done=False` there and the base case is gone. Now $Q(s, a)$ is regressed onto
$r + \gamma \max Q(s')$ at *every* transition including the terminal one, so the whole
system is a recursion with no anchor. Each sweep multiplies the previous estimate by
$\gamma$ and adds a positive reward, and the $\max$ over an approximator with noisy
outputs biases each step slightly upward. There is nothing to converge to.

Measured over three seeds at twelve thousand steps: mean $|Q|$ finishing at 571, 354 and
367 against a true CartPole ceiling of 100, with the greedy return pinned at the floor
(8.8, 9.8, 8.8) on every one. Left to run five times longer it passes $10^8$ - it does not
settle anywhere, because there is nothing for it to settle to.

Note the difference in *shape* from `CONFIG_A` on the log plot. `CONFIG_A` explodes: the
target chases the network and the feedback is multiplicative, so it reaches $10^7$ within a
few thousand steps. `CONFIG_B` inflates: each sweep adds a little more, smoothly and
relentlessly. Two pathologies of the same leg - `CONFIG_A` removes the thing that holds the
target *still*, `CONFIG_B` removes the thing that keeps it *grounded* - and they do not look
alike.

**Why the other two legs are demonstrated on Baird and not here.**

Because we tried, and CartPole cannot resolve them.

The obvious way to attack the off-policy replay leg is to shrink the buffer. That was
measured over three seeds, and a 32-transition buffer scored a mean final return of 238
against the working configuration's 133 - it is *better*. The reason is instructive:
shrinking the buffer does not only destroy decorrelation, it makes the training data come
from the policy currently being followed, which removes the off-policy leg altogether and
leaves one of the safe pairs. There is no longer a triad to be deadly. Sampling a large
buffer in strict temporal order was tried too, which keeps the replay off-policy while
still destroying decorrelation: mean final return 222, also no worse than working.

The honest conclusion is not "the replay buffer does not matter". It is that CartPole is a
poor witness: four state dimensions, a dense reward, short episodes, and a run-to-run
spread in final return larger than the effect being looked for. A benchmark that cannot
resolve an effect is not evidence that the effect is absent.

Which is exactly why section 2.0 exists. Baird's counterexample has no sampling noise, no
optimiser, no exploration schedule and no reward signal to confound anything - and there
the three legs separate cleanly, with an eigenvalue you can compute rather than a curve you
have to squint at. Use the small exact problem to establish *that* something is true, and
the messy realistic one to see what it looks like when it happens to you. Doing it in the
other order is how people end up quoting a table they have never checked.
<!-- END SOLUTION -->
"""
    ),
    md(
        r"""
## Where this goes next

Notebook 2 keeps all three ingredients and changes what they are made of.

The function approximator becomes a 0.5-billion-parameter language model, the action space
becomes "every token in a 152 000-token vocabulary", and an episode is a single generated
completion. The reward stops being a number the environment hands you and becomes a Python
function you write yourself - a regex for the output format, and a string comparison for the
answer. Choosing that function is choosing what the model will become, which is a heavier
responsibility than choosing a learning rate.

One thing genuinely goes away: GRPO does not bootstrap. It scores a whole group of
completions, normalises the rewards within the group to get advantages, and never
constructs a target out of its own value estimate. One leg of the triad removed - and the
failure modes that replace it, reward hacking and catastrophic forgetting, are the subject
of the next eighty-five minutes.
"""
    ),
]


def main() -> int:
    write(CELLS, OUTPUT, title="RL practice 1: the classics")
    validate(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
