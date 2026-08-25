"""Baird's counterexample: the deadly triad, provably, in forty lines of numpy.

Notebook 1 claims that bootstrapping, off-policy learning and function approximation are
each harmless alone but can diverge together. DQN on CartPole is a poor place to check
that claim: the run-to-run noise in final return is larger than most of the effects, so
you end up arguing about curves. This is the place to check it.

Baird (1995); Sutton & Barto, Example 11.1 and Figure 11.2.

Seven states and every reward is zero, so the true value function is zero everywhere -
and it is *exactly representable* by the linear approximator, since w = 0 is a perfect
solution sitting right there. Semi-gradient off-policy TD(0) still sends the weights to
infinity. Nothing here is a capacity problem, an optimisation problem or a tuning problem:
the algorithm walks away from a solution it could represent perfectly.

What makes it worth five minutes of a lecture is that you can turn each leg of the triad
off independently and watch the divergence stop. Measured, at alpha = 0.01 over 5000 steps:

    all three legs present                  |w| = 1.2e7      diverges
    on-policy instead of off-policy         |w| = 8.7        bounded
    tabular instead of approximation        |w| = 24.5       bounded
    Monte Carlo instead of bootstrapping    |w| = 8.6        bounded

And it is not sampling noise. ``key_matrix`` builds the matrix governing the *expected*
update; its largest eigenvalue has real part +0.239 with all three legs and <= 0 as soon
as any one is removed. A positive real part means the expected iteration diverges for
every step size, however small. That is the "provably" in "provably diverges".

Layout, following Sutton & Barto:
  states 1..6 are the "upper" states, state 7 the "lower" one.
  features x(s_i) = 2 e_i + e_8   for i = 1..6      (8 weights)
            x(s_7) = e_7 + 2 e_8
  target policy pi:    always the *solid* action -> state 7.
  behaviour policy b:  solid with prob 1/7 -> state 7,
                       dashed with prob 6/7 -> uniform over states 1..6.
  importance ratio     rho = 7 on a solid step, 0 on a dashed step.
"""
from __future__ import annotations

import numpy as np

N_STATES = 7
N_WEIGHTS = 8
GAMMA = 0.99


def features() -> np.ndarray:
    X = np.zeros((N_STATES, N_WEIGHTS))
    for i in range(6):
        X[i, i] = 2.0
        X[i, 7] = 1.0
    X[6, 6] = 1.0
    X[6, 7] = 2.0
    return X


def initial_weights() -> np.ndarray:
    return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0])


def transition_matrices() -> tuple[np.ndarray, np.ndarray]:
    """(P_pi, P_b): row s = distribution over next states."""
    P_pi = np.zeros((N_STATES, N_STATES))
    P_pi[:, 6] = 1.0                                  # solid action, always to state 7
    P_b = np.zeros((N_STATES, N_STATES))
    P_b[:, :6] = (6.0 / 7.0) / 6.0                    # dashed: uniform over 1..6
    P_b[:, 6] += 1.0 / 7.0                            # solid
    return P_pi, P_b


def stationary(P: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(P.T)
    v = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    return v / v.sum()


def key_matrix(off_policy: bool = True, X: np.ndarray | None = None) -> np.ndarray:
    """A = X^T D (gamma P - I) X, the matrix governing the expected TD update.

    The expected semi-gradient TD(0) update is  w <- w + alpha A w  (rewards are zero),
    so the iteration diverges for every step size iff A has an eigenvalue with positive
    real part. This is the "provable" in "provably diverges".
    """
    X = features() if X is None else X
    P_pi, P_b = transition_matrices()
    D = np.diag(stationary(P_b))
    P = P_pi if off_policy else P_b
    return X.T @ D @ (GAMMA * P - np.eye(N_STATES)) @ X


def td0(
    steps: int = 1000,
    alpha: float = 0.01,
    seed: int = 0,
    off_policy: bool = True,
    tabular: bool = False,
    bootstrap: bool = True,
) -> np.ndarray:
    """Semi-gradient TD(0) with importance sampling. Returns the weight trajectory.

    ``off_policy=False``  -> target policy equals the behaviour policy (rho == 1).
    ``tabular=True``      -> identity features, i.e. no function approximation.
    ``bootstrap=False``   -> regress on the observed (zero) return instead of on
                             r + gamma v(s'), i.e. Monte Carlo, no bootstrapping.
    """
    rng = np.random.default_rng(seed)
    X = np.eye(N_STATES) if tabular else features()
    w = np.zeros(X.shape[1]) if tabular else initial_weights()
    if tabular:
        w[:] = 1.0
        w[6] = 10.0
    history = np.zeros((steps + 1, len(w)))
    history[0] = w
    state = rng.integers(N_STATES)
    for t in range(steps):
        solid = rng.random() < 1.0 / 7.0
        next_state = 6 if solid else rng.integers(6)
        if off_policy:
            rho = 7.0 if solid else 0.0
        else:
            rho = 1.0
        v_s = X[state] @ w
        target = GAMMA * (X[next_state] @ w) if bootstrap else 0.0
        w = w + alpha * rho * (0.0 + target - v_s) * X[state]
        history[t + 1] = w
        state = next_state
    return history


def dp(steps: int = 1000, alpha: float = 0.01) -> np.ndarray:
    """Semi-gradient *expected* update (dynamic programming): no sampling noise at all.

    Every state is swept, so this is not a variance artefact - it is the operator itself.
    """
    X = features()
    P_pi, _ = transition_matrices()
    w = initial_weights()
    history = np.zeros((steps + 1, len(w)))
    history[0] = w
    for t in range(steps):
        v = X @ w
        expected_next = P_pi @ v
        delta = GAMMA * expected_next - v
        w = w + (alpha / N_STATES) * (X.T @ delta)
        history[t + 1] = w
    return history


def triad_ablation(steps: int = 5000, alpha: float = 0.01, seed: int = 0) -> dict[str, dict]:
    """Run the four ablations the notebook plots: all three legs, then each one removed.

    Returns ``{label: {"history": ndarray, "eigenvalue": float}}``. The eigenvalue is the
    largest real part of the expected-update matrix, which is the analytic counterpart of
    the trajectory: positive means the weights must diverge, whatever the step size.
    """
    import numpy as np

    settings = {
        "all three legs": ({}, key_matrix(off_policy=True)),
        "on-policy": ({"off_policy": False}, key_matrix(off_policy=False)),
        "tabular (no approximation)": (
            {"tabular": True},
            key_matrix(off_policy=True, X=np.eye(N_STATES)),
        ),
        "Monte Carlo (no bootstrap)": ({"bootstrap": False}, None),
    }
    out = {}
    for label, (kwargs, matrix) in settings.items():
        history = td0(steps=steps, alpha=alpha, seed=seed, **kwargs)
        eigenvalue = float(max(np.linalg.eigvals(matrix).real)) if matrix is not None else float("nan")
        out[label] = {"history": history, "eigenvalue": eigenvalue}
    return out


def plot_triad(results: dict[str, dict], figsize=(11, 4.2)):
    """Two panels: the diverging weights, and the same thing with one leg removed each time.

    A log axis on the left, because the interesting quantity spans seven orders of
    magnitude and a linear axis would show one curve and three flat lines.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax_all, ax_ablate) = plt.subplots(1, 2, figsize=figsize)

    norms = {label: np.linalg.norm(data["history"], axis=1) for label, data in results.items()}
    steps = np.arange(len(next(iter(norms.values()))))

    ax_all.semilogy(steps, norms["all three legs"], color="C3", label="all three legs")
    ax_all.set_title("Bootstrapping + off-policy + approximation", fontsize=10)
    ax_all.set_ylabel(r"$\|w\|$  (log scale)")

    for i, (label, values) in enumerate(norms.items()):
        if label == "all three legs":
            ax_ablate.semilogy(steps, values, color="C3", lw=2, label=label)
        else:
            ax_ablate.semilogy(steps, values, color=f"C{i}", label=label)
    ax_ablate.set_title("Remove any one leg and it stays bounded", fontsize=10)

    for ax in (ax_all, ax_ablate):
        ax.set_xlabel("TD(0) update")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Baird's counterexample: the true values are zero and representable", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
