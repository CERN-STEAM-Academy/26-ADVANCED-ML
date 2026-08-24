"""Plotting for the GRPO training logs.

The live dashboard is not an animation. It clears the notebook output and redraws the
whole figure from the CSV on disk. That is deliberate: a redraw cannot get out of sync
with the log, it survives a kernel that has been sitting idle, and it works identically
whether the numbers came from a live run or from a pre-staged reference CSV. Simple and
robust beats clever, especially in front of an audience.

The six panels are chosen to be exactly the evidence Act 3 asks students to reason about:
what the objective was rewarding, what it was silently spending, and what it never
constrained at all.
"""

from __future__ import annotations

import math
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: Schema of the training log CSV. ``callbacks.MetricsCSVCallback`` writes exactly these
#: columns, in this order, and every plot here reads from them. Values that were not
#: measured on a given step are written as empty (NaN) rather than forward-filled, so
#: that "we did not look" and "we looked and it was zero" stay distinguishable.
LOG_COLUMNS: list[str] = [
    "step",
    "wall_seconds",
    "loss",
    "learning_rate",
    "reward_mean",
    "reward_std",
    "frac_zero_std_groups",
    "completion_length_mean",
    "entropy",
    "kl",
    "general_ppl",
    "task_accuracy",
]


def load_log(path: str | os.PathLike) -> pd.DataFrame:
    """Read a training log CSV, sorted by step, with numeric dtypes."""
    df = pd.read_csv(path)
    missing = [c for c in LOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns {missing}")
    return df.sort_values("step").reset_index(drop=True)


def series(df: pd.DataFrame, column: str) -> tuple[Sequence[float], Sequence[float]]:
    """Step/value pairs for a column, with unmeasured rows dropped.

    Metrics are logged at different cadences - perplexity every few steps, accuracy
    rarely - so most columns are mostly empty. Dropping the empty rows here rather than
    forward-filling them keeps "we did not look" distinct from "we looked and it was
    unchanged", which matters when the plot is the evidence.
    """
    sub = df[["step", column]].dropna()
    return sub["step"].to_numpy(), sub[column].to_numpy()


#: Kept as a private alias because the plotting helpers below were written against it.
_series = series


def _line(ax, df, column, title, ylabel, color="C0", **kwargs):
    steps, values = _series(df, column)
    ax.plot(steps, values, color=color, marker="." if len(steps) < 40 else None, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    return steps, values


def plot_dashboard(
    log_path: str | os.PathLike,
    title: str = "GRPO training",
    baseline_ppl: float | None = None,
    baseline_accuracy: float | None = None,
    figsize: tuple[float, float] = (15, 7.5),
):
    """Draw the 2x3 diagnostic grid from a training log CSV.

    ``baseline_ppl`` and ``baseline_accuracy`` are the Act 0 "before" numbers. Drawing
    them as horizontal reference lines is what turns a wiggly curve into a claim.
    """
    df = load_log(log_path)
    if len(df) == 0:
        # pandas gives object dtype to every column of a zero-row frame, which reaches
        # matplotlib as an object array and raises an opaque ufunc error. A log can be
        # empty for an ordinary reason: the header is written before the step-0 row.
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, f"{os.path.basename(str(log_path))} has no rows yet",
                ha="center", va="center", fontsize=11)
        return fig
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f"{title}  ({len(df)} steps logged)", fontsize=12)

    # (0,0) reward, with the +/- std band, and eval accuracy on a twin axis.
    ax = axes[0][0]
    steps, mean = _series(df, "reward_mean")
    _, std = _series(df, "reward_std")
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    ax.plot(steps, mean, color="C0", label="reward mean")
    if len(std) == len(mean):
        ax.fill_between(steps, mean - std, mean + std, color="C0", alpha=0.2, label="+/- std")
    ax.set_title("Reward (what we asked for)", fontsize=10)
    ax.set_xlabel("training step", fontsize=8)
    ax.set_ylabel("reward", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)
    acc_steps, acc = _series(df, "task_accuracy")
    if len(acc_steps):
        twin = ax.twinx()
        twin.plot(acc_steps, acc, color="C2", marker="o", ms=4, label="held-out accuracy")
        twin.set_ylabel("held-out accuracy", fontsize=8, color="C2")
        twin.tick_params(labelsize=8, colors="C2")
        twin.set_ylim(0, 1)
        if baseline_accuracy is not None:
            twin.axhline(baseline_accuracy, color="C2", ls=":", lw=1)
    ax.legend(fontsize=7, loc="lower right")

    # (0,1) general perplexity: the thing nobody was optimising.
    ax = axes[0][1]
    _line(ax, df, "general_ppl", "General-text perplexity (what we spent)", "perplexity", color="C3")
    if baseline_ppl is not None:
        ax.axhline(baseline_ppl, color="C3", ls=":", lw=1)
        ax.text(0.02, 0.04, f"baseline {baseline_ppl:.2f}", transform=ax.transAxes, fontsize=7, color="C3")

    # (0,2) KL to the reference policy: unbounded when beta = 0.
    ax = axes[0][2]
    _line(ax, df, "kl", "KL to reference policy (the leash)", "KL", color="C1")

    # (1,0) policy entropy.
    _line(axes[1][0], df, "entropy", "Policy entropy (exploration)", "nats / token", color="C4")

    # (1,1) completion length.
    _line(axes[1][1], df, "completion_length_mean", "Mean completion length", "tokens", color="C5")

    # (1,2) zero-std groups: the fraction of prompts contributing no gradient at all.
    ax = axes[1][2]
    _line(ax, df, "frac_zero_std_groups", "Groups with zero reward std", "fraction", color="C6")
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_scissors(
    log_path: str | os.PathLike,
    title: str = "The scissors",
    baseline_ppl: float | None = None,
    baseline_accuracy: float | None = None,
    ax=None,
):
    """Task accuracy and general perplexity on twin axes: the figure from the slides.

    One blade goes up (the thing we rewarded) and the other goes up too (the thing we
    did not, where up is bad). The gap between them is the cost of the objective.
    """
    df = load_log(log_path)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    acc_steps, acc = _series(df, "task_accuracy")
    ppl_steps, ppl = _series(df, "general_ppl")

    if baseline_accuracy is not None:
        acc_steps = [0, *acc_steps] if (len(acc_steps) == 0 or acc_steps[0] != 0) else acc_steps
        acc = [baseline_accuracy, *acc] if len(acc) < len(acc_steps) else acc

    ax.plot(acc_steps, acc, color="C2", marker="o", ms=4, label="task accuracy (held out)")
    ax.set_xlabel("training step")
    ax.set_ylabel("task accuracy", color="C2")
    ax.tick_params(axis="y", colors="C2")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    twin = ax.twinx()
    twin.plot(ppl_steps, ppl, color="C3", label="general-text perplexity")
    twin.set_ylabel("general-text perplexity", color="C3")
    twin.tick_params(axis="y", colors="C3")
    if baseline_ppl is not None:
        twin.axhline(baseline_ppl, color="C3", ls=":", lw=1)

    ax.set_title(title)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="center right")
    return ax.figure


def plot_scissors_comparison(
    log_paths: Sequence[str | os.PathLike],
    labels: Sequence[str],
    baseline_ppl: float | None = None,
    baseline_accuracy: float | None = None,
    title: str = "Act 1 vs Act 4: does the KL leash close the blades?",
    figsize: tuple[float, float] = (11, 4.5),
):
    """Overlay several runs' scissors plots. This is the Act 4 payoff figure.

    Accuracy is drawn on the left panel and perplexity on the right, rather than four
    curves on twin axes of one panel, because four curves on twin axes is unreadable
    from the back of a lecture room.
    """
    if len(log_paths) != len(labels):
        raise ValueError(f"{len(log_paths)} logs but {len(labels)} labels")
    frames = [load_log(p) for p in log_paths]

    fig, (ax_acc, ax_ppl) = plt.subplots(1, 2, figsize=figsize)
    for i, (df, label) in enumerate(zip(frames, labels)):
        color = f"C{i}"
        steps, acc = _series(df, "task_accuracy")
        ax_acc.plot(steps, acc, color=color, marker="o", ms=4, label=label)
        steps, ppl = _series(df, "general_ppl")
        ax_ppl.plot(steps, ppl, color=color, label=label)

    if baseline_accuracy is not None:
        ax_acc.axhline(baseline_accuracy, color="k", ls=":", lw=1, label="base model")
    if baseline_ppl is not None:
        ax_ppl.axhline(baseline_ppl, color="k", ls=":", lw=1, label="base model")

    ax_acc.set_title("Task accuracy (held out) - higher is better", fontsize=10)
    ax_acc.set_xlabel("training step")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0, 1)

    ax_ppl.set_title("General-text perplexity - lower is better", fontsize=10)
    ax_ppl.set_xlabel("training step")
    ax_ppl.set_ylabel("perplexity")

    for ax in (ax_acc, ax_ppl):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def redraw(log_path, display_handle=None, **kwargs):
    """Clear the cell output and redraw the dashboard. Called from the live callback."""
    from IPython.display import clear_output, display

    clear_output(wait=True)
    fig = plot_dashboard(log_path, **kwargs)
    display(fig)
    plt.close(fig)


def summarise(log_path: str | os.PathLike) -> pd.DataFrame:
    """First/last values of every logged quantity, as a small table.

    Used in Act 3, where the point is to read the numbers rather than squint at curves.
    """
    df = load_log(log_path)
    rows = []
    for column in LOG_COLUMNS:
        if column in ("step", "wall_seconds"):
            continue
        steps, values = _series(df, column)
        if len(values) == 0:
            continue
        first, last = float(values[0]), float(values[-1])
        change = last - first
        rows.append(
            {
                "quantity": column,
                "first": round(first, 4),
                "last": round(last, 4),
                "change": round(change, 4),
                "ratio": round(last / first, 3) if first not in (0.0,) and not math.isnan(first) else float("nan"),
                "measured_at_steps": len(values),
            }
        )
    return pd.DataFrame(rows)
