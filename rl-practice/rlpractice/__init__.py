"""Support code for the CERN STEAM Academy 2026 reinforcement-learning practice session.

Notebook 1 (classics) uses :mod:`rlpractice.dqn`.
Notebook 2 (GRPO) uses everything else:

* :mod:`rlpractice.arithmetic`   - the task, generated from a seed, never downloaded
* :mod:`rlpractice.rewards`      - the objective, which is just two Python functions
* :mod:`rlpractice.evaluation`   - FROZEN before/after measurements; do not parameterise
* :mod:`rlpractice.general_text` - the embedded prose corpus the forgetting probe reads
* :mod:`rlpractice.callbacks`    - metrics logging, NaN guard, measured time budget
* :mod:`rlpractice.dashboard`    - the plots

Nothing here reaches the network at import time.
"""

__version__ = "1.0.0"

__all__ = [
    "arithmetic",
    "callbacks",
    "dashboard",
    "dqn",
    "evaluation",
    "general_text",
    "rewards",
]
