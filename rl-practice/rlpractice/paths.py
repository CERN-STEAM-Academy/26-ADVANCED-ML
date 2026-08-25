"""Where to find things that are too big to commit.

The repository deliberately does not contain the base model: it is roughly a gigabyte of
weights, it is not source, and thirty students each downloading it at the same moment is
not a plan. So the model - and optionally the pre-staged reference artefacts - can live on
a **shared, read-only filesystem**, which at CERN means an EOS path such as::

    /eos/project/.../rl-practice/

Point the session at it once. From a notebook - which is where this actually happens -
put the path in the configuration cell at the top and call::

    paths.use_shared("/eos/project/s/steam/rl-practice")

From a shell, or in a Kubeflow notebook-server spec where an administrator sets it for
everyone at once, use the environment variable instead::

    export RLPRACTICE_SHARED_DIR=/eos/project/s/steam/rl-practice

Both work, and ``use_shared`` also exports the variable so that anything the notebook
shells out to - ``!python tools/prestage.py``, say - inherits it. Do not fight with
``%env`` and kernel restarts; there is a variable in the first cell for this.

Design rules, all of which exist because the shared volume is read-only and because a
session that fails at the wrong moment is worse than one that never started:

* **Nothing here ever writes.** Resolution is pure lookup. Training output, logs and
  snapshots always go to the writable repository checkout, never to the shared volume.
* **The repository wins over the shared volume.** Committed artefacts are version-matched
  to the code that reads them; a stale adapter on a shared volume paired with newer code
  is a silent wrongness, and silent wrongness is the thing worth engineering against. The
  base model is the exception in practice only because the repository has no copy of it.
* **Resolution is reported, not guessed.** ``describe()`` prints exactly which candidate
  won for every artefact. When something is missing at nine in the morning in front of a
  room, the useful thing is a list of the paths that were tried.

This module imports nothing but the standard library on purpose, so that notebook 1 can
use it without dragging in torch.
"""

from __future__ import annotations

import os

#: Points at a shared, usually read-only, directory laid out like the repository's
#: ``assets/`` - so ``$RLPRACTICE_SHARED_DIR/base_model``, ``.../reference_adapters`` and
#: so on. Unset means "everything is in the checkout".
SHARED_ENV = "RLPRACTICE_SHARED_DIR"

#: Overrides the base-model location specifically, beating every other candidate. Useful
#: when the weights are somewhere that does not follow the assets layout.
MODEL_ENV = "RLPRACTICE_MODEL_DIR"

#: A directory is accepted as a model only if it contains this. Checking for the directory
#: alone would happily accept a half-finished download, which then fails much later inside
#: transformers with a far less obvious message.
MODEL_SENTINEL = "config.json"


def repo_root() -> str:
    """The repository checkout containing this package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_assets() -> str:
    return os.path.join(repo_root(), "assets")


#: Set by ``use_shared`` from a notebook cell. Takes precedence over the environment, so
#: that a student who edits the configuration cell does not have to know or care what the
#: notebook server was launched with.
_OVERRIDE: str | None = None


def use_shared(path: str | None, verbose: bool = True) -> str | None:
    """Point this session at a shared assets directory. Returns the resolved path.

    Call it from the configuration cell at the top of a notebook::

        SHARED_DIR = "/eos/project/s/steam/rl-practice"   # or None
        paths.use_shared(SHARED_DIR)

    Passing None clears the override and falls back to the environment variable, which is
    what an instructor would set on the notebook server for a whole class.

    This also exports ``RLPRACTICE_SHARED_DIR`` into ``os.environ`` so that anything the
    notebook shells out to inherits the setting, and it validates the path immediately -
    finding out that an EOS mount is not there is much better at the top of the notebook
    than forty minutes in.
    """
    global _OVERRIDE

    if path is None:
        _OVERRIDE = None
        os.environ.pop(SHARED_ENV, None)
        if verbose:
            print("[paths] using the repository checkout only.\n" + describe())
        return None

    path = os.path.expanduser(str(path).strip())
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{path!r} is not a readable directory.\n"
            "If it is an EOS path, check that the mount is up (try `ls /eos`) and that "
            "your Kerberos token is valid. If you meant to use the copy inside the "
            "repository, set SHARED_DIR = None."
        )

    _OVERRIDE = path
    os.environ[SHARED_ENV] = path
    if verbose:
        print(describe())
    return path


def shared_root() -> str | None:
    """The shared assets directory, if one is configured and actually present.

    A configured-but-absent path is reported rather than silently ignored: on a Kubeflow
    notebook server an EOS mount that has not come up looks exactly like a typo, and the
    difference matters.
    """
    if _OVERRIDE is not None:
        return _OVERRIDE
    raw = os.environ.get(SHARED_ENV, "").strip()
    if not raw:
        return None
    if not os.path.isdir(raw):
        print(
            f"[paths] {SHARED_ENV}={raw!r} is set but is not a readable directory. "
            "Falling back to the repository checkout. If this is an EOS path, check that "
            "the mount is up and that your krb5 token is valid."
        )
        return None
    return raw


def candidates(relative: str) -> list[str]:
    """Every place ``relative`` might live, in priority order."""
    found = [os.path.join(repo_assets(), relative)]
    shared = shared_root()
    if shared:
        found.append(os.path.join(shared, relative))
    return found


def asset(relative: str, must_exist: bool = True) -> str | None:
    """Resolve an artefact under ``assets/``, searching the shared volume as a fallback.

    Returns None when it is nowhere, so that callers can choose their own failure - some
    of them have a perfectly good fallback (the model can come from the hub) and some do
    not.
    """
    for path in candidates(relative):
        if os.path.exists(path):
            return path
    return None if must_exist else candidates(relative)[0]


def model_dir() -> str | None:
    """The base model directory, or None if it must come from the hub.

    Order: ``$RLPRACTICE_MODEL_DIR``, then ``assets/base_model`` in the checkout, then
    ``$RLPRACTICE_SHARED_DIR/base_model``.
    """
    explicit = os.environ.get(MODEL_ENV, "").strip()
    if explicit:
        if os.path.exists(os.path.join(explicit, MODEL_SENTINEL)):
            return explicit
        print(
            f"[paths] {MODEL_ENV}={explicit!r} is set but contains no {MODEL_SENTINEL}. "
            "Ignoring it and looking elsewhere."
        )

    for path in candidates("base_model"):
        if os.path.exists(os.path.join(path, MODEL_SENTINEL)):
            return path
    return None


def is_writable(path: str) -> bool:
    """Can we create files under this directory? Shared volumes are usually read-only."""
    while path and not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            return False
        path = parent
    return os.access(path, os.W_OK)


def describe() -> str:
    """A printable resolution report. Notebooks show this before anything else runs."""
    shared = shared_root()
    lines = [
        f"  repository            {repo_root()}",
        f"  {SHARED_ENV:<22}{shared or '(unset - using the checkout only)'}",
    ]
    if shared:
        lines.append(f"  shared volume writable {is_writable(shared)}  (read-only is expected and fine)")

    model = model_dir()
    lines.append(f"  base model            {model or '(not found locally - will download from the hub)'}")

    for name, relative in (
        ("reference adapters", "reference_adapters"),
        ("reference logs", "reference_logs"),
        ("snapshots", "snapshots"),
        ("DQN runs", "dqn"),
    ):
        resolved = asset(relative)
        lines.append(f"  {name:<22}{resolved or '(missing)'}")
    return "\n".join(lines)


def require(relative: str, why: str) -> str:
    """Resolve an artefact or fail with a message that says what to do about it."""
    path = asset(relative)
    if path is not None:
        return path
    tried = "\n".join(f"    {candidate}" for candidate in candidates(relative))
    raise FileNotFoundError(
        f"Could not find '{relative}', which is needed to {why}.\nLooked in:\n{tried}\n"
        f"Either run tools/prestage.py to build it, or set {SHARED_ENV} to a directory "
        "that contains it (on a CERN Kubeflow server this is usually an /eos path)."
    )
