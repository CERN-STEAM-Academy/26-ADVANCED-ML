"""Tests for shared-volume resolution.

These matter operationally rather than theoretically. The base model is not in the
repository, so on a CERN Kubeflow server it comes either from the hub or from a shared
read-only EOS path, and the difference between "found it" and "silently did not" is a
session that starts and one that does not.
"""

from __future__ import annotations

import os

import pytest

from rlpractice import paths


@pytest.fixture(autouse=True)
def _isolate_global_path_state():
    """use_shared() deliberately mutates module state and os.environ so that subprocesses
    inherit the setting. That is right in a notebook and poison in a test suite: one test
    pointing at a tmp_path left every later test loading the model from a directory pytest
    had already deleted. Snapshot and restore around every test in this file."""
    import os as _os

    saved_override = paths._OVERRIDE
    saved_exported = paths._EXPORTED_BY_US
    saved_env = _os.environ.get(paths.SHARED_ENV)
    try:
        yield
    finally:
        paths._OVERRIDE = saved_override
        paths._EXPORTED_BY_US = saved_exported
        if saved_env is None:
            _os.environ.pop(paths.SHARED_ENV, None)
        else:
            _os.environ[paths.SHARED_ENV] = saved_env


@pytest.fixture
def fake_layout(tmp_path, monkeypatch):
    """A fresh clone (empty assets/) plus a shared volume that has the weights."""
    checkout = tmp_path / "checkout"
    shared = tmp_path / "eos" / "rl-practice"
    (checkout / "assets").mkdir(parents=True)
    (shared / "base_model").mkdir(parents=True)
    (shared / "base_model" / paths.MODEL_SENTINEL).write_text("{}")
    (shared / "reference_adapters" / "act1").mkdir(parents=True)

    monkeypatch.setattr(paths, "repo_root", lambda: str(checkout))
    monkeypatch.delenv(paths.SHARED_ENV, raising=False)
    monkeypatch.delenv(paths.MODEL_ENV, raising=False)
    return checkout, shared


def test_without_a_shared_volume_a_fresh_clone_finds_no_model(fake_layout):
    """The honest starting state: cloning the repo does not get you the weights."""
    assert paths.shared_root() is None
    assert paths.model_dir() is None


def test_shared_volume_supplies_the_model(fake_layout, monkeypatch):
    _, shared = fake_layout
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    assert paths.model_dir() == str(shared / "base_model")
    assert paths.asset("reference_adapters") == str(shared / "reference_adapters")


def test_the_checkout_wins_over_the_shared_volume(fake_layout, monkeypatch):
    """Committed artefacts are version-matched to the code that reads them; a stale
    adapter on a shared volume paired with newer code is a silent wrongness."""
    checkout, shared = fake_layout
    (checkout / "assets" / "reference_adapters").mkdir()
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    assert paths.asset("reference_adapters") == str(checkout / "assets" / "reference_adapters")


def test_explicit_model_env_beats_everything(fake_layout, monkeypatch, tmp_path):
    _, shared = fake_layout
    explicit = tmp_path / "somewhere else"
    explicit.mkdir()
    (explicit / paths.MODEL_SENTINEL).write_text("{}")
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    monkeypatch.setenv(paths.MODEL_ENV, str(explicit))
    assert paths.model_dir() == str(explicit)


def test_a_half_finished_download_is_not_accepted_as_a_model(fake_layout, monkeypatch):
    """A directory without config.json fails much later, inside transformers, with a far
    less obvious message. Reject it here instead."""
    checkout, shared = fake_layout
    (checkout / "assets" / "base_model").mkdir()      # exists but empty
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    assert paths.model_dir() == str(shared / "base_model")


def test_a_missing_shared_path_is_reported_not_silently_ignored(fake_layout, monkeypatch, capsys):
    """An EOS mount that has not come up looks exactly like a typo, and the difference
    matters at nine in the morning."""
    monkeypatch.setenv(paths.SHARED_ENV, "/eos/definitely/not/mounted")
    assert paths.shared_root() is None
    assert "not a readable directory" in capsys.readouterr().out


def test_an_empty_shared_env_is_treated_as_unset(fake_layout, monkeypatch):
    monkeypatch.setenv(paths.SHARED_ENV, "   ")
    assert paths.shared_root() is None


def test_require_names_every_path_it_tried(fake_layout, monkeypatch):
    _, shared = fake_layout
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    with pytest.raises(FileNotFoundError) as excinfo:
        paths.require("nonexistent_thing", "run the exercise")
    message = str(excinfo.value)
    assert "run the exercise" in message
    assert str(shared) in message and "assets" in message
    assert paths.SHARED_ENV in message


def test_read_only_shared_volume_is_detected(fake_layout, monkeypatch):
    """Shared volumes are read-only; nothing may try to write to one."""
    checkout, shared = fake_layout
    model = shared / "base_model"
    os.chmod(model, 0o555)
    try:
        assert paths.is_writable(str(checkout)) is True
        # The directory itself is read-only, so nothing may create files in it.
        assert paths.is_writable(str(model)) is False
        # And a path *under* a read-only directory is equally unwritable: is_writable
        # walks up to the nearest directory that exists and asks about that one.
        assert paths.is_writable(str(model / "not_created_yet")) is False
    finally:
        os.chmod(model, 0o755)


def test_describe_reports_every_artefact(fake_layout, monkeypatch):
    _, shared = fake_layout
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    report = paths.describe()
    for expected in ("repository", "base model", "reference adapters", "snapshots", "DQN runs"):
        assert expected in report


def test_nothing_in_this_module_writes(fake_layout, monkeypatch):
    """Resolution is pure lookup. The shared volume is read-only, so a stray mkdir here
    would fail on the one machine that matters and nowhere else."""
    checkout, shared = fake_layout
    monkeypatch.setenv(paths.SHARED_ENV, str(shared))
    before = {str(p) for p in checkout.rglob("*")} | {str(p) for p in shared.rglob("*")}
    paths.describe()
    paths.model_dir()
    paths.asset("snapshots")
    paths.asset("does_not_exist", must_exist=False)
    after = {str(p) for p in checkout.rglob("*")} | {str(p) for p in shared.rglob("*")}
    assert before == after


# --- the tarball is unpacked by hand, so the layout is the thing that goes wrong --------


def test_shared_dir_pointed_one_level_too_high_says_so(tmp_path):
    """Thirty people unpack assets.tar.gz wherever they like. Some will point SHARED_DIR
    at the directory they unpacked *into* rather than the one the archive created, and
    everything downstream would still 'work' until a training cell cannot find weights."""
    unpacked_into = tmp_path / "downloads"
    (unpacked_into / "rl-practice-assets" / "base_model").mkdir(parents=True)
    (unpacked_into / "rl-practice-assets" / "base_model" / paths.MODEL_SENTINEL).write_text("{}")

    with pytest.raises(FileNotFoundError) as excinfo:
        paths.use_shared(str(unpacked_into))
    message = str(excinfo.value)
    assert "does not contain base_model/" in message
    assert "Did you mean" in message
    assert "rl-practice-assets" in message


def test_shared_dir_with_no_base_model_anywhere_explains_the_rule(tmp_path):
    empty = tmp_path / "somewhere"
    (empty / "notes").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        paths.use_shared(str(empty))
    message = str(excinfo.value)
    assert "directly contains" in message
    assert "notes" in message


def test_a_correctly_pointed_shared_dir_is_accepted(tmp_path, monkeypatch):
    good = tmp_path / "rl-practice-assets"
    (good / "base_model").mkdir(parents=True)
    (good / "base_model" / paths.MODEL_SENTINEL).write_text("{}")
    monkeypatch.setattr(paths, "repo_root", lambda: str(tmp_path / "checkout"))
    (tmp_path / "checkout" / "assets").mkdir(parents=True)
    assert paths.use_shared(str(good), verbose=False) == str(good)
    assert paths.model_dir() == str(good / "base_model")
    paths.use_shared(None, verbose=False)


def test_use_shared_none_does_not_unset_the_environment(monkeypatch, tmp_path):
    """An image can bake the assets in and announce them with ENV RLPRACTICE_SHARED_DIR.
    The notebook's default SHARED_DIR = None must fall back to that, not delete it -
    otherwise the model goes missing on the one machine where it was already present."""
    baked = tmp_path / "opt" / "rl-practice-assets"
    (baked / "base_model").mkdir(parents=True)
    (baked / "base_model" / paths.MODEL_SENTINEL).write_text("{}")
    monkeypatch.setattr(paths, "repo_root", lambda: str(tmp_path / "checkout"))
    (tmp_path / "checkout" / "assets").mkdir(parents=True)
    monkeypatch.setenv(paths.SHARED_ENV, str(baked))

    assert paths.use_shared(None, verbose=False) == str(baked)
    assert os.environ.get(paths.SHARED_ENV) == str(baked), "the env var was unset"
    assert paths.model_dir() == str(baked / "base_model")
