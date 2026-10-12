"""Test bootstrap.

`backend` reads DICH_HOME and may write config at import time, so the redirect
must happen before any test module imports it. Everything lands in a throwaway
directory; the real ~/.config/dich is never touched.
"""
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

_SANDBOX = Path(tempfile.mkdtemp(prefix="dich-tests-"))
os.environ["DICH_HOME"] = str(_SANDBOX / "dich")
os.environ.setdefault("HF_HUB_CACHE", str(_SANDBOX / "hf"))

import pytest


@pytest.fixture()
def backend():
    """The backend module."""
    import backend as b
    return b


@pytest.fixture()
def backend_cache(backend, tmp_path, monkeypatch):
    """Backend with CACHE_DIR isolated per test and clean model globals."""
    monkeypatch.setattr(backend, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(backend, "MODEL_NAME", "test/model")
    monkeypatch.setattr(backend, "SOURCE_LANGUAGE", "Vietnamese")
    monkeypatch.setattr(backend, "TARGET_LANGUAGE", "English")
    return backend


@pytest.fixture()
def requires_display():
    """Skip instead of error when no Tk session exists (headless CI)."""
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.withdraw()
    yield root
    root.destroy()
