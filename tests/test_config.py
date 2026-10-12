"""Config loading and the VERSION read."""

import json
from pathlib import Path


def test_explicit_path_merges_over_defaults(backend, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "custom/model"}), encoding="utf-8")
    config = backend.load_config(path)
    assert config["model"] == "custom/model"
    assert config["target_language"] == "English"  # default preserved


def test_missing_file_yields_defaults(backend, tmp_path):
    config = backend.load_config(tmp_path / "absent.json")
    assert config["model"] == backend.DEFAULT_CONFIG["model"]


def test_corrupt_file_yields_defaults(backend, tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    config = backend.load_config(path)
    assert config["model"] == backend.DEFAULT_CONFIG["model"]


def test_explicit_path_never_writes(backend, tmp_path):
    # load_config(path=...) skips migration, so a test can't touch the real home.
    backend.load_config(tmp_path / "absent.json")
    assert list(tmp_path.iterdir()) == []


def test_version_is_read_from_the_file(backend, tmp_path, monkeypatch):
    version_file = tmp_path / "VERSION"
    version_file.write_text("1.2.3\n", encoding="utf-8")
    monkeypatch.setattr(backend, "VERSION_FILE", version_file)
    assert backend.get_version() == "1.2.3"


def test_unparseable_version_is_dev(backend, tmp_path, monkeypatch):
    version_file = tmp_path / "VERSION"
    version_file.write_text("not-a-version", encoding="utf-8")
    monkeypatch.setattr(backend, "VERSION_FILE", version_file)
    assert backend.get_version() == "dev"


def test_missing_version_file_is_dev(backend, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "VERSION_FILE", tmp_path / "absent")
    assert backend.get_version() == "dev"


# --- import hygiene: importing backend must not touch the filesystem --------

_REPO = Path(__file__).resolve().parent.parent


def _probe(tmp_path, statement):
    """Run `statement` in a subprocess with a fresh DICH_HOME; return stdout."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "DICH_HOME": str(tmp_path / "home")}
    result = subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_creates_nothing(tmp_path):
    out = _probe(
        tmp_path,
        """
import os
from pathlib import Path
import dich.backend  # noqa: F401
print("clean" if not Path(os.environ["DICH_HOME"]).exists() else "dirty")
""",
    )
    assert out == "clean"


def test_init_config_creates_and_migrates(tmp_path):
    out = _probe(
        tmp_path,
        """
import os
from pathlib import Path
import dich.backend as backend
backend.init_config()
home = Path(os.environ["DICH_HOME"])
print("config:", (home / "config.json").exists())
print("models:", (home / "models").is_dir())
""",
    )
    assert out == "config: True\nmodels: True"


def test_resource_path_resolves_version(backend):
    assert backend.resource_path("VERSION") == backend.VERSION_FILE
    version = (_REPO / "dich" / "VERSION").read_text(encoding="utf-8").strip()
    assert backend.get_version() == version


def test_resource_path_when_frozen(backend, tmp_path, monkeypatch):
    internal = tmp_path / "_internal"
    (internal / "dich").mkdir(parents=True)
    (internal / "dich" / "VERSION").write_text("9.9.9", encoding="utf-8")
    monkeypatch.setattr(backend, "IS_FROZEN", True)
    monkeypatch.setattr(backend.sys, "_MEIPASS", str(internal), raising=False)
    assert backend.resource_path("VERSION") == internal / "dich" / "VERSION"


def test_init_config_reads_disk_and_refreshes_globals(backend, tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "custom/model"}), encoding="utf-8")
    monkeypatch.setattr(backend, "CONFIG_PATH", path)
    names = (
        "CONFIG",
        "MODEL_NAME",
        "SOURCE_LANGUAGE",
        "TARGET_LANGUAGE",
        "_CONFIG_LOADED",
    )
    snapshot = {n: getattr(backend, n) for n in names}
    try:
        backend._CONFIG_LOADED = False
        backend.init_config()
        assert backend.MODEL_NAME == "custom/model"
        assert backend.init_config() is backend.CONFIG  # idempotent
    finally:
        for name, value in snapshot.items():
            setattr(backend, name, value)
