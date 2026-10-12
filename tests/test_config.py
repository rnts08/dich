"""Config loading and the VERSION read."""
import json


def test_explicit_path_merges_over_defaults(backend, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": "custom/model"}), encoding="utf-8")
    config = backend.load_config(path)
    assert config["model"] == "custom/model"
    assert config["target_language"] == "English"     # default preserved


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
