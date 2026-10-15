"""
Model management: what is cached, how big it is, and unloading.

Regression cover for the reason this exists: the model entry used to be free
text, so a typo silently started a multi-gigabyte download with no size shown
and nothing indicating the model was new.
"""

from types import SimpleNamespace

import pytest


def _make_cached(models_dir, repo, megabytes=1):
    snap = models_dir / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    blob = snap / "model.safetensors"
    blob.write_bytes(b"0" * (megabytes * 1024 * 1024))
    return blob


def test_human_bytes_reads_like_a_person_writes_it(backend):
    assert backend.human_bytes(512) == "512 B"
    assert backend.human_bytes(1536) == "1.5 KB"
    assert backend.human_bytes(2 * 1024**3) == "2.0 GB"


def test_cached_models_lists_repos_and_sizes(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(backend, "MODELS_DIR", tmp_path)
    _make_cached(tmp_path, "org/small", megabytes=1)
    _make_cached(tmp_path, "org/big", megabytes=3)

    models = backend.cached_models()

    assert [m["model"] for m in models] == ["org/big", "org/small"]  # biggest first
    assert models[0]["size"] == "3.0 MB"
    assert models[0]["bytes"] == 3 * 1024 * 1024


def test_cached_models_is_empty_when_nothing_is_cached(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(backend, "MODELS_DIR", tmp_path)
    assert backend.cached_models() == []


def test_is_model_cached_needs_a_snapshot_not_just_a_dir(
    tmp_path, monkeypatch, backend
):
    monkeypatch.setattr(backend, "MODELS_DIR", tmp_path)
    # A bare directory (an interrupted download) is not a usable model.
    (tmp_path / "models--org--repo").mkdir()
    assert backend.is_model_cached("org/repo") is False

    _make_cached(tmp_path, "org/repo")
    assert backend.is_model_cached("org/repo") is True


def test_model_size_is_none_for_an_uncached_repo(tmp_path, monkeypatch, backend):
    monkeypatch.setattr(backend, "MODELS_DIR", tmp_path)
    assert backend.model_size_bytes("org/missing") is None


def test_estimated_ram_prefers_the_disk_size_when_cached(
    tmp_path, monkeypatch, backend
):
    monkeypatch.setattr(backend, "MODELS_DIR", tmp_path)
    _make_cached(tmp_path, "org/repo", megabytes=100)

    def must_not_hit_the_network(repo):
        raise AssertionError("cached models must not need the Hub")

    monkeypatch.setattr(backend, "estimated_download_bytes", must_not_hit_the_network)

    assert backend.estimated_ram_bytes("org/repo") == int(100 * 1024**2 * 1.25)


def test_estimated_download_returns_none_when_offline(monkeypatch, backend):
    import httpx

    def boom(*_a, **_k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(backend.httpx, "get", boom)

    assert backend.estimated_download_bytes("org/repo") is None


def test_estimated_download_sums_the_siblings(monkeypatch, backend):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"siblings": [{"size": 100}, {"size": 200}, {"no_size": True}]}

    monkeypatch.setattr(backend.httpx, "get", lambda *a, **k: FakeResponse())

    assert backend.estimated_download_bytes("org/repo") == 300


def test_unload_models_frees_everything_but_the_keeper(monkeypatch, backend):
    freed = []

    class FakeEngine:
        def __init__(self, name):
            self.model_name = name

        def unload(self):
            freed.append(self.model_name)

    monkeypatch.setattr(
        backend,
        "_ENGINES",
        {"keep/me": FakeEngine("keep/me"), "drop/too": FakeEngine("drop/too")},
    )

    names = backend.unload_models(keep="keep/me")

    assert names == ["drop/too"]
    assert freed == ["drop/too"]
    assert "keep/me" in backend._ENGINES
    assert "drop/too" not in backend._ENGINES


def test_unload_models_never_raises_when_an_engine_resists(monkeypatch, backend):
    class Angry:
        model_name = "angry/model"

        def unload(self):
            raise RuntimeError("still attached")

    monkeypatch.setattr(backend, "_ENGINES", {"angry/model": Angry()})

    assert backend.unload_models() == ["angry/model"]  # no exception


def test_memory_pressure_parses_vm_stat(monkeypatch, backend):
    output = (
        "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
        "Pages free:                                 1000.\n"
        "Pages active:                               4000.\n"
        "Pages inactive:                             3000.\n"
        "Pages speculative:                           500.\n"
        "Pages throttled:                               0.\n"
        "Pages wired down:                            500.\n"
    )

    def fake_run(cmd, **kw):
        assert cmd == ["vm_stat"]
        return SimpleNamespace(stdout=output, returncode=0)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)

    # free + speculative = 1500; total = 9000; pressure = 1 - 1500/9000 = 0.8333
    assert backend.memory_pressure() == pytest.approx(0.8333, abs=1e-4)


def test_memory_pressure_none_when_vm_stat_fails(monkeypatch, backend):
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no vm_stat")),
    )
    assert backend.memory_pressure() is None


def test_memory_pressure_threshold_constant(backend):
    assert 0.0 < backend.MEMORY_PRESSURE_THRESHOLD < 1.0
