"""Microphone archive: raw input saved for later re-transcription."""

import os
import wave


def test_audio_archive_dir_defaults_to_config_audio(monkeypatch, backend, tmp_path):
    monkeypatch.setattr(backend, "CONFIG_ROOT", tmp_path)
    monkeypatch.setitem(backend.CONFIG, "audio_archive_path", "")

    directory = backend.audio_archive_dir()

    assert directory == tmp_path / "audio"
    assert directory.is_dir()


def test_audio_archive_dir_uses_configured_path(monkeypatch, backend, tmp_path):
    monkeypatch.setitem(backend.CONFIG, "audio_archive_path", str(tmp_path / "custom"))

    directory = backend.audio_archive_dir()

    assert directory == tmp_path / "custom"
    assert directory.is_dir()


def test_archive_files_newest_first(tmp_path, backend):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    os.utime(a, (100, 100))
    os.utime(b, (200, 200))

    files = backend.archive_files(tmp_path)

    assert [p.name for p in files] == ["b.wav", "a.wav"]


def test_archive_files_ignores_non_wav(tmp_path, backend):
    (tmp_path / "notes.txt").write_bytes(b"x")
    assert backend.archive_files(tmp_path) == []


def test_archive_storage_bytes_sums_wavs_only(tmp_path, backend):
    (tmp_path / "a.wav").write_bytes(b"0" * 100)
    (tmp_path / "b.wav").write_bytes(b"1" * 200)
    (tmp_path / "ignore.txt").write_bytes(b"9" * 999)

    assert backend.archive_storage_bytes(tmp_path) == 300


def test_clear_audio_archive_removes_wavs_only(tmp_path, backend):
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "b.wav").write_bytes(b"y")
    (tmp_path / "keep.txt").write_bytes(b"z")

    removed = backend.clear_audio_archive(tmp_path)

    assert removed == 2
    assert not (tmp_path / "a.wav").exists()
    assert not (tmp_path / "b.wav").exists()
    assert (tmp_path / "keep.txt").exists()


def test_speech_engine_on_audio_archives_when_enabled(tmp_path, backend):
    import numpy as np

    from dich import speech_engine

    engine = speech_engine.SpeechEngine()
    engine._archive_enabled = True
    engine._on_audio(np.array([[0.25], [-0.25]], dtype=np.float32), 2, None, None)

    assert len(engine._archive_blocks) == 1


def test_speech_engine_on_audio_does_not_archive_when_disabled(tmp_path, backend):
    import numpy as np

    from dich import speech_engine

    engine = speech_engine.SpeechEngine()
    engine._on_audio(np.array([[0.25]], dtype=np.float32), 1, None, None)

    assert engine._archive_blocks == []


def test_speech_engine_finalize_writes_wav(tmp_path, backend):
    import numpy as np

    from dich import speech_engine

    engine = speech_engine.SpeechEngine()
    engine._archive_dir = tmp_path
    engine._archive_blocks = [np.array([0.5, -0.5], dtype=np.float32)]
    engine._archive_enabled = True

    path = engine._finalize_archive()

    assert path is not None
    assert path.exists()
    assert path.suffix == ".wav"
    assert engine.last_archive_path == path
    assert engine._archive_enabled is False  # consumed

    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == speech_engine.SAMPLE_RATE


def test_speech_engine_finalize_noop_when_not_enabled(tmp_path, backend):
    from dich import speech_engine

    engine = speech_engine.SpeechEngine()
    engine._archive_dir = tmp_path
    assert engine._finalize_archive() is None
