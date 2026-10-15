"""
Same-format export and folder batch translation.

Regression cover: "Save Translation" always wrote `.txt`, so a translated
`.docx` came back as a text file, and there was no way to translate a folder.
"""

from types import SimpleNamespace

import pytest

from dich import backend


def test_export_suffix_keeps_rebuildable_formats():
    assert backend.export_suffix_for("a/b/notes.docx") == ".docx"
    assert backend.export_suffix_for("notes.md") == ".md"
    assert backend.export_suffix_for("notes.txt") == ".txt"


def test_export_suffix_falls_back_to_text_for_pdf():
    # A PDF's flat text has no faithful container to re-emit.
    assert backend.export_suffix_for("paper.pdf") == ".txt"


def test_export_writes_plain_text(tmp_path, backend):
    out = tmp_path / "x.md"
    backend.export_text("# Title\n\nbody", str(out))
    assert out.read_text(encoding="utf-8") == "# Title\n\nbody"


def test_export_rebuilds_a_docx(tmp_path, backend):
    docx = pytest.importorskip("docx")
    out = tmp_path / "x.docx"

    backend.export_text("# Heading\n\nFirst paragraph.\n\nSecond.", str(out))

    document = docx.Document(str(out))
    texts = [p.text for p in document.paragraphs if p.text]
    assert "Heading" in texts
    assert "First paragraph." in texts


def test_folder_files_lists_only_supported_files(tmp_path, backend):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "b.md").write_text("hi", encoding="utf-8")
    (tmp_path / "skip.bin").write_bytes(b"\x00")
    (tmp_path / "sub").mkdir()

    files = backend.folder_files(str(tmp_path))

    assert [f.split("/")[-1] for f in files] == ["a.txt", "b.md"]


def test_folder_files_is_empty_for_a_missing_folder(backend):
    assert backend.folder_files("/no/such/folder/here") == []


def test_translate_folder_writes_alongside_with_a_language_suffix(
    tmp_path, monkeypatch, backend
):
    (tmp_path / "one.txt").write_text("xin chào", encoding="utf-8")
    (tmp_path / "two.md").write_text("tạm biệt", encoding="utf-8")
    monkeypatch.setattr(backend, "TARGET_LANGUAGE", "English")

    def fake_translate_file(path, model=None, on_progress=None):
        return SimpleNamespace(text=f"EN:{path}", cached=False)

    monkeypatch.setattr(backend, "translate_file", fake_translate_file)
    seen = []

    out = backend.translate_folder(str(tmp_path), on_file=lambda p, r: seen.append(p))

    assert len(out["results"]) == 2
    assert all(r["ok"] for r in out["results"])
    assert (tmp_path / "one.en.txt").read_text(encoding="utf-8").startswith("EN:")
    assert (tmp_path / "two.en.md").exists()
    assert len(seen) == 2  # progress callback fired per file


def test_translate_folder_keeps_going_after_a_failure(tmp_path, monkeypatch, backend):
    (tmp_path / "bad.txt").write_text("x", encoding="utf-8")
    (tmp_path / "good.txt").write_text("y", encoding="utf-8")

    def flaky(path, model=None, on_progress=None):
        if path.endswith("bad.txt"):
            raise RuntimeError("decode failed")
        return SimpleNamespace(text="ok")

    monkeypatch.setattr(backend, "translate_file", flaky)

    out = backend.translate_folder(str(tmp_path))

    by_file = {r["file"].split("/")[-1]: r for r in out["results"]}
    assert by_file["bad.txt"]["ok"] is False
    assert by_file["good.txt"]["ok"] is True
    assert (tmp_path / "good.en.txt").exists()
