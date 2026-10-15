"""OCR for image-based (scanned) PDFs via tesseract + pdftoppm.

The real binaries are installed here, but tests fake subprocess so nothing is
actually rendered or recognized — the commands and the joining are what matter.
"""

import subprocess
import sys
import types
from types import SimpleNamespace


def test_tesseract_available_when_binary_present(backend, monkeypatch):
    monkeypatch.setattr(
        backend.shutil, "which", lambda cmd: "/opt/homebrew/bin/tesseract"
    )
    assert backend.tesseract_available() is True


def test_tesseract_available_false_when_missing(backend, monkeypatch):
    monkeypatch.setattr(backend.shutil, "which", lambda cmd: None)
    assert backend.tesseract_available() is False


def test_tesseract_langs_prefers_vie_when_installed(backend, monkeypatch):
    def fake_run(cmd, **kw):
        return SimpleNamespace(stdout="osd\nvie\neng\n", returncode=0)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    assert backend._tesseract_langs() == "vie+eng"


def test_tesseract_langs_falls_back_to_eng(backend, monkeypatch):
    def fake_run(cmd, **kw):
        return SimpleNamespace(stdout="osd\neng\n", returncode=0)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    assert backend._tesseract_langs() == "eng"


def test_ocr_image_runs_tesseract_and_returns_text(backend, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd == ["tesseract", "--list-langs"]:
            return subprocess.CompletedProcess(cmd, 0, "osd\nvie\neng\n", "")
        if cmd[0] == "tesseract":
            return subprocess.CompletedProcess(cmd, 0, "xin chào\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    monkeypatch.setattr(backend, "tesseract_available", lambda: True)

    assert backend.ocr_image("page-1.png") == "xin chào\n"
    assert calls[-1] == ["tesseract", "page-1.png", "stdout", "-l", "vie+eng"]


def test_ocr_image_uses_requested_lang(backend, monkeypatch):
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        if cmd[0] == "tesseract":
            return subprocess.CompletedProcess(cmd, 0, "ok\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    monkeypatch.setattr(backend, "tesseract_available", lambda: True)

    backend.ocr_image("page-1.png", lang="eng")
    assert ["tesseract", "page-1.png", "stdout", "-l", "eng"] in seen


def test_ocr_pdf_joins_pages_with_markers(backend, monkeypatch):
    monkeypatch.setattr(
        backend,
        "render_pdf_pages",
        lambda pdf_path, out_dir, dpi=200: ["page-1.png", "page-2.png"],
    )
    seen = []

    def fake_ocr(image_path, lang=None):
        seen.append((image_path, lang))
        return f"text of {image_path}\n"

    monkeypatch.setattr(backend, "ocr_image", fake_ocr)
    monkeypatch.setattr(backend, "tesseract_available", lambda: True)

    out = backend.ocr_pdf("dummy.pdf", lang="eng")

    assert seen == [("page-1.png", "eng"), ("page-2.png", "eng")]
    assert "--- page page-1 ---" in out
    assert "--- page page-2 ---" in out
    assert "text of page-1.png" in out


def test_read_pdf_ocr_fallback(monkeypatch, backend):
    # pypdf yields nothing → falls back to OCR.
    fake_page = types.SimpleNamespace(extract_text=lambda: "")
    fake_reader = types.SimpleNamespace(pages=[fake_page, fake_page])
    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = lambda path: fake_reader
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    monkeypatch.setitem(backend.CONFIG, "ocr_enabled", True)
    monkeypatch.setattr(backend, "tesseract_available", lambda: True)
    ocr_called = []
    monkeypatch.setattr(
        backend, "ocr_pdf", lambda path: ocr_called.append(path) or "OCR TEXT"
    )

    assert backend._read_pdf("x.pdf") == "OCR TEXT"
    assert ocr_called == ["x.pdf"]


def test_read_pdf_skips_ocr_when_disabled(monkeypatch, backend):
    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = lambda path: types.SimpleNamespace(
        pages=[types.SimpleNamespace(extract_text=lambda: "")]
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    monkeypatch.setitem(backend.CONFIG, "ocr_enabled", False)
    monkeypatch.setattr(
        backend,
        "ocr_pdf",
        lambda path: (_ for _ in ()).throw(AssertionError("OCR must not run")),
    )

    assert backend._read_pdf("x.pdf") == ""


def test_read_pdf_returns_text_layer_when_present(monkeypatch, backend):
    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = lambda path: types.SimpleNamespace(
        pages=[types.SimpleNamespace(extract_text=lambda: "real text")]
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    monkeypatch.setitem(backend.CONFIG, "ocr_enabled", True)
    called = []
    monkeypatch.setattr(backend, "ocr_pdf", lambda path: called.append(path))

    assert backend._read_pdf("x.pdf") == "real text"
    assert called == []
