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


def test_read_pdf_uses_pdftotext_when_pypdf_empty(monkeypatch, backend):
    """A layout-preserving extractor is used when pypdf yields nothing."""
    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = lambda path: types.SimpleNamespace(
        pages=[types.SimpleNamespace(extract_text=lambda: "")]
    )
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    monkeypatch.setattr(
        backend.shutil,
        "which",
        lambda cmd: "/usr/bin/pdftotext" if cmd == "pdftotext" else None,
    )
    monkeypatch.setattr(
        backend.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(
            returncode=0, stdout="layout text\n", stderr=""
        ),
    )

    assert backend._read_pdf("x.pdf") == "layout text\n"


def test_pdftotext_returns_empty_when_poppler_missing(monkeypatch, backend):
    monkeypatch.setattr(backend.shutil, "which", lambda cmd: None)

    assert backend._pdftotext_text("x.pdf") == ""


def test_image_suffix_from_magic_bytes(backend):
    assert backend._image_suffix(b"\xff\xd8\xff\x10") == ".jpg"
    assert backend._image_suffix(b"\x89PNG\r\n\x1a\n\x00") == ".png"
    assert backend._image_suffix(b"GIF89a\x01") == ".gif"
    assert backend._image_suffix(b"BM\x02\x00") == ".bmp"
    assert backend._image_suffix(b"II*\x00") == ".tif"
    assert backend._image_suffix(b"MM*\x00") == ".tif"
    assert backend._image_suffix(b"random bytes") == ".png"


def test_ocr_embedded_images_returns_empty_without_tesseract(monkeypatch, backend):
    monkeypatch.setattr(backend, "tesseract_available", lambda: False)
    assert backend.ocr_embedded_images("x.pdf") == ""


def test_ocr_embedded_images_skips_images_without_data(monkeypatch, backend):
    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    monkeypatch.setattr(backend, "tesseract_available", lambda: True)

    fake_img = types.SimpleNamespace(data=b"")
    fake_page = types.SimpleNamespace(images=[fake_img])
    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = lambda path: types.SimpleNamespace(pages=[fake_page])
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    assert backend.ocr_embedded_images("x.pdf") == ""


def test_ocr_embedded_images_ocr_each_image(monkeypatch, backend):

    def fake_run(cmd, **kw):
        if cmd[0] == "tesseract":
            return subprocess.CompletedProcess(cmd, 0, f"text of {cmd[1]}\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    monkeypatch.setattr(backend, "tesseract_available", lambda: True)

    fake_img = types.SimpleNamespace(data=b"\x89PNG\r\n\x1a\nfake")
    fake_page = types.SimpleNamespace(images=[fake_img])
    fake_pypdf = types.ModuleType("pypdf")
    fake_pypdf.PdfReader = lambda path: types.SimpleNamespace(pages=[fake_page])
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    out = backend.ocr_embedded_images("x.pdf")

    assert "--- image 1 on page 1 ---" in out
    assert "text of" in out


def test_read_pdf_appends_embedded_image_ocr(monkeypatch, backend, tmp_path):
    # pypdf yields real text; embedded image OCR is appended.
    fake_pypdf = types.ModuleType("pypdf")
    fake_page = types.SimpleNamespace(
        extract_text=lambda: "real text",
        images=[types.SimpleNamespace(data=b"\x89PNG\r\n\x1a\nfake")],
    )
    fake_pypdf.PdfReader = lambda path: types.SimpleNamespace(pages=[fake_page])
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    monkeypatch.setitem(backend.CONFIG, "ocr_enabled", True)
    monkeypatch.setattr(backend, "tesseract_available", lambda: True)

    ocr_calls = []

    def fake_ocr_image(image_path, lang=None):
        ocr_calls.append(image_path)
        return "DIAGRAM TEXT\n"

    monkeypatch.setattr(backend, "ocr_image", fake_ocr_image)

    out = backend._read_pdf("x.pdf")

    assert "real text" in out
    assert "--- image 1 on page 1 ---" in out
    assert "DIAGRAM TEXT" in out
    assert len(ocr_calls) == 1
