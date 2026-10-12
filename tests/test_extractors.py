"""Every document extractor returns the text the model will be shown.

The fixtures are generated here rather than taken from test_data/, which is
gitignored — CI must be able to run these without the developer's samples.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_pdf(backend):
    out = backend.extract_text_from_file(str(FIXTURES / "vi_sample.pdf"))
    assert "Tài liệu mẫu tiếng Việt." in _norm(out)
    assert "ư, ơ, ạ" in _norm(out)          # diacritics survive extraction


def test_docx_paragraphs_and_tables(backend, tmp_path):
    import docx
    path = tmp_path / "sample.docx"
    document = docx.Document()
    document.add_heading("Mục một", level=1)
    document.add_paragraph("Xin chào các bạn.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Ông Nguyễn"
    table.rows[0].cells[1].text = "Việt Nam"
    document.save(path)

    out = backend.extract_text_from_file(str(path))
    assert "Mục một" in out
    assert "Xin chào các bạn." in out
    assert "Ông Nguyễn\tViệt Nam" in out    # table rows are tab-joined


def test_docm_dispatches_to_the_docx_extractor(backend, tmp_path):
    # python-docx reads the zip regardless of the macro-enabled extension.
    path = tmp_path / "sample.docm"
    shutil.copy(FIXTURES / "vi_sample.docx", path)
    assert backend.extract_text_from_file(str(path))


def test_rtf(backend, tmp_path):
    path = tmp_path / "sample.rtf"
    path.write_text(r"{\rtf1\ansi Xin chao cac ban.}", encoding="utf-8")
    assert "Xin chao cac ban." in backend.extract_text_from_file(str(path))


def test_markdown_is_source_not_html(backend, tmp_path):
    # Regression: markdown used to be converted to HTML before translation,
    # so the output pane rendered literal <h1> tags.
    path = tmp_path / "sample.md"
    path.write_text("# Tiêu đề\n\nd **đậm**\n", encoding="utf-8")
    out = backend.extract_text_from_file(str(path))
    assert out == "# Tiêu đề\n\nd **đậm**\n"
    assert "<h1" not in out


def test_txt(backend, tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("nội dung", encoding="utf-8")
    assert backend.extract_text_from_file(str(path)) == "nội dung"


def test_unknown_extension_lists_what_is_supported(backend, tmp_path):
    path = tmp_path / "sample.xyz"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="txt"):
        backend.extract_text_from_file(str(path))


@pytest.mark.skipif(shutil.which("textutil") is None, reason="macOS textutil")
def test_legacy_doc_via_textutil(backend, tmp_path):
    # textutil cannot create .doc from nothing: convert the RTF fixture.
    path = tmp_path / "sample.doc"
    subprocess.run(
        ["textutil", "-convert", "doc", "-output", str(path),
         str(FIXTURES / "vi_sample.rtf")],
        check=True,
    )
    assert backend.extract_text_from_file(str(path))
