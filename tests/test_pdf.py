from pathlib import Path

import pytest

from openbb_ada.pdf import Pdf


def test_pdf_reader():
    test_pdf_path = Path(__file__).parent / "test_data" / "openbb_story.pdf"
    reader = Pdf(test_pdf_path.read_bytes())
    pages = reader.get_text()

    assert len(pages) == 5
    assert "$DOGE in Mars" in pages[2]


def test_pdf_reader_empty_content_raises_runtime_error():
    with pytest.raises(RuntimeError):
        Pdf(b"")
