"""Turn a PDF manual into a ``Document``. Needs the ``pdf`` extra."""

from pathlib import Path

import pypdfium2

from machine_fault_triage.errors import DocumentLoadError
from machine_fault_triage.inputs import Document, Page

_BROKEN_WORD = "\ufffe"

EXTRACTION_VERSION = 1
"""Raised whenever ``load_pdf`` starts to return different text for the same file."""


def load_pdf(path: Path, *, document_id: str, title: str) -> Document:
    """Read the text layer of a PDF, one ``Page`` per PDF page that has text.

    Page numbers are positions in the file, starting at 1. They can differ
    from the numbers printed on the pages, for example after a cover sheet.
    Words that the layout broke across two lines are joined again, so that a
    quote of the whole word can be found on the page. Pages without text are
    left out. No character recognition is done.

    Raises:
        DocumentLoadError: If the file cannot be read as a PDF, or if no page
            has a text layer, as in a scanned manual.
    """
    try:
        pdf = pypdfium2.PdfDocument(path)
    except (OSError, pypdfium2.PdfiumError) as error:
        message = f"{path} cannot be read as a PDF: {error}"
        raise DocumentLoadError(message) from error
    try:
        texts = [_page_text(pdf, index) for index in range(len(pdf))]
    finally:
        pdf.close()
    pages = tuple(Page(number=number, text=text) for number, text in enumerate(texts, 1) if text)
    if not pages:
        message = f"{path} has no text layer; scanned documents need character recognition first"
        raise DocumentLoadError(message)
    return Document(id=document_id, title=title, pages=pages)


def _page_text(pdf: pypdfium2.PdfDocument, index: int) -> str:
    page = pdf[index]
    text_page = page.get_textpage()
    try:
        raw: str = text_page.get_text_range()
    finally:
        text_page.close()
        page.close()
    lines = raw.replace("\r\n", "\n").replace("\r", "\n")
    return lines.replace(f"{_BROKEN_WORD}\n", "").replace(_BROKEN_WORD, "").strip()
