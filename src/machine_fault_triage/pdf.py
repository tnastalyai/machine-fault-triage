"""Turn a PDF manual into a ``Document``. Needs the ``pdf`` extra."""

import re
from pathlib import Path

import pypdfium2
import pypdfium2.raw

from machine_fault_triage.errors import DocumentLoadError
from machine_fault_triage.inputs import Document, Page
from machine_fault_triage.models import PageSource
from machine_fault_triage.ports import OcrEngine, PageImage

_BROKEN_WORD = "\ufffe"
_HYPHENATED = re.compile(r"(?<=[^\W\d_])-\n(?=[^\W\d_])")
_OCR_DPI = 200
_SMALL_IMAGE = 0.12

EXTRACTION_VERSION = 2
"""Raised whenever ``load_pdf`` starts to return different text for the same file."""


def load_pdf(path: Path, *, document_id: str, title: str, ocr: OcrEngine | None = None) -> Document:
    """Read the text of a PDF, one ``Page`` per PDF page that has text.

    Page numbers are positions in the file, starting at 1. They can differ
    from the numbers printed on the pages, for example after a cover sheet.
    Words that the layout broke across two lines are joined again, so that a
    quote of the whole word can be found on the page.

    A page without a text layer is left out, unless ``ocr`` is given: then
    the page is rendered and its text recognised, and the page is marked with
    ``PageSource.OCR``. Pages that have a text layer are never recognised.
    Recognition is slow, about a second a page, and can misread characters.

    ``Page.small_images`` counts the embedded pictures narrower and lower
    than an eighth of the page, the size of an icon or a button. Pictures
    drawn as vector shapes are not counted.

    Raises:
        DocumentLoadError: If the file cannot be read as a PDF, or if no page
            yields text, as in a scanned manual read without ``ocr``.
    """
    try:
        pdf = pypdfium2.PdfDocument(path)
    except (OSError, pypdfium2.PdfiumError) as error:
        message = f"{path} cannot be read as a PDF: {error}"
        raise DocumentLoadError(message) from error
    try:
        pages = tuple(
            page for index in range(len(pdf)) if (page := _page(pdf, index, ocr)) is not None
        )
    finally:
        pdf.close()
    if not pages:
        message = (
            f"{path} has no text layer; scanned documents need character recognition first"
            if ocr is None
            else f"{path} has no text layer, and no text was recognised on its pages"
        )
        raise DocumentLoadError(message)
    return Document(id=document_id, title=title, pages=pages)


def _page(pdf: pypdfium2.PdfDocument, index: int, ocr: OcrEngine | None) -> Page | None:
    page = pdf[index]
    try:
        text = _text_layer(page)
        source = PageSource.TEXT_LAYER
        small_images = _small_images(page)
        if not text and ocr is not None:
            text = _HYPHENATED.sub("", ocr.recognize(_image(page))).strip()
            source = PageSource.OCR
    finally:
        page.close()
    if not text:
        return None
    return Page(number=index + 1, text=text, source=source, small_images=small_images)


def _text_layer(page: pypdfium2.PdfPage) -> str:
    text_page = page.get_textpage()
    try:
        raw: str = text_page.get_text_range()
    finally:
        text_page.close()
    lines = raw.replace("\r\n", "\n").replace("\r", "\n")
    return lines.replace(f"{_BROKEN_WORD}\n", "").replace(_BROKEN_WORD, "").strip()


def _small_images(page: pypdfium2.PdfPage) -> int:
    width, height = page.get_size()
    count = 0
    for image in page.get_objects(filter=[pypdfium2.raw.FPDF_PAGEOBJ_IMAGE], max_depth=4):
        left, bottom, right, top = image.get_bounds()
        count += right - left < _SMALL_IMAGE * width and top - bottom < _SMALL_IMAGE * height
    return count


def _image(page: pypdfium2.PdfPage) -> PageImage:
    bitmap = page.render(scale=_OCR_DPI / 72, rev_byteorder=True)
    try:
        packed = bytes(bitmap.buffer)
        row = bitmap.width * 3
        rows = (
            packed[start : start + row]
            for start in range(0, bitmap.stride * bitmap.height, bitmap.stride)
        )
        return PageImage(bitmap.width, bitmap.height, b"".join(rows))
    finally:
        bitmap.close()
