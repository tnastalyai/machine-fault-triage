from pathlib import Path

import pytest

from machine_fault_triage.errors import DocumentLoadError
from machine_fault_triage.grounding import contains_quote
from machine_fault_triage.models import PageSource
from machine_fault_triage.pdf import load_pdf
from machine_fault_triage.testing import FakeOcrEngine
from tests.pdf_files import write_pdf


def test_each_page_with_text_becomes_a_page_numbered_by_its_position(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "manual.pdf", [["Cover"], [], ["E7: flow sensor fault."]])

    document = load_pdf(path, document_id="manual-1", title="Service manual")

    assert (document.id, document.title) == ("manual-1", "Service manual")
    assert [(page.number, page.text) for page in document.pages] == [
        (1, "Cover"),
        (3, "E7: flow sensor fault."),
    ]


def test_word_broken_across_lines_can_be_quoted_whole(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "manual.pdf", [["Clean the strain-", "er and vent the circuit."]])

    (page,) = load_pdf(path, document_id="manual-1", title="Service manual").pages

    assert contains_quote(page.text, "Clean the strainer and vent the circuit.")


def test_hyphen_inside_a_part_number_is_kept(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "manual.pdf", [["Order part 4410-", "2291 for this unit."]])

    (page,) = load_pdf(path, document_id="manual-1", title="Service manual").pages

    assert page.text == "Order part 4410-\n2291 for this unit."


def test_document_without_a_text_layer_is_rejected(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "scan.pdf", [[], []])

    with pytest.raises(DocumentLoadError, match="no text layer"):
        load_pdf(path, document_id="manual-1", title="Service manual")


def test_file_that_is_not_a_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.pdf"
    path.write_text("not a pdf")

    with pytest.raises(DocumentLoadError, match="cannot be read as a PDF"):
        load_pdf(path, document_id="manual-1", title="Service manual")


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="cannot be read as a PDF"):
        load_pdf(tmp_path / "absent.pdf", document_id="manual-1", title="Service manual")


def test_page_without_text_is_recognised_and_marked_when_an_engine_is_given(
    tmp_path: Path,
) -> None:
    path = write_pdf(tmp_path / "manual.pdf", [["Cover"], []])
    engine = FakeOcrEngine(["E7: flow sensor fault."])

    document = load_pdf(path, document_id="manual-1", title="Service manual", ocr=engine)

    assert [(page.number, page.text, page.source) for page in document.pages] == [
        (1, "Cover", PageSource.TEXT_LAYER),
        (2, "E7: flow sensor fault.", PageSource.OCR),
    ]
    assert len(engine.images) == 1


def test_engine_receives_the_page_as_rows_of_rgb_pixels(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "scan.pdf", [[]])
    engine = FakeOcrEngine(["E7"])

    load_pdf(path, document_id="manual-1", title="Service manual", ocr=engine)

    (image,) = engine.images
    assert (image.width, image.height) == (1700, 2200)
    assert len(image.rgb) == 1700 * 2200 * 3
    assert set(image.rgb) == {255}


def test_recognised_word_broken_across_lines_can_be_quoted_whole(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "scan.pdf", [[]])
    engine = FakeOcrEngine(["Clean the strain-\ner and vent.\nOrder part 4410-\n2291 now."])

    (page,) = load_pdf(path, document_id="manual-1", title="Service manual", ocr=engine).pages

    assert page.text == "Clean the strainer and vent.\nOrder part 4410-\n2291 now."


def test_scan_on_which_nothing_is_recognised_is_rejected(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "scan.pdf", [[], []])

    with pytest.raises(DocumentLoadError, match="no text was recognised"):
        load_pdf(
            path, document_id="manual-1", title="Service manual", ocr=FakeOcrEngine(["", " \n"])
        )


def test_small_pictures_on_a_page_are_counted(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "manual.pdf", [["Icons on the main screen"], ["E7"]], icons=[6])

    document = load_pdf(path, document_id="manual-1", title="Service manual")

    assert [page.small_images for page in document.pages] == [6, 0]
