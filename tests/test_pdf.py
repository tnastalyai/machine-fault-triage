from pathlib import Path

import pytest

from machine_fault_triage.errors import DocumentLoadError
from machine_fault_triage.grounding import contains_quote
from machine_fault_triage.pdf import load_pdf
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
