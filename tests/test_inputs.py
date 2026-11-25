import pytest
from pydantic import ValidationError

from machine_fault_triage import Document, FaultReport, Page


@pytest.mark.parametrize("description", ["", "   ", "\n\t"])
def test_blank_fault_report_is_rejected(description: str) -> None:
    with pytest.raises(ValidationError):
        FaultReport(description=description, language="en")


def test_fault_report_keeps_the_customers_words_as_written() -> None:
    report = FaultReport(description="  E7 again.\n", language="en")

    assert report.description == "  E7 again.\n"


def test_document_with_two_pages_of_the_same_number_is_rejected() -> None:
    pages = (Page(number=7, text="First."), Page(number=7, text="Second."))

    with pytest.raises(ValidationError, match="two pages with the same number"):
        Document(id="manual-1", title="Manual", pages=pages)
