import pytest
from pydantic import ValidationError

from machine_fault_triage import Document, FaultReport, Page, PageSource, Photo


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


def test_report_with_photos_survives_json_and_keeps_their_bytes_out_of_its_repr() -> None:
    photo = Photo(media_type="image/png", data=b"\x89PNG\xff\x00 secret pixels")
    report = FaultReport(description="See the photo.", language="en", photos=(photo, photo))

    assert FaultReport.model_validate_json(report.model_dump_json()) == report
    assert "secret" not in repr(report)


def test_photo_must_be_an_image_of_a_known_type_and_not_empty() -> None:
    with pytest.raises(ValidationError):
        Photo.model_validate({"media_type": "application/pdf", "data": b"%PDF"})
    with pytest.raises(ValidationError):
        Photo(media_type="image/jpeg", data=b"")


def test_page_stored_before_sources_existed_loads_as_text_layer() -> None:
    stored = Page.model_validate_json('{"number": 3, "text": "E7: flow sensor fault."}')
    recognised = Page(number=3, text="E7: flow sensor fault.", source=PageSource.OCR)

    assert stored.source is PageSource.TEXT_LAYER
    assert Page.model_validate_json(recognised.model_dump_json()) == recognised
