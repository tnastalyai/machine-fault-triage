"""Evidence handed to a triage run: the fault report, documents and service records."""

from collections.abc import Sequence
from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from machine_fault_triage.models import HistoryScope, PageSource


class _Input(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Photo(_Input):
    """A picture attached to a fault report, such as the display of the device.

    The bytes are passed on as given. Remove location and other metadata
    before building a report if the photo may leave your systems.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", ser_json_bytes="base64", val_json_bytes="base64"
    )

    media_type: Literal["image/jpeg", "image/png", "image/webp"]
    data: bytes = Field(min_length=1, repr=False)


class FaultReport(_Input):
    """What the customer reported, in their own words, with any photos they attached."""

    description: str = Field(min_length=1)
    language: str = Field(min_length=2)
    photos: tuple[Photo, ...] = ()

    @field_validator("description")
    @classmethod
    def _description_says_something(cls, description: str) -> str:
        if not description.strip():
            message = "a fault report must not be blank"
            raise ValueError(message)
        return description


class Page(_Input):
    """Extracted text of one document page.

    ``source`` says whether the text was stored in the file or recognised
    from an image of the page. Recognised text can hold misread characters
    that nothing in the library can detect.
    """

    number: int = Field(ge=1)
    text: str
    source: PageSource = PageSource.TEXT_LAYER


class Document(_Input):
    """A manual or data sheet that belongs to the device or its model."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    pages: tuple[Page, ...]
    supersedes: tuple[str, ...] = ()
    """Ids of older editions that this document replaces."""

    @model_validator(mode="after")
    def _page_numbers_are_unique(self) -> Self:
        numbers = [page.number for page in self.pages]
        if len(numbers) != len(set(numbers)):
            message = "a document cannot hold two pages with the same number"
            raise ValueError(message)
        return self

    def page(self, number: int) -> Page | None:
        """Return the page with the given number, if the document has one."""
        return next((page for page in self.pages if page.number == number), None)


def without_superseded(documents: Sequence[Document]) -> Sequence[Document]:
    """Return the documents that no other supplied document replaces.

    An old edition next to its replacement would let the model quote a value
    that is no longer valid, and the quote would still verify.
    """
    replaced = {old for document in documents for old in document.supersedes}
    if not replaced.intersection(document.id for document in documents):
        return documents
    return [document for document in documents if document.id not in replaced]


class ServiceRecord(_Input):
    """A past repair of this device or of another device of the same model."""

    id: str = Field(min_length=1)
    recorded_on: date
    scope: HistoryScope
    summary: str = Field(min_length=1)
    parts_replaced: tuple[str, ...] = ()
