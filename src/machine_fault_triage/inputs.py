"""Evidence handed to a triage run: the fault report, documents and service records."""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from machine_fault_triage.models import HistoryScope


class _Input(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class FaultReport(_Input):
    """What the customer reported, in their own words."""

    description: str = Field(min_length=1)
    language: str = Field(min_length=2)


class Page(_Input):
    """Extracted text of one document page."""

    number: int = Field(ge=1)
    text: str


class Document(_Input):
    """A manual or data sheet that belongs to the device or its model."""

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    pages: tuple[Page, ...]

    def page(self, number: int) -> Page | None:
        """Return the page with the given number, if the document has one."""
        return next((page for page in self.pages if page.number == number), None)


class ServiceRecord(_Input):
    """A past repair of this device or of another device of the same model."""

    id: str = Field(min_length=1)
    recorded_on: date
    scope: HistoryScope
    summary: str = Field(min_length=1)
    parts_replaced: tuple[str, ...] = ()
