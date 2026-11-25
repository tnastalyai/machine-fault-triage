"""Shapes a model is asked to produce, before any claim has been verified."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from machine_fault_triage.models import Confidence

_Name = Annotated[str, StringConstraints(max_length=300)]
_Line = Annotated[str, StringConstraints(max_length=1000)]
_Text = Annotated[str, StringConstraints(min_length=1, max_length=3000)]
_Lines = Annotated[tuple[_Line, ...], Field(max_length=20)]


class _Draft(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DraftDocumentCitation(_Draft):
    """A quote the model claims to have read on a document page."""

    type: Literal["document"] = "document"
    document_id: _Name
    page: int
    quote: Annotated[str, StringConstraints(max_length=3000)]


class DraftHistoryCitation(_Draft):
    """A service record the model claims supports the cause."""

    type: Literal["history"] = "history"
    record_id: _Name


DraftCitation = Annotated[DraftDocumentCitation | DraftHistoryCitation, Field(discriminator="type")]


class DraftCause(_Draft):
    """A cause as proposed by the model."""

    title: _Name = Field(min_length=1)
    confidence: Confidence
    why: _Text
    checks: _Lines = ()
    fix: Annotated[str, StringConstraints(max_length=3000)] | None = None
    citations: tuple[DraftCitation, ...] = Field(default=(), max_length=20)


class DraftTriage(_Draft):
    """The complete model answer for one fault report.

    Every text and list has an upper bound, several times above what answers
    need, so that a runaway answer is rejected instead of passed on.
    """

    summary: _Text
    enough_evidence: bool
    causes: tuple[DraftCause, ...] = Field(default=(), max_length=10)
    parts_to_bring: _Lines = ()
    safety_warnings: _Lines = ()
    questions_for_customer: _Lines = ()
    missing: _Lines = ()
