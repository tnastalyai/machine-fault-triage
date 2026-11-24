"""Shapes a model is asked to produce, before any claim has been verified."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from machine_fault_triage.models import Confidence


class _Draft(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DraftDocumentCitation(_Draft):
    """A quote the model claims to have read on a document page."""

    type: Literal["document"] = "document"
    document_id: str
    page: int
    quote: str


class DraftHistoryCitation(_Draft):
    """A service record the model claims supports the cause."""

    type: Literal["history"] = "history"
    record_id: str


DraftCitation = Annotated[DraftDocumentCitation | DraftHistoryCitation, Field(discriminator="type")]


class DraftCause(_Draft):
    """A cause as proposed by the model."""

    title: str = Field(min_length=1)
    confidence: Confidence
    why: str = Field(min_length=1)
    checks: tuple[str, ...] = ()
    fix: str | None = None
    citations: tuple[DraftCitation, ...] = ()


class DraftTriage(_Draft):
    """The complete model answer for one fault report."""

    summary: str
    enough_evidence: bool
    causes: tuple[DraftCause, ...] = ()
    parts_to_bring: tuple[str, ...] = ()
    safety_warnings: tuple[str, ...] = ()
    questions_for_customer: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
