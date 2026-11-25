"""Result models returned by a triage run."""

from collections.abc import Iterable
from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TriageStatus(StrEnum):
    """Outcome of a triage run."""

    OK = "ok"
    NOT_ENOUGH_DATA = "not_enough_data"
    REFUSED = "refused"
    FAILED = "failed"


class PageSource(StrEnum):
    """Where the text of a document page came from."""

    TEXT_LAYER = "text_layer"
    OCR = "ocr"


class EvidenceLevel(StrEnum):
    """Strongest kind of evidence the result rests on."""

    MANUAL = "manual"
    HISTORY = "history"
    GENERAL_KNOWLEDGE = "general_knowledge"
    NONE = "none"


class Confidence(StrEnum):
    """Qualitative confidence of a reading or a proposed cause."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HistoryScope(StrEnum):
    """Whether a service record belongs to this device or to its model."""

    DEVICE = "device"
    MODEL = "model"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DocumentCitation(_Model):
    """A quote from a document page that supports a cause."""

    type: Literal["document"] = "document"
    document_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    page: int = Field(ge=1)
    section: str | None = None
    quote: str = Field(min_length=1)
    verified: bool = False
    source: PageSource = PageSource.TEXT_LAYER
    """``ocr`` when the cited page was recognised from an image and may hold misreads."""


class HistoryCitation(_Model):
    """A past service record that supports a cause."""

    type: Literal["history"] = "history"
    record_id: str = Field(min_length=1)
    recorded_on: date
    scope: HistoryScope
    verified: bool = False


Citation = Annotated[DocumentCitation | HistoryCitation, Field(discriminator="type")]


class ErrorCode(_Model):
    """An error code as reported and as normalized for lookup."""

    raw: str = Field(min_length=1)
    normalized: str = Field(min_length=1)
    meaning: str | None = None
    read_from: str | None = None
    read_confidence: Confidence = Confidence.MEDIUM


class Cause(_Model):
    """A probable cause with the checks and fix a technician should try."""

    title: str = Field(min_length=1)
    confidence: Confidence
    why: str = Field(min_length=1)
    checks: tuple[str, ...] = ()
    fix: str | None = None
    citations: tuple[Citation, ...] = ()

    @property
    def is_grounded(self) -> bool:
        """Whether at least one citation was verified against the evidence."""
        return any(citation.verified for citation in self.citations)

    @model_validator(mode="after")
    def _ungrounded_cause_has_low_confidence(self) -> Self:
        if not self.is_grounded and self.confidence is not Confidence.LOW:
            message = "a cause without a verified citation must have low confidence"
            raise ValueError(message)
        return self


class Usage(_Model):
    """Tokens billed for one model call, as reported by the provider.

    ``input_tokens`` counts every input token. ``cached_input_tokens`` says
    how many of them the provider read from its prompt cache, which it bills
    at a lower price.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)


def total_usage(usages: Iterable[Usage | None]) -> Usage | None:
    """Add up the usage of several model calls, or return ``None`` if none reported any."""
    reported = [usage for usage in usages if usage is not None]
    if not reported:
        return None
    return Usage(
        input_tokens=sum(usage.input_tokens for usage in reported),
        output_tokens=sum(usage.output_tokens for usage in reported),
        cached_input_tokens=sum(usage.cached_input_tokens for usage in reported),
    )


class TriageResult(_Model):
    """Structured hint for the technician produced by one triage run.

    ``usage`` holds the tokens of the model call, when the model reports
    them, also for a call that was refused or cut off. ``embedding_tokens``
    holds the tokens that page selection spent on embeddings for this run.
    ``failure`` says why the model answer was rejected, when the run failed
    because the answer did not match the schema. Treat it as untrusted text. ``attempts`` counts the
    model calls of the run; it is above one when an answer was asked for again.
    ``photo_readings`` holds what the photo reader saw on each photo of the
    report, in order, so that a person can compare it with the photo; it is
    untrusted text too. ``photo_usage`` holds the tokens that reading spent.
    """

    status: TriageStatus
    summary: str
    evidence_level: EvidenceLevel
    language: str = Field(min_length=2)
    error_code: ErrorCode | None = None
    causes: tuple[Cause, ...] = ()
    parts_to_bring: tuple[str, ...] = ()
    safety_warnings: tuple[str, ...] = ()
    questions_for_customer: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unsupported_figures: tuple[str, ...] = ()
    prompt_version: int | None = None
    usage: Usage | None = None
    embedding_tokens: int = Field(default=0, ge=0)
    failure: str | None = None
    attempts: int = Field(default=1, ge=1)
    photo_readings: tuple[str, ...] = ()
    photo_usage: Usage | None = None

    @model_validator(mode="after")
    def _status_matches_content(self) -> Self:
        if self.status is TriageStatus.OK and not self.causes:
            message = "an ok result must contain at least one cause"
            raise ValueError(message)
        if self.status is not TriageStatus.OK and self.causes:
            message = "only an ok result may contain causes"
            raise ValueError(message)
        if self.evidence_level is EvidenceLevel.NONE and self.causes:
            message = "a result with causes must state its evidence level"
            raise ValueError(message)
        return self
