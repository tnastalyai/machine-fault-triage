"""Result models returned by a triage run."""

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


class TriageResult(_Model):
    """Structured hint for the technician produced by one triage run.

    ``usage`` holds the tokens of the model call, when the model reports
    them, also for a call that was refused or cut off. ``embedding_tokens``
    holds the tokens that page selection spent on embeddings for this run.
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
