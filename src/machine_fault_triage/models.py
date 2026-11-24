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


class DocumentSource(_Model):
    """A quote from a document page that supports a cause."""

    type: Literal["document"] = "document"
    document_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)
    verified: bool = False


class HistorySource(_Model):
    """A past service record that supports a cause."""

    type: Literal["history"] = "history"
    record_id: str = Field(min_length=1)
    recorded_on: date
    scope: HistoryScope


Source = Annotated[DocumentSource | HistorySource, Field(discriminator="type")]


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
    sources: tuple[Source, ...] = ()

    @property
    def is_grounded(self) -> bool:
        """Whether a verified document quote or a service record backs the cause."""
        return any(isinstance(source, HistorySource) or source.verified for source in self.sources)

    @model_validator(mode="after")
    def _ungrounded_cause_has_low_confidence(self) -> Self:
        if not self.is_grounded and self.confidence is not Confidence.LOW:
            message = "a cause without a verified source must have low confidence"
            raise ValueError(message)
        return self


class TriageResult(_Model):
    """Structured hint for the technician produced by one triage run."""

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
