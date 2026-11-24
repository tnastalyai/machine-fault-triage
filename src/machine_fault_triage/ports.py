"""Interfaces a host application implements to plug in its model and data."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from machine_fault_triage.inputs import Document, ServiceRecord


class UntrustedBlock(BaseModel):
    """Text that came from outside and must be treated as data, never as instructions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    content: str


class GenerationRequest(BaseModel):
    """One model call, with trusted instructions kept apart from untrusted text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instructions: str = Field(min_length=1)
    untrusted: tuple[UntrustedBlock, ...] = ()


class GenerationStatus(StrEnum):
    """How a model call ended."""

    COMPLETED = "completed"
    REFUSED = "refused"
    TRUNCATED = "truncated"
    INVALID = "invalid"


class Generation[OutputT: BaseModel](BaseModel):
    """Outcome of a model call: the parsed output, or the reason there is none."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: GenerationStatus
    output: OutputT | None = None

    @model_validator(mode="after")
    def _output_present_only_when_completed(self) -> "Generation[OutputT]":
        if (self.status is GenerationStatus.COMPLETED) != (self.output is not None):
            message = "a generation carries output exactly when it completed"
            raise ValueError(message)
        return self


class StructuredModel(Protocol):
    """A language model that answers in a requested schema."""

    def generate[OutputT: BaseModel](
        self, request: GenerationRequest, output_type: type[OutputT]
    ) -> Generation[OutputT]:
        """Answer the request as an instance of ``output_type``.

        Refusals, truncated answers and answers that do not match the schema are
        reported through the returned status, not raised.
        """
        ...


class DocumentProvider(Protocol):
    """Source of the documents that describe a device."""

    def documents_for(self, *, tenant_id: str, device_id: str) -> Sequence[Document]:
        """Return the documents of the device and of its model, within one tenant."""
        ...


class HistoryProvider(Protocol):
    """Source of past service records."""

    def records_for(self, *, tenant_id: str, device_id: str) -> Sequence[ServiceRecord]:
        """Return past repairs of the device and of its model, within one tenant."""
        ...
