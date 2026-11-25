"""Interfaces a host application implements to plug in its model and data."""

from collections.abc import Sequence
from enum import StrEnum
from typing import NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from machine_fault_triage.inputs import Document, FaultReport, Photo, ServiceRecord
from machine_fault_triage.models import Usage

_LONGEST_REJECTION = 1000
_FIXED_WORDING = {
    "extra_forbidden": "holds a field the schema does not have",
    "union_tag_invalid": "has a type the schema does not have",
}


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
    """Outcome of a model call: the parsed output, or the reason there is none.

    ``usage`` is optional so that a model that cannot report it still fits.
    ``detail`` says why an answer was rejected as invalid: the fields that
    broke the schema, without the values the model wrote. It is still derived
    from a model answer and must be handled as untrusted text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: GenerationStatus
    output: OutputT | None = None
    usage: Usage | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _output_present_only_when_completed(self) -> "Generation[OutputT]":
        if (self.status is GenerationStatus.COMPLETED) != (self.output is not None):
            message = "a generation carries output exactly when it completed"
            raise ValueError(message)
        return self


def rejection(error: ValidationError) -> str:
    """Describe why a model answer broke its schema, leaving out what the model wrote.

    The name of an unexpected field and an unknown kind of citation are
    chosen by the model, so they are replaced with fixed wording. The text is
    still derived from a model answer: pass it on as data, not as instructions.
    """
    described = []
    for problem in error.errors(include_input=False, include_url=False, include_context=False):
        kind = problem["type"]
        named = problem["loc"][:-1] if kind == "extra_forbidden" else problem["loc"]
        where = ".".join(str(part) for part in named) or "answer"
        described.append(f"{where}: {_FIXED_WORDING.get(kind, problem['msg'])}")
    return "; ".join(described)[:_LONGEST_REJECTION]


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


class PhotoReadings(NamedTuple):
    """What a reader saw on some photos, with the tokens it spent when it says."""

    texts: Sequence[str]
    usage: Usage | None = None


class PhotoReader(Protocol):
    """Model that writes down what a photo of a device shows."""

    def read(self, photos: Sequence[Photo]) -> PhotoReadings:
        """Return one text per photo, in the order given.

        A photo that shows nothing readable gets an empty text. The texts are
        derived from pictures the customer sent and are untrusted.
        """
        ...


class PageImage(NamedTuple):
    """A page rendered to pixels: rows of red, green and blue bytes, top row first."""

    width: int
    height: int
    rgb: bytes


class OcrEngine(Protocol):
    """Character recognition for document pages that have no text layer."""

    @property
    def fingerprint(self) -> str:
        """Name the engine, its models and settings.

        A stored document is read again when this changes.
        """
        ...

    def recognize(self, image: PageImage) -> str:
        """Return the text on the page, one line of the page per line, or an empty string."""
        ...


class DocumentProvider(Protocol):
    """Source of the documents that describe a device."""

    def documents_for(self, *, tenant_id: str, device_id: str) -> Sequence[Document]:
        """Return the documents of the device and of its model, within one tenant."""
        ...


class Selection(NamedTuple):
    """The pages chosen for one report, and the tokens spent on choosing them."""

    documents: Sequence[Document]
    embedding_tokens: int = 0


class PageSelector(Protocol):
    """Chooser of the document pages worth sending to the model for one report."""

    def select(self, report: FaultReport, documents: Sequence[Document]) -> Selection:
        """Return the documents cut down to the pages relevant to the report."""
        ...


class Embeddings(NamedTuple):
    """Vectors of some texts, with the tokens billed for them when the provider says."""

    vectors: Sequence[Sequence[float]]
    tokens: int = 0


class Embedder(Protocol):
    """Model that turns text into vectors, so that texts can be compared by meaning."""

    def embed_pages(self, texts: Sequence[str]) -> Embeddings:
        """Return one vector per document page, in the order given."""
        ...

    def embed_query(self, text: str) -> Embeddings:
        """Return the one vector of a fault report that pages are searched with."""
        ...


class TextSplitter(Protocol):
    """Cuts the text of a page into smaller pieces that pages are ranked by."""

    @property
    def description(self) -> str:
        """A short name of the splitting for evaluation reports."""
        ...

    def split(self, text: str) -> Sequence[str]:
        """Return the pieces of ``text``, in order."""
        ...


@runtime_checkable
class PageRanker(Protocol):
    """A page selector that can also say in which order it ranked the pages it keeps."""

    def rank(self, report: FaultReport, documents: Sequence[Document]) -> Sequence[tuple[str, int]]:
        """Return the kept pages as ``(document id, page number)``, best first."""
        ...


class HistoryProvider(Protocol):
    """Source of past service records."""

    def records_for(self, *, tenant_id: str, device_id: str) -> Sequence[ServiceRecord]:
        """Return past repairs of the device and of its model, within one tenant."""
        ...
