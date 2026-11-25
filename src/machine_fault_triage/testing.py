"""Test doubles for applications and adapters built on this library."""

from collections import deque
from collections.abc import Iterable, Mapping, Sequence

from pydantic import BaseModel

from machine_fault_triage.errors import TriageError
from machine_fault_triage.inputs import Photo
from machine_fault_triage.models import Usage
from machine_fault_triage.ports import (
    Embeddings,
    Generation,
    GenerationRequest,
    GenerationStatus,
    PageImage,
    PhotoReadings,
)


class FakeModelExhaustedError(TriageError):
    """A fake model was called more often than it was scripted for."""


class FakeModel:
    """A scripted model that never touches the network.

    Each call consumes the next scripted reply. A reply is either the output to
    return or a non-completed status such as a refusal. Requests are recorded
    so a test can assert on what the model was shown.
    """

    def __init__(
        self,
        replies: Iterable[BaseModel | GenerationStatus],
        *,
        usage: Usage | None = None,
        invalid_detail: str | None = None,
    ) -> None:
        """Script the replies returned by successive calls, and the usage each one reports.

        ``invalid_detail`` is the reason given with every invalid reply.
        """
        self._replies = deque(replies)
        self._usage = usage
        self._invalid_detail = invalid_detail
        self.requests: list[GenerationRequest] = []

    def generate[OutputT: BaseModel](
        self, request: GenerationRequest, output_type: type[OutputT]
    ) -> Generation[OutputT]:
        """Return the next scripted reply."""
        self.requests.append(request)
        if not self._replies:
            message = "the fake model has no scripted reply left"
            raise FakeModelExhaustedError(message)
        reply = self._replies.popleft()
        if not isinstance(reply, GenerationStatus | output_type):
            reply = GenerationStatus.INVALID
        if isinstance(reply, GenerationStatus):
            detail = self._invalid_detail if reply is GenerationStatus.INVALID else None
            return Generation[OutputT](status=reply, usage=self._usage, detail=detail)
        return Generation[OutputT](
            status=GenerationStatus.COMPLETED, output=reply, usage=self._usage
        )


class FakeEmbedder:
    """A scripted embedder that never touches the network.

    Vectors are looked up by a word the text contains, so a test can state
    which texts mean the same. Texts that contain none of the words get a
    vector that is similar to nothing. Calls are recorded.
    """

    def __init__(
        self, meanings: Mapping[str, Sequence[float]], *, tokens_per_text: int = 0
    ) -> None:
        """Script the vector returned for every text that contains a given word.

        ``tokens_per_text`` is what the embedder reports as billed for each text.
        """
        self._meanings = dict(meanings)
        self._tokens_per_text = tokens_per_text
        self._size = len(next(iter(self._meanings.values()), ()))
        self.page_calls: list[list[str]] = []
        self.queries: list[str] = []

    def embed_pages(self, texts: Sequence[str]) -> Embeddings:
        """Return the scripted vector of each page."""
        self.page_calls.append(list(texts))
        return Embeddings(
            [self._vector(text) for text in texts], self._tokens_per_text * len(texts)
        )

    def embed_query(self, text: str) -> Embeddings:
        """Return the scripted vector of a fault report."""
        self.queries.append(text)
        return Embeddings([self._vector(text)], self._tokens_per_text)

    def _vector(self, text: str) -> Sequence[float]:
        for word, vector in self._meanings.items():
            if word in text:
                return vector
        return [0.0] * self._size


class FakePhotoReader:
    """A scripted photo reader that never touches the network.

    Each photo consumes the next scripted reading. Calls are recorded.
    """

    def __init__(self, readings: Iterable[str], *, usage: Usage | None = None) -> None:
        """Script the readings returned for successive photos, and the usage of each call."""
        self._readings = deque(readings)
        self._usage = usage
        self.calls: list[list[Photo]] = []

    def read(self, photos: Sequence[Photo]) -> PhotoReadings:
        """Return the next scripted reading for each photo."""
        self.calls.append(list(photos))
        if len(self._readings) < len(photos):
            message = "the fake photo reader has no scripted reading left"
            raise FakeModelExhaustedError(message)
        return PhotoReadings([self._readings.popleft() for _ in photos], self._usage)


class FakeOcrEngine:
    """A scripted recognition engine that reads no pixels.

    Each page consumes the next scripted text. The images are recorded.
    """

    def __init__(self, texts: Iterable[str], *, fingerprint: str = "fake engine") -> None:
        """Script the text returned for successive pages, and the engine's fingerprint."""
        self._texts = deque(texts)
        self._fingerprint = fingerprint
        self.images: list[PageImage] = []

    @property
    def fingerprint(self) -> str:
        """The scripted fingerprint."""
        return self._fingerprint

    def recognize(self, image: PageImage) -> str:
        """Return the next scripted text."""
        self.images.append(image)
        if not self._texts:
            message = "the fake recognition engine has no scripted text left"
            raise FakeModelExhaustedError(message)
        return self._texts.popleft()
