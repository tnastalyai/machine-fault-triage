"""Test doubles for applications and adapters built on this library."""

from collections import deque
from collections.abc import Iterable

from pydantic import BaseModel

from machine_fault_triage.errors import TriageError
from machine_fault_triage.ports import Generation, GenerationRequest, GenerationStatus


class FakeModelExhaustedError(TriageError):
    """A fake model was called more often than it was scripted for."""


class FakeModel:
    """A scripted model that never touches the network.

    Each call consumes the next scripted reply. A reply is either the output to
    return or a non-completed status such as a refusal. Requests are recorded
    so a test can assert on what the model was shown.
    """

    def __init__(self, replies: Iterable[BaseModel | GenerationStatus]) -> None:
        """Script the replies returned by successive calls."""
        self._replies = deque(replies)
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
        if isinstance(reply, GenerationStatus):
            return Generation[OutputT](status=reply)
        if not isinstance(reply, output_type):
            return Generation[OutputT](status=GenerationStatus.INVALID)
        return Generation[OutputT](status=GenerationStatus.COMPLETED, output=reply)
