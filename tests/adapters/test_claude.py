import json
import subprocess
import sys
from typing import Any

import anthropic
import httpx2
import pytest

from machine_fault_triage import GenerationRequest, GenerationStatus, UntrustedBlock, Usage
from machine_fault_triage.adapters.claude import ClaudeModel, Effort
from machine_fault_triage.drafts import DraftTriage
from machine_fault_triage.errors import ModelCallError
from tests.factories import draft_cause, draft_triage, manual_citation

REQUEST = GenerationRequest(
    instructions="Diagnose the fault.",
    untrusted=(
        UntrustedBlock(label="fault report", content="Display shows E7."),
        UntrustedBlock(label="document id=manual-1 page=77", content="E7: flow sensor"),
    ),
)


class FakeApi:
    def __init__(
        self, *, text: str = "{}", stop_reason: str = "end_turn", status: int = 200
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self._text = text
        self._stop_reason = stop_reason
        self._status = status

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self._status != httpx2.codes.OK:
            error = {"type": "error", "error": {"type": "api_error", "message": "boom"}}
            return httpx2.Response(self._status, json=error)
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "s"},
                    {"type": "text", "text": self._text},
                ],
                "stop_reason": self._stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 90},
            },
        )


def model_for(api: FakeApi, effort: Effort | None = None) -> ClaudeModel:
    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(api)),
    )
    return ClaudeModel("claude-test", client=client, effort=effort)


def test_schema_valid_answer_is_returned_as_the_requested_type() -> None:
    draft = draft_triage([draft_cause([manual_citation()])])
    api = FakeApi(text=draft.model_dump_json())

    generation = model_for(api).generate(REQUEST, DraftTriage)

    assert generation.status is GenerationStatus.COMPLETED
    assert generation.output == draft
    assert generation.usage == Usage(input_tokens=100, output_tokens=5)


def test_instructions_go_to_the_system_prompt_and_evidence_to_the_user_message() -> None:
    api = FakeApi()

    model_for(api).generate(REQUEST, DraftTriage)

    (sent,) = api.requests
    (message,) = sent["messages"]
    evidence = [block["text"] for block in message["content"][1:]]
    assert sent["model"] == "claude-test"
    assert sent["system"].startswith("Diagnose the fault.")
    assert "Display shows E7." not in sent["system"]
    assert message["role"] == "user"
    assert "source: fault report\nDisplay shows E7." in evidence[0]
    assert "source: document id=manual-1 page=77\nE7: flow sensor" in evidence[1]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert "effort" not in sent["output_config"]


def test_evidence_is_wrapped_in_a_tag_that_changes_with_every_request() -> None:
    api = FakeApi()
    model = model_for(api)

    model.generate(REQUEST, DraftTriage)
    model.generate(REQUEST, DraftTriage)

    first, second = (sent["messages"][0]["content"][1]["text"] for sent in api.requests)
    tag = first.split("\n", maxsplit=1)[0]
    assert tag.startswith("<evidence-")
    assert first.endswith(tag.replace("<", "</"))
    assert tag.strip("<>") in api.requests[0]["system"]
    assert first != second


def test_effort_is_sent_when_configured() -> None:
    api = FakeApi()

    model_for(api, effort="low").generate(REQUEST, DraftTriage)

    assert api.requests[0]["output_config"]["effort"] == "low"


@pytest.mark.parametrize(
    ("stop_reason", "text", "status"),
    [
        ("refusal", "", GenerationStatus.REFUSED),
        ("max_tokens", '{"summary": "E7 poi', GenerationStatus.TRUNCATED),
        ("end_turn", '{"summary": 7}', GenerationStatus.INVALID),
    ],
)
def test_answer_without_usable_output_is_reported_through_the_status(
    stop_reason: str, text: str, status: GenerationStatus
) -> None:
    api = FakeApi(text=text, stop_reason=stop_reason)

    generation = model_for(api).generate(REQUEST, DraftTriage)

    assert generation.status is status
    assert generation.output is None
    assert generation.usage == Usage(input_tokens=100, output_tokens=5)


def test_failed_api_call_is_raised_as_a_library_error() -> None:
    with pytest.raises(ModelCallError, match="the Claude API call failed"):
        model_for(FakeApi(status=500)).generate(REQUEST, DraftTriage)


def test_client_is_created_from_the_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert isinstance(ClaudeModel("claude-test"), ClaudeModel)


def test_core_package_does_not_import_the_provider_sdk() -> None:
    script = "import sys, machine_fault_triage.cli; sys.exit('anthropic' in sys.modules)"

    completed = subprocess.run([sys.executable, "-c", script], check=False)  # noqa: S603

    assert completed.returncode == 0
