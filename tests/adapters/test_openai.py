import json
from typing import Any

import httpx2
import openai
import pytest

from machine_fault_triage import GenerationRequest, GenerationStatus, UntrustedBlock, Usage
from machine_fault_triage.adapters.openai import OpenAIModel, ReasoningEffort
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
USAGE = {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}


class FakeApi:
    def __init__(self, body: dict[str, Any] | None = None, *, status: int = 200) -> None:
        self.requests: list[dict[str, Any]] = []
        self._body = body if body is not None else answer("{}")
        self._status = status

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self._status != httpx2.codes.OK:
            return httpx2.Response(self._status, json={"error": {"message": "boom"}})
        return httpx2.Response(200, json=self._body)


def answer(text: str, **overrides: object) -> dict[str, Any]:
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-test",
        "status": "completed",
        "output": [{"type": "reasoning", "id": "rs_1", "summary": []}, message],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": USAGE,
        **overrides,
    }


def refusal() -> dict[str, Any]:
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "refusal", "refusal": "I cannot help with that."}],
    }
    return answer("", output=[message])


def model_for(api: FakeApi, reasoning_effort: ReasoningEffort | None = None) -> OpenAIModel:
    client = openai.OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(api)),
    )
    return OpenAIModel("gpt-test", client=client, reasoning_effort=reasoning_effort)


def test_schema_valid_answer_is_returned_with_its_usage() -> None:
    draft = draft_triage([draft_cause([manual_citation()])])
    api = FakeApi(answer(draft.model_dump_json()))

    generation = model_for(api).generate(REQUEST, DraftTriage)

    assert generation.status is GenerationStatus.COMPLETED
    assert generation.output == draft
    assert generation.usage == Usage(input_tokens=100, output_tokens=5)


def test_instructions_and_evidence_travel_in_separate_fields() -> None:
    api = FakeApi()

    model_for(api).generate(REQUEST, DraftTriage)

    (sent,) = api.requests
    (message,) = sent["input"]
    evidence = [block["text"] for block in message["content"][1:]]
    tag = evidence[0].split("\n", maxsplit=1)[0]
    assert sent["model"] == "gpt-test"
    assert sent["instructions"].startswith("Diagnose the fault.")
    assert "Display shows E7." not in sent["instructions"]
    assert message["role"] == "user"
    assert tag.startswith("<evidence-")
    assert tag.strip("<>") in sent["instructions"]
    assert "source: fault report\nDisplay shows E7." in evidence[0]
    assert "source: document id=manual-1 page=77\nE7: flow sensor" in evidence[1]
    assert "reasoning" not in sent


def test_schema_is_sent_in_the_strict_dialect_the_api_accepts() -> None:
    api = FakeApi()

    model_for(api).generate(REQUEST, DraftTriage)

    answer_format = api.requests[0]["text"]["format"]
    schema = json.dumps(answer_format["schema"])
    assert (answer_format["type"], answer_format["strict"]) == ("json_schema", True)
    assert answer_format["name"] == "DraftTriage"
    assert "oneOf" not in schema
    assert "discriminator" not in schema
    assert '"anyOf": [{"$ref": "#/$defs/DraftDocumentCitation"}' in schema


def test_reasoning_effort_is_sent_when_configured() -> None:
    api = FakeApi()

    model_for(api, reasoning_effort="low").generate(REQUEST, DraftTriage)

    assert api.requests[0]["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize(
    ("body", "status"),
    [
        (refusal(), GenerationStatus.REFUSED),
        (
            answer("", status="incomplete", incomplete_details={"reason": "content_filter"}),
            GenerationStatus.REFUSED,
        ),
        (
            answer(
                '{"summary": "E7 poi',
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            ),
            GenerationStatus.TRUNCATED,
        ),
        (answer('{"summary": 7}'), GenerationStatus.INVALID),
    ],
)
def test_answer_without_usable_output_is_reported_through_the_status(
    body: dict[str, Any], status: GenerationStatus
) -> None:
    generation = model_for(FakeApi(body)).generate(REQUEST, DraftTriage)

    assert generation.status is status
    assert generation.output is None
    assert generation.usage == Usage(input_tokens=100, output_tokens=5)


def test_missing_usage_is_reported_as_unknown() -> None:
    api = FakeApi(answer(draft_triage([]).model_dump_json(), usage=None))

    assert model_for(api).generate(REQUEST, DraftTriage).usage is None


def test_failed_api_call_is_raised_as_a_library_error() -> None:
    with pytest.raises(ModelCallError, match="the OpenAI API call failed"):
        model_for(FakeApi(status=500)).generate(REQUEST, DraftTriage)


def test_client_is_created_from_the_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert isinstance(OpenAIModel("gpt-test"), OpenAIModel)
