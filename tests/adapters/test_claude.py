import json
import subprocess
import sys
from typing import Any

import anthropic
import httpx2
import pytest

from machine_fault_triage import (
    GenerationRequest,
    GenerationStatus,
    Photo,
    UntrustedBlock,
    Usage,
)
from machine_fault_triage.adapters.claude import ClaudeModel, ClaudePhotoReader, Effort
from machine_fault_triage.drafts import DraftTriage
from machine_fault_triage.errors import ModelCallError
from machine_fault_triage.prompts import PHOTO_READING
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
    assert generation.usage == Usage(input_tokens=100, output_tokens=5, cached_input_tokens=90)


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


def test_repeated_request_is_sent_byte_for_byte_with_no_tag_named_in_the_system_prompt() -> None:
    api = FakeApi()
    model = model_for(api)

    model.generate(REQUEST, DraftTriage)
    model.generate(REQUEST, DraftTriage)

    first, second = api.requests
    tag = first["messages"][0]["content"][1]["text"].split("\n", maxsplit=1)[0]
    assert first == second
    assert tag.startswith("<evidence-")
    assert tag.strip("<>") not in first["system"]
    assert tag.strip("<>").rsplit("-", maxsplit=1)[0] in first["system"]


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
    assert generation.usage == Usage(input_tokens=100, output_tokens=5, cached_input_tokens=90)
    assert (generation.detail is not None) == (status is GenerationStatus.INVALID)


def test_invalid_answer_names_the_fields_that_broke_the_schema() -> None:
    generation = model_for(FakeApi(text='{"summary": 7}')).generate(REQUEST, DraftTriage)

    assert generation.detail == (
        "summary: Input should be a valid string; enough_evidence: Field required"
    )


def test_failed_api_call_is_raised_as_a_library_error() -> None:
    with pytest.raises(ModelCallError, match="the Claude API call failed"):
        model_for(FakeApi(status=500)).generate(REQUEST, DraftTriage)


def test_client_is_created_from_the_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert isinstance(ClaudeModel("claude-test"), ClaudeModel)


PHOTO = Photo(media_type="image/jpeg", data=b"\xff\xd8 display")


def reader_for(api: FakeApi) -> ClaudePhotoReader:
    client = anthropic.Anthropic(
        api_key="test-key",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(api)),
    )
    return ClaudePhotoReader("claude-test", client=client)


def test_each_photo_is_read_in_a_call_of_its_own_and_the_usage_is_added_up() -> None:
    api = FakeApi(text="  Display shows E7.\n")

    readings = reader_for(api).read([PHOTO, PHOTO])

    assert readings.texts == ["Display shows E7.", "Display shows E7."]
    assert readings.usage == Usage(input_tokens=200, output_tokens=10, cached_input_tokens=180)
    assert len(api.requests) == len(readings.texts)


def test_photo_travels_as_an_image_apart_from_the_fixed_reading_instructions() -> None:
    api = FakeApi(text="Display shows E7.")

    reader_for(api).read([PHOTO])

    (sent,) = api.requests
    (message,) = sent["messages"]
    assert sent["system"] == PHOTO_READING
    assert message["content"] == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9ggZGlzcGxheQ=="},
        }
    ]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_photo_the_model_declines_or_does_not_finish_gets_an_empty_reading(
    stop_reason: str,
) -> None:
    readings = reader_for(FakeApi(text="E", stop_reason=stop_reason)).read([PHOTO])

    assert readings.texts == [""]
    assert readings.usage == Usage(input_tokens=100, output_tokens=5, cached_input_tokens=90)


def test_failed_photo_reading_call_is_raised_as_a_library_error() -> None:
    with pytest.raises(ModelCallError, match="the Claude API call failed"):
        reader_for(FakeApi(status=500)).read([PHOTO])


def test_photo_reader_is_created_from_the_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    assert isinstance(ClaudePhotoReader("claude-test"), ClaudePhotoReader)


def test_core_package_does_not_import_a_provider_sdk() -> None:
    script = (
        "import sys, machine_fault_triage.cli;"
        "optional = ('anthropic', 'openai', 'pypdfium2', 'chonkie');"
        "sys.exit(any(name in sys.modules for name in optional))"
    )

    completed = subprocess.run([sys.executable, "-c", script], check=False)  # noqa: S603

    assert completed.returncode == 0
