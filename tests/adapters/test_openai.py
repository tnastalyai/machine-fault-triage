import json
from typing import Any

import httpx2
import openai
import pytest

from machine_fault_triage import (
    GenerationRequest,
    GenerationStatus,
    Photo,
    UntrustedBlock,
    Usage,
)
from machine_fault_triage.adapters.openai import (
    OpenAIEmbedder,
    OpenAIModel,
    OpenAIPhotoReader,
    ReasoningEffort,
)
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
USAGE = {
    "input_tokens": 100,
    "input_tokens_details": {"cached_tokens": 40},
    "output_tokens": 5,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 105,
}


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
    assert generation.usage == Usage(input_tokens=100, output_tokens=5, cached_input_tokens=40)


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
    assert tag.strip("<>") not in sent["instructions"]
    assert tag.strip("<>").rsplit("-", maxsplit=1)[0] in sent["instructions"]
    assert "source: fault report\nDisplay shows E7." in evidence[0]
    assert "source: document id=manual-1 page=77\nE7: flow sensor" in evidence[1]
    assert "reasoning" not in sent


def test_repeated_request_is_sent_byte_for_byte_so_the_provider_can_cache_it() -> None:
    api = FakeApi()
    model = model_for(api)

    model.generate(REQUEST, DraftTriage)
    model.generate(REQUEST, DraftTriage)

    first, second = api.requests
    assert first == second


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
    assert generation.usage == Usage(input_tokens=100, output_tokens=5, cached_input_tokens=40)
    assert (generation.detail is not None) == (status is GenerationStatus.INVALID)


def test_missing_usage_is_reported_as_unknown() -> None:
    api = FakeApi(answer(draft_triage([]).model_dump_json(), usage=None))

    assert model_for(api).generate(REQUEST, DraftTriage).usage is None


def test_usage_without_a_cache_breakdown_counts_no_cached_tokens() -> None:
    plain = {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}
    api = FakeApi(answer(draft_triage([]).model_dump_json(), usage=plain))

    usage = model_for(api).generate(REQUEST, DraftTriage).usage

    assert usage == Usage(input_tokens=100, output_tokens=5)


def test_invalid_answer_names_the_fields_that_broke_the_schema() -> None:
    generation = model_for(FakeApi(answer('{"summary": 7}'))).generate(REQUEST, DraftTriage)

    assert generation.detail == (
        "summary: Input should be a valid string; enough_evidence: Field required"
    )


def test_failed_api_call_is_raised_as_a_library_error() -> None:
    with pytest.raises(ModelCallError, match="the OpenAI API call failed"):
        model_for(FakeApi(status=500)).generate(REQUEST, DraftTriage)


def test_client_is_created_from_the_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert isinstance(OpenAIModel("gpt-test"), OpenAIModel)


class FakeEmbeddingApi:
    def __init__(self, *, status: int = 200, reports_usage: bool = True) -> None:
        self.requests: list[dict[str, Any]] = []
        self._status = status
        self._reports_usage = reports_usage

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        sent = json.loads(request.content)
        self.requests.append(sent)
        if self._status != httpx2.codes.OK:
            return httpx2.Response(self._status, json={"error": {"message": "boom"}})
        data = [
            {"object": "embedding", "index": index, "embedding": [float(len(text)), 1.0]}
            for index, text in enumerate(sent["input"])
        ]
        tokens = sum(len(text) for text in sent["input"])
        usage = {"prompt_tokens": tokens, "total_tokens": tokens}
        body = {"object": "list", "model": sent["model"], "data": data[::-1]}
        if self._reports_usage:
            body["usage"] = usage
        return httpx2.Response(200, json=body)


def embedder_for(api: FakeEmbeddingApi) -> OpenAIEmbedder:
    client = openai.OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(api)),
    )
    return OpenAIEmbedder("embed-test", client=client)


def test_pages_come_back_as_vectors_in_the_order_given() -> None:
    api = FakeEmbeddingApi()

    vectors = embedder_for(api).embed_pages(["a", "bbb", "cc"])

    assert vectors == ([[1.0, 1.0], [3.0, 1.0], [2.0, 1.0]], 6)
    assert api.requests == [
        {"model": "embed-test", "input": ["a", "bbb", "cc"], "encoding_format": "float"}
    ]


def test_many_pages_are_sent_in_several_requests() -> None:
    api = FakeEmbeddingApi()

    vectors, tokens = embedder_for(api).embed_pages(["x" * (index + 1) for index in range(70)])

    assert [len(sent["input"]) for sent in api.requests] == [32, 32, 6]
    assert [vector[0] for vector in vectors] == [float(index + 1) for index in range(70)]
    assert tokens == sum(range(1, 71))


def test_fault_report_is_embedded_on_its_own() -> None:
    api = FakeEmbeddingApi()

    assert embedder_for(api).embed_query("abcd") == ([[4.0, 1.0]], 4)
    assert api.requests[0]["input"] == ["abcd"]


def test_embedding_answer_without_usage_counts_no_tokens() -> None:
    api = FakeEmbeddingApi(reports_usage=False)

    assert embedder_for(api).embed_query("abcd") == ([[4.0, 1.0]], 0)


def test_failed_embedding_call_is_raised_as_a_library_error() -> None:
    with pytest.raises(ModelCallError, match="the OpenAI API call failed"):
        embedder_for(FakeEmbeddingApi(status=500)).embed_query("abcd")


def test_embedder_client_is_created_from_the_environment_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    assert isinstance(OpenAIEmbedder("embed-test"), OpenAIEmbedder)


PHOTO = Photo(media_type="image/jpeg", data=b"\xff\xd8 display")


def reader_for(api: FakeApi) -> OpenAIPhotoReader:
    client = openai.OpenAI(
        api_key="test-key",
        max_retries=0,
        http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(api)),
    )
    return OpenAIPhotoReader("gpt-test", client=client)


def test_each_photo_is_read_in_a_call_of_its_own_and_the_usage_is_added_up() -> None:
    api = FakeApi(answer("  Display shows E7.\n"))

    readings = reader_for(api).read([PHOTO, PHOTO])

    assert readings.texts == ["Display shows E7.", "Display shows E7."]
    assert readings.usage == Usage(input_tokens=200, output_tokens=10, cached_input_tokens=80)
    assert len(api.requests) == len(readings.texts)


def test_photo_travels_as_an_image_apart_from_the_fixed_reading_instructions() -> None:
    api = FakeApi(answer("Display shows E7."))

    reader_for(api).read([PHOTO])

    (sent,) = api.requests
    (message,) = sent["input"]
    (image,) = message["content"]
    assert sent["instructions"] == PHOTO_READING
    assert image["type"] == "input_image"
    assert image["image_url"] == "data:image/jpeg;base64,/9ggZGlzcGxheQ=="


@pytest.mark.parametrize(
    "body",
    [
        refusal(),
        answer("E", status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
    ],
)
def test_photo_the_model_declines_or_does_not_finish_gets_an_empty_reading(
    body: dict[str, Any],
) -> None:
    readings = reader_for(FakeApi(body)).read([PHOTO])

    assert readings.texts == [""]
    assert readings.usage == Usage(input_tokens=100, output_tokens=5, cached_input_tokens=40)


def test_failed_photo_reading_call_is_raised_as_a_library_error() -> None:
    with pytest.raises(ModelCallError, match="the OpenAI API call failed"):
        reader_for(FakeApi(status=500)).read([PHOTO])


def test_reading_without_reported_usage_reports_none() -> None:
    readings = reader_for(FakeApi(answer("E7", usage=None))).read([PHOTO])

    assert readings.usage is None
