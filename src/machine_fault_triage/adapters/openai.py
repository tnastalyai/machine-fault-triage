"""OpenAI models behind the model and embedder ports. Needs the ``openai`` extra."""

import base64
from collections.abc import Sequence
from typing import Any, Literal

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputTextParam,
    ResponseTextConfigParam,
)
from openai.types.shared_params import Reasoning
from pydantic import BaseModel, ValidationError

from machine_fault_triage.adapters._evidence import OPENING, with_boundary_rule, wrap
from machine_fault_triage.errors import ModelCallError
from machine_fault_triage.inputs import Photo
from machine_fault_triage.models import Usage, total_usage
from machine_fault_triage.ports import (
    Embeddings,
    Generation,
    GenerationRequest,
    GenerationStatus,
    PhotoReadings,
    UntrustedBlock,
    rejection,
)
from machine_fault_triage.prompts import PHOTO_READING

_EMBEDDING_BATCH = 32

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class OpenAIModel:
    """A model served through the OpenAI Responses API that answers in the requested schema.

    Instructions go into the ``instructions`` field. Untrusted blocks go into
    the user message, each wrapped in a tag named after a keyed digest of the
    block, so document text cannot forge the end of its own block. The answer is
    constrained to the schema by the API and validated again here.
    """

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: ReasoningEffort | None = None,
        max_output_tokens: int = 16000,
        client: openai.OpenAI | None = None,
    ) -> None:
        """Choose the model id, and optionally the reasoning effort and a configured client.

        Without ``client`` the SDK reads its credentials from the environment.
        """
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_output_tokens = max_output_tokens
        self._client = client or openai.OpenAI()

    def generate[OutputT: BaseModel](
        self, request: GenerationRequest, output_type: type[OutputT]
    ) -> Generation[OutputT]:
        """Answer the request as an instance of ``output_type``.

        Raises:
            ModelCallError: If the API call fails after the SDK's own retries.
        """
        evidence = EasyInputMessageParam(role="user", content=_evidence(request.untrusted))
        answer_format = ResponseTextConfigParam(
            format={
                "type": "json_schema",
                "strict": True,
                "name": output_type.__name__,
                "schema": _strict_schema(output_type),
            }
        )
        reasoning = (
            openai.omit
            if self._reasoning_effort is None
            else Reasoning(effort=self._reasoning_effort)
        )
        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=with_boundary_rule(request.instructions),
                input=[evidence],
                text=answer_format,
                max_output_tokens=self._max_output_tokens,
                reasoning=reasoning,
            )
        except openai.OpenAIError as error:
            message = f"the OpenAI API call failed: {error}"
            raise ModelCallError(message) from error
        usage = _usage(response)
        status = _status_without_output(response)
        if status is not None:
            return Generation[OutputT](status=status, usage=usage)
        try:
            output = output_type.model_validate_json(response.output_text)
        except ValidationError as error:
            return Generation[OutputT](
                status=GenerationStatus.INVALID, usage=usage, detail=rejection(error)
            )
        return Generation[OutputT](status=GenerationStatus.COMPLETED, output=output, usage=usage)


class OpenAIEmbedder:
    """An OpenAI embedding model behind the ``Embedder`` port.

    Pages and fault reports are sent to the embeddings endpoint as they are.
    Wrap it in ``CachedEmbedder`` to pay for each page only once.
    """

    def __init__(self, model: str, *, client: openai.OpenAI | None = None) -> None:
        """Choose the embedding model id and optionally a configured client.

        Without ``client`` the SDK reads its credentials from the environment.
        """
        self._model = model
        self._client = client or openai.OpenAI()

    def embed_pages(self, texts: Sequence[str]) -> Embeddings:
        """Return one vector per page, in the order given, and the tokens billed.

        Raises:
            ModelCallError: If the API call fails after the SDK's own retries.
        """
        vectors: list[Sequence[float]] = []
        tokens = 0
        for start in range(0, len(texts), _EMBEDDING_BATCH):
            batch = self._embed(list(texts[start : start + _EMBEDDING_BATCH]))
            vectors.extend(batch.vectors)
            tokens += batch.tokens
        return Embeddings(vectors, tokens)

    def embed_query(self, text: str) -> Embeddings:
        """Return the vector of a fault report and the tokens billed.

        Raises:
            ModelCallError: If the API call fails after the SDK's own retries.
        """
        return self._embed([text])

    def _embed(self, texts: list[str]) -> Embeddings:
        try:
            response = self._client.embeddings.create(
                model=self._model, input=texts, encoding_format="float"
            )
        except openai.OpenAIError as error:
            message = f"the OpenAI API call failed: {error}"
            raise ModelCallError(message) from error
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        return Embeddings(vectors, response.usage.total_tokens if response.usage else 0)


class OpenAIPhotoReader:
    """An OpenAI model that can see images, behind the ``PhotoReader`` port.

    Each photo is read in a call of its own, so a reading can never be
    attributed to another photo. The photo is sent to the API as given,
    metadata included. Not yet tried on real photos of displays.
    """

    def __init__(
        self, model: str, *, max_output_tokens: int = 1000, client: openai.OpenAI | None = None
    ) -> None:
        """Choose the model and the most tokens one reading may take.

        Without ``client`` the SDK reads its credentials from the environment.
        """
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._client = client or openai.OpenAI()

    def read(self, photos: Sequence[Photo]) -> PhotoReadings:
        """Return what each photo shows, in the order given.

        A photo the model declines to read, or does not finish reading, gets
        an empty text.

        Raises:
            ModelCallError: If an API call fails after the SDK's own retries.
        """
        texts = []
        usages = []
        for photo in photos:
            try:
                response = self._client.responses.create(
                    model=self._model,
                    instructions=PHOTO_READING,
                    input=[EasyInputMessageParam(role="user", content=[_image(photo)])],
                    max_output_tokens=self._max_output_tokens,
                )
            except openai.OpenAIError as error:
                message = f"the OpenAI API call failed: {error}"
                raise ModelCallError(message) from error
            read = _status_without_output(response) is None
            texts.append(response.output_text.strip() if read else "")
            usages.append(_usage(response))
        return PhotoReadings(texts, total_usage(usages))


def _image(photo: Photo) -> ResponseInputImageParam:
    encoded = base64.b64encode(photo.data).decode("ascii")
    return ResponseInputImageParam(
        type="input_image",
        detail="high",
        image_url=f"data:{photo.media_type};base64,{encoded}",
    )


def _status_without_output(response: Response) -> GenerationStatus | None:
    refused = any(
        content.type == "refusal"
        for item in response.output
        if item.type == "message"
        for content in item.content
    )
    reason = response.incomplete_details.reason if response.incomplete_details else None
    if refused or reason == "content_filter":
        return GenerationStatus.REFUSED
    if response.status == "incomplete":
        return GenerationStatus.TRUNCATED
    return None


def _usage(response: Response) -> Usage | None:
    if response.usage is None:
        return None
    details = response.usage.input_tokens_details
    return Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cached_input_tokens=details.cached_tokens if details else 0,
    )


def _evidence(blocks: tuple[UntrustedBlock, ...]) -> ResponseInputMessageContentListParam:
    return [
        {"type": "input_text", "text": OPENING},
        *(ResponseInputTextParam(type="input_text", text=wrap(block)) for block in blocks),
    ]


def _strict_schema(output_type: type[BaseModel]) -> dict[str, Any]:
    schema = openai.pydantic_function_tool(output_type)["function"].get("parameters", {})
    rewritten: dict[str, Any] = _without_one_of(schema)
    return rewritten


def _without_one_of(node: Any) -> Any:  # noqa: ANN401
    if isinstance(node, list):
        return [_without_one_of(item) for item in node]
    if not isinstance(node, dict):
        return node
    rewritten = {
        key: _without_one_of(value) for key, value in node.items() if key != "discriminator"
    }
    if "oneOf" in rewritten:
        rewritten["anyOf"] = rewritten.pop("oneOf")
    return rewritten
