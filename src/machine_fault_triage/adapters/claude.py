"""Claude behind the model and photo reader ports. Needs the ``anthropic`` extra."""

import base64
from collections.abc import Sequence
from typing import Literal

import anthropic
from anthropic.types import ImageBlockParam, OutputConfigParam, TextBlockParam
from pydantic import BaseModel, ValidationError

from machine_fault_triage.adapters._evidence import OPENING, with_boundary_rule, wrap
from machine_fault_triage.errors import ModelCallError
from machine_fault_triage.inputs import Photo
from machine_fault_triage.models import Usage, total_usage
from machine_fault_triage.ports import (
    Generation,
    GenerationRequest,
    GenerationStatus,
    PhotoReadings,
    UntrustedBlock,
    rejection,
)
from machine_fault_triage.prompts import PHOTO_READING

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ClaudeModel:
    """A Claude model that answers in the requested schema.

    Instructions go into the system prompt. Untrusted blocks go into the user
    message, each wrapped in a tag named after a keyed digest of the block, so
    document text cannot forge the end of its own block. The answer is
    constrained to the schema by the API and validated again here.
    """

    def __init__(
        self,
        model: str,
        *,
        effort: Effort | None = None,
        max_tokens: int = 16000,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        """Choose the model id, and optionally the effort level and a configured client.

        Without ``client`` the SDK reads its credentials from the environment.
        """
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic()

    def generate[OutputT: BaseModel](
        self, request: GenerationRequest, output_type: type[OutputT]
    ) -> Generation[OutputT]:
        """Answer the request as an instance of ``output_type``.

        Raises:
            ModelCallError: If the API call fails after the SDK's own retries.
        """
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=with_boundary_rule(request.instructions),
                messages=[{"role": "user", "content": _evidence(request.untrusted)}],
                output_config=self._output_config(output_type),
            )
        except anthropic.APIError as error:
            message_text = f"the Claude API call failed: {error}"
            raise ModelCallError(message_text) from error
        usage = _usage(message.usage)
        if message.stop_reason == "refusal":
            return Generation[OutputT](status=GenerationStatus.REFUSED, usage=usage)
        if message.stop_reason == "max_tokens":
            return Generation[OutputT](status=GenerationStatus.TRUNCATED, usage=usage)
        text = "".join(block.text for block in message.content if block.type == "text")
        try:
            output = output_type.model_validate_json(text)
        except ValidationError as error:
            return Generation[OutputT](
                status=GenerationStatus.INVALID, usage=usage, detail=rejection(error)
            )
        return Generation[OutputT](status=GenerationStatus.COMPLETED, output=output, usage=usage)

    def _output_config(self, output_type: type[BaseModel]) -> OutputConfigParam:
        config: OutputConfigParam = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(output_type)}
        }
        if self._effort is not None:
            config["effort"] = self._effort
        return config


class ClaudePhotoReader:
    """A Claude model behind the ``PhotoReader`` port.

    Each photo is read in a call of its own, so a reading can never be
    attributed to another photo. The photo is sent to the API as given,
    metadata included. Not yet tried against the API.
    """

    def __init__(
        self, model: str, *, max_tokens: int = 1000, client: anthropic.Anthropic | None = None
    ) -> None:
        """Choose the model and the most tokens one reading may take.

        Without ``client`` the SDK reads its credentials from the environment.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic()

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
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=PHOTO_READING,
                    messages=[{"role": "user", "content": [_image(photo)]}],
                )
            except anthropic.APIError as error:
                message_text = f"the Claude API call failed: {error}"
                raise ModelCallError(message_text) from error
            read = message.stop_reason not in {"refusal", "max_tokens"}
            text = "".join(block.text for block in message.content if block.type == "text")
            texts.append(text.strip() if read else "")
            usages.append(_usage(message.usage))
        return PhotoReadings(texts, total_usage(usages))


def _image(photo: Photo) -> ImageBlockParam:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": photo.media_type,
            "data": base64.b64encode(photo.data).decode("ascii"),
        },
    }


def _usage(usage: anthropic.types.Usage) -> Usage:
    written = usage.cache_creation_input_tokens or 0
    read = usage.cache_read_input_tokens or 0
    return Usage(
        input_tokens=usage.input_tokens + written + read,
        output_tokens=usage.output_tokens,
        cached_input_tokens=read,
    )


def _evidence(blocks: tuple[UntrustedBlock, ...]) -> list[TextBlockParam]:
    return [
        {"type": "text", "text": OPENING},
        *(TextBlockParam(type="text", text=wrap(block)) for block in blocks),
    ]
