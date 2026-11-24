"""Claude behind the ``StructuredModel`` port. Needs the ``anthropic`` extra."""

from typing import Literal

import anthropic
from anthropic.types import OutputConfigParam, TextBlockParam
from pydantic import BaseModel, ValidationError

from machine_fault_triage.adapters._evidence import OPENING, new_boundary, with_boundary_rule, wrap
from machine_fault_triage.errors import ModelCallError
from machine_fault_triage.ports import (
    Generation,
    GenerationRequest,
    GenerationStatus,
    UntrustedBlock,
    Usage,
)

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ClaudeModel:
    """A Claude model that answers in the requested schema.

    Instructions go into the system prompt. Untrusted blocks go into the user
    message, each wrapped in a tag whose name is random per request, so
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
        boundary = new_boundary()
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=with_boundary_rule(request.instructions, boundary),
                messages=[{"role": "user", "content": _evidence(request.untrusted, boundary)}],
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
        except ValidationError:
            return Generation[OutputT](status=GenerationStatus.INVALID, usage=usage)
        return Generation[OutputT](status=GenerationStatus.COMPLETED, output=output, usage=usage)

    def _output_config(self, output_type: type[BaseModel]) -> OutputConfigParam:
        config: OutputConfigParam = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(output_type)}
        }
        if self._effort is not None:
            config["effort"] = self._effort
        return config


def _usage(usage: anthropic.types.Usage) -> Usage:
    cached = (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
    return Usage(input_tokens=usage.input_tokens + cached, output_tokens=usage.output_tokens)


def _evidence(blocks: tuple[UntrustedBlock, ...], boundary: str) -> list[TextBlockParam]:
    return [
        {"type": "text", "text": OPENING},
        *(TextBlockParam(type="text", text=wrap(block, boundary)) for block in blocks),
    ]
