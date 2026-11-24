"""OpenAI models behind the ``StructuredModel`` port. Needs the ``openai`` extra."""

from typing import Any, Literal

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputMessageContentListParam,
    ResponseInputTextParam,
    ResponseTextConfigParam,
)
from openai.types.shared_params import Reasoning
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

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class OpenAIModel:
    """A model served through the OpenAI Responses API that answers in the requested schema.

    Instructions go into the ``instructions`` field. Untrusted blocks go into
    the user message, each wrapped in a tag whose name is random per request,
    so document text cannot forge the end of its own block. The answer is
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
        boundary = new_boundary()
        evidence = EasyInputMessageParam(
            role="user", content=_evidence(request.untrusted, boundary)
        )
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
                instructions=with_boundary_rule(request.instructions, boundary),
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
        except ValidationError:
            return Generation[OutputT](status=GenerationStatus.INVALID, usage=usage)
        return Generation[OutputT](status=GenerationStatus.COMPLETED, output=output, usage=usage)


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
    return Usage(
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens
    )


def _evidence(
    blocks: tuple[UntrustedBlock, ...], boundary: str
) -> ResponseInputMessageContentListParam:
    return [
        {"type": "input_text", "text": OPENING},
        *(
            ResponseInputTextParam(type="input_text", text=wrap(block, boundary))
            for block in blocks
        ),
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
