"""The baseline triage run: all evidence in one model call, then verification."""

from collections.abc import Sequence

from machine_fault_triage.drafts import DraftTriage
from machine_fault_triage.errors import InvalidRequestError
from machine_fault_triage.grounding import evidence_level, ground, unsupported_figures
from machine_fault_triage.inputs import (
    Document,
    FaultReport,
    ServiceRecord,
    without_superseded,
)
from machine_fault_triage.models import EvidenceLevel, TriageResult, TriageStatus
from machine_fault_triage.ports import (
    DocumentProvider,
    GenerationRequest,
    GenerationStatus,
    HistoryProvider,
    PageSelector,
    StructuredModel,
    UntrustedBlock,
)
from machine_fault_triage.prompts import CURRENT_PROMPT_VERSION, instructions

_STATUS_WITHOUT_OUTPUT = {
    GenerationStatus.REFUSED: (TriageStatus.REFUSED, "The model declined to answer."),
    GenerationStatus.TRUNCATED: (TriageStatus.FAILED, "The model answer was cut off."),
    GenerationStatus.INVALID: (TriageStatus.FAILED, "The model answer did not match the schema."),
}


def run_triage(  # noqa: PLR0913
    report: FaultReport,
    *,
    tenant_id: str,
    device_id: str,
    model: StructuredModel,
    documents: DocumentProvider,
    history: HistoryProvider,
    pages: PageSelector | None = None,
    prompt_version: int = CURRENT_PROMPT_VERSION,
) -> TriageResult:
    """Produce a grounded hint for the technician from one fault report.

    A document that another supplied document supersedes is left out.
    Without ``pages`` every page of every document is sent to the model. With
    it, only the selected pages are sent, and a quote counts as verified only
    if it is found on one of them. ``prompt_version`` selects an older version
    of the instructions, to repeat a past evaluation.

    The result carries the tokens the run spent, so that the caller can log
    them per tenant and device: those of the model call in ``usage`` and
    those of page selection in ``embedding_tokens``.

    Raises:
        InvalidRequestError: If ``tenant_id`` or ``device_id`` is blank, or if
            no prompt has the given version.
    """
    if not tenant_id.strip() or not device_id.strip():
        message = "tenant_id and device_id must not be blank"
        raise InvalidRequestError(message)
    prompt = instructions(prompt_version)
    device_documents = without_superseded(
        documents.documents_for(tenant_id=tenant_id, device_id=device_id)
    )
    embedding_tokens = 0
    if pages is not None:
        selection = pages.select(report, device_documents)
        device_documents, embedding_tokens = selection.documents, selection.embedding_tokens
    records = history.records_for(tenant_id=tenant_id, device_id=device_id)
    request = _request(prompt, report, device_documents, records)
    generation = model.generate(request, DraftTriage)
    spent = {"usage": generation.usage, "embedding_tokens": embedding_tokens}
    if generation.output is None:
        status, summary = _STATUS_WITHOUT_OUTPUT[generation.status]
        return _without_causes(status, summary, report, prompt_version).model_copy(update=spent)
    figures = unsupported_figures(
        _answer_texts(generation.output),
        (f"{block.label}\n{block.content}" for block in request.untrusted),
    )
    result = _result(generation.output, report, device_documents, records, prompt_version)
    return result.model_copy(update={"unsupported_figures": figures, **spent})


def _request(
    prompt: str,
    report: FaultReport,
    documents: Sequence[Document],
    records: Sequence[ServiceRecord],
) -> GenerationRequest:
    pages = tuple(
        UntrustedBlock(
            label=f"document id={document.id} title={document.title!r} page={page.number}",
            content=page.text,
        )
        for document in documents
        for page in document.pages
    )
    history = tuple(
        UntrustedBlock(
            label=f"service record id={record.id} date={record.recorded_on.isoformat()}"
            f" scope={record.scope.value}",
            content="\n".join((record.summary, *record.parts_replaced)),
        )
        for record in records
    )
    return GenerationRequest(
        instructions=prompt,
        untrusted=(
            UntrustedBlock(label="fault report", content=report.description),
            *pages,
            *history,
        ),
    )


def _result(
    draft: DraftTriage,
    report: FaultReport,
    documents: Sequence[Document],
    records: Sequence[ServiceRecord],
    prompt_version: int,
) -> TriageResult:
    if not draft.enough_evidence or not draft.causes:
        return TriageResult(
            status=TriageStatus.NOT_ENOUGH_DATA,
            summary=draft.summary,
            evidence_level=EvidenceLevel.NONE,
            language=report.language,
            questions_for_customer=draft.questions_for_customer,
            safety_warnings=draft.safety_warnings,
            missing=draft.missing,
            prompt_version=prompt_version,
        )
    causes = ground(draft.causes, documents, records)
    return TriageResult(
        status=TriageStatus.OK,
        summary=draft.summary,
        evidence_level=evidence_level(causes),
        language=report.language,
        causes=causes,
        parts_to_bring=draft.parts_to_bring,
        safety_warnings=draft.safety_warnings,
        questions_for_customer=draft.questions_for_customer,
        missing=draft.missing,
        prompt_version=prompt_version,
    )


def _answer_texts(draft: DraftTriage) -> list[str]:
    texts = [draft.summary, *draft.parts_to_bring, *draft.safety_warnings]
    for cause in draft.causes:
        texts.extend((cause.title, cause.why, cause.fix or "", *cause.checks))
    return texts


def _without_causes(
    status: TriageStatus, summary: str, report: FaultReport, prompt_version: int
) -> TriageResult:
    return TriageResult(
        status=status,
        summary=summary,
        evidence_level=EvidenceLevel.NONE,
        language=report.language,
        prompt_version=prompt_version,
    )
