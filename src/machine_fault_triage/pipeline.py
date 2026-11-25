"""The baseline triage run: all evidence in one model call, then verification."""

from collections.abc import Sequence

from machine_fault_triage.drafts import DraftTriage
from machine_fault_triage.errors import InvalidRequestError, ModelCallError
from machine_fault_triage.grounding import (
    evidence_level,
    ground,
    unreported_figures,
    unsupported_figures,
)
from machine_fault_triage.inputs import (
    Document,
    FaultReport,
    ServiceRecord,
    without_superseded,
)
from machine_fault_triage.models import (
    EvidenceLevel,
    TriageResult,
    TriageStatus,
    Usage,
    total_usage,
)
from machine_fault_triage.ports import (
    DocumentProvider,
    Generation,
    GenerationRequest,
    GenerationStatus,
    HistoryProvider,
    PageSelector,
    PhotoReader,
    PhotoReadings,
    StructuredModel,
    UntrustedBlock,
)
from machine_fault_triage.prompts import (
    CURRENT_PROMPT_VERSION,
    RETRY_AFTER_INVALID,
    RETRY_AFTER_TRUNCATED,
    instructions,
)

_STATUS_WITHOUT_OUTPUT = {
    GenerationStatus.REFUSED: (TriageStatus.REFUSED, "The model declined to answer."),
    GenerationStatus.TRUNCATED: (TriageStatus.FAILED, "The model answer was cut off."),
    GenerationStatus.INVALID: (TriageStatus.FAILED, "The model answer did not match the schema."),
}
_LONGEST_DEVICE_MODEL = 200
_LONGEST_READING = 2000
_RETRY_NOTES = {
    GenerationStatus.INVALID: RETRY_AFTER_INVALID,
    GenerationStatus.TRUNCATED: RETRY_AFTER_TRUNCATED,
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
    evidence_limit: int = 1_000_000,
    report_limit: int = 20_000,
    retries: int = 1,
    device_model: str | None = None,
    photo_reader: PhotoReader | None = None,
    photo_limit: int = 5,
    photo_size_limit: int = 5_000_000,
) -> TriageResult:
    """Produce a grounded hint for the technician from one fault report.

    A document that another supplied document supersedes is left out.
    Without ``pages`` every page of every document is sent to the model. With
    it, only the selected pages are sent, and a quote counts as verified only
    if it is found on one of them. ``prompt_version`` selects an older version
    of the instructions, to repeat a past evaluation.

    ``device_model`` names the model and variant of the device, as the
    caller's records hold it. It reaches the model as evidence, so that it
    can keep to the values a page gives for that model. The instructions do
    not mention it, page selection does not use it, and no check in code
    proves that the quoted row belongs to that model.

    Photos of the report are read by ``photo_reader`` before pages are
    selected, so that a code shown only on a photo can find its page. What
    was read reaches the model as evidence and is returned in
    ``photo_readings`` for a person to compare with the photos. A misread
    code still leads to verified quotes, for the wrong code. ``photo_limit``
    caps the photos of one report and ``photo_size_limit`` the bytes of each.

    The result carries the tokens the run spent, so that the caller can log
    them per tenant and device: those of the model call in ``usage``, those
    of page selection in ``embedding_tokens`` and those of reading the photos
    in ``photo_usage``.

    ``evidence_limit`` caps the characters of evidence sent in one call, so
    that a request too large for the model is refused before it is paid for.
    The default is about 250,000 tokens of English text.

    ``report_limit`` caps the characters of the fault report; raise it for
    reports that arrive as long emails. A page selector that matches by
    meaning embeds the report whole, so an embedding model whose input is
    shorter than the report rejects it.

    An answer that breaks the schema or is cut off is asked for again, up to
    ``retries`` times, with a note on what was wrong. A refusal is final.
    Every attempt is billed, and ``usage`` totals them.

    Raises:
        InvalidRequestError: If ``tenant_id`` or ``device_id`` is blank, if
            ``device_model`` is blank or longer than 200 characters, if
            no prompt has the given version, if ``retries`` is negative, if
            two documents or two service records share an id, if the report
            exceeds ``report_limit``, if it has photos and no
            ``photo_reader`` is given or the photos exceed their limits, or
            if the evidence exceeds ``evidence_limit``.
        ModelCallError: If the photo reader does not return one text per photo.
    """
    check_report_size(report, report_limit)
    if retries < 0:
        message = "retries must not be negative"
        raise InvalidRequestError(message)
    if not tenant_id.strip() or not device_id.strip():
        message = "tenant_id and device_id must not be blank"
        raise InvalidRequestError(message)
    check_device_model(device_model)
    check_photos(report, photo_limit, photo_size_limit, has_reader=photo_reader is not None)
    prompt = instructions(prompt_version)
    device_documents = without_superseded(
        documents.documents_for(tenant_id=tenant_id, device_id=device_id)
    )
    _check_unique("document", [document.id for document in device_documents])
    records = history.records_for(tenant_id=tenant_id, device_id=device_id)
    _check_unique("service record", [record.id for record in records])
    readings = _read_photos(report, photo_reader)
    embedding_tokens = 0
    if pages is not None:
        selection = pages.select(_with_readings(report, readings.texts), device_documents)
        device_documents, embedding_tokens = selection.documents, selection.embedding_tokens
    request = evidence_request(
        prompt,
        report,
        device_documents,
        records,
        device_model=device_model,
        photo_readings=readings.texts,
    )
    check_evidence_size(request, evidence_limit)
    generation, usages = _generate(model, request, retries)
    spent = {
        "usage": total_usage(usages),
        "embedding_tokens": embedding_tokens,
        "attempts": len(usages),
        "photo_readings": tuple(readings.texts),
        "photo_usage": readings.usage,
    }
    if generation.output is None:
        status, summary = _STATUS_WITHOUT_OUTPUT[generation.status]
        failed = _without_causes(status, summary, report, prompt_version)
        return failed.model_copy(update={**spent, "failure": generation.detail})
    figures = unsupported_figures(
        _answer_texts(generation.output),
        (f"{block.label}\n{block.content}" for block in request.untrusted),
    )
    result = _result(generation.output, report, device_documents, records, prompt_version)
    facts = generation.output.report_facts
    restated = {
        "report_facts": facts,
        "unreported_figures": unreported_figures(facts, (report.description, *readings.texts)),
    }
    return result.model_copy(update={"unsupported_figures": figures, **restated, **spent})


def _generate(
    model: StructuredModel, request: GenerationRequest, retries: int
) -> tuple[Generation[DraftTriage], list[Usage | None]]:
    generation = model.generate(request, DraftTriage)
    usages = [generation.usage]
    while generation.status in _RETRY_NOTES and len(usages) <= retries:
        generation = model.generate(_again(request, generation), DraftTriage)
        usages.append(generation.usage)
    return generation, usages


def _again(request: GenerationRequest, rejected: Generation[DraftTriage]) -> GenerationRequest:
    reason = (
        (UntrustedBlock(label="rejected answer", content=rejected.detail),)
        if rejected.detail
        else ()
    )
    return GenerationRequest(
        instructions=request.instructions + _RETRY_NOTES[rejected.status],
        untrusted=(*request.untrusted, *reason),
    )


def _check_unique(kind: str, ids: Sequence[str]) -> None:
    repeated = sorted({item for item in ids if ids.count(item) > 1})
    if repeated:
        message = f"two {kind}s share the id {repeated[0]!r}; a citation could not tell them apart"
        raise InvalidRequestError(message)


def check_report_size(report: FaultReport, limit: int) -> None:
    """Refuse a fault report that is longer than ``limit`` characters.

    Raises:
        InvalidRequestError: If the report exceeds the limit.
    """
    size = len(report.description)
    if size > limit:
        message = (
            f"the fault report holds {size} characters, more than the limit of {limit};"
            " shorten it or raise the limit"
        )
        raise InvalidRequestError(message)


def check_device_model(device_model: str | None) -> None:
    """Refuse a device model that is blank or too long to be a model name.

    Raises:
        InvalidRequestError: If the name is blank or longer than 200 characters.
    """
    if device_model is None:
        return
    if not device_model.strip() or len(device_model) > _LONGEST_DEVICE_MODEL:
        message = (
            f"device_model must not be blank or longer than {_LONGEST_DEVICE_MODEL} characters"
        )
        raise InvalidRequestError(message)


def check_photos(report: FaultReport, limit: int, size_limit: int, *, has_reader: bool) -> None:
    """Refuse photos that nothing would read, or more or larger photos than allowed.

    Raises:
        InvalidRequestError: If the report has photos and there is no reader,
            more than ``limit`` photos, or a photo above ``size_limit`` bytes.
    """
    if not report.photos:
        return
    if not has_reader:
        message = "the fault report has photos, but no photo_reader was given to read them"
        raise InvalidRequestError(message)
    largest = max(len(photo.data) for photo in report.photos)
    if len(report.photos) > limit or largest > size_limit:
        message = (
            f"the fault report has {len(report.photos)} photos, the largest of {largest} bytes;"
            f" the limits are {limit} photos of {size_limit} bytes each"
        )
        raise InvalidRequestError(message)


def _read_photos(report: FaultReport, reader: PhotoReader | None) -> PhotoReadings:
    if reader is None or not report.photos:
        return PhotoReadings(())
    readings = reader.read(report.photos)
    if len(readings.texts) != len(report.photos):
        message = (
            f"the photo reader returned {len(readings.texts)} readings"
            f" for {len(report.photos)} photos"
        )
        raise ModelCallError(message)
    return PhotoReadings(tuple(text[:_LONGEST_READING] for text in readings.texts), readings.usage)


def _with_readings(report: FaultReport, readings: Sequence[str]) -> FaultReport:
    seen = [text for text in readings if text.strip()]
    if not seen:
        return report
    return report.model_copy(update={"description": "\n".join((report.description, *seen))})


def check_evidence_size(request: GenerationRequest, limit: int) -> None:
    """Refuse a request whose evidence is longer than ``limit`` characters.

    Raises:
        InvalidRequestError: If the evidence exceeds the limit.
    """
    size = sum(len(block.label) + len(block.content) for block in request.untrusted)
    if size > limit:
        message = (
            f"the evidence holds {size} characters, more than the limit of {limit};"
            " select fewer pages or raise the limit"
        )
        raise InvalidRequestError(message)


def evidence_request(  # noqa: PLR0913
    prompt: str,
    report: FaultReport,
    documents: Sequence[Document],
    records: Sequence[ServiceRecord],
    *,
    device_model: str | None = None,
    photo_readings: Sequence[str] = (),
) -> GenerationRequest:
    """Build the one model call of a triage run: instructions and untrusted evidence."""
    photos = tuple(
        UntrustedBlock(
            label=f"photo {number} of the fault report, as read by a model", content=text
        )
        for number, text in enumerate(photo_readings, start=1)
        if text.strip()
    )
    device = (
        ()
        if device_model is None
        else (UntrustedBlock(label="device model", content=device_model),)
    )
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
            *photos,
            *device,
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
