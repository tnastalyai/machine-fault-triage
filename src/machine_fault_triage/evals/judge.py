"""A judging model's opinion on whether the cited evidence supports each cause."""

import hashlib
import statistics
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, ConfigDict, Field

from machine_fault_triage.errors import InvalidRequestError, ModelCallError
from machine_fault_triage.evals.cases import Dataset, EvalCase
from machine_fault_triage.evals.report import Judgement, RepeatOutcome, RunReport, Verdict
from machine_fault_triage.evals.scoring import Metric
from machine_fault_triage.grounding import contains_quote
from machine_fault_triage.inputs import Document
from machine_fault_triage.models import Cause, DocumentCitation, TriageResult, Usage, total_usage
from machine_fault_triage.ports import (
    Generation,
    GenerationRequest,
    GenerationStatus,
    StructuredModel,
    UntrustedBlock,
)
from machine_fault_triage.prompts import CAUSE_JUDGING

_WORTH = {Verdict.SUPPORTS: 1.0, Verdict.PARTLY: 0.5, Verdict.DOES_NOT: 0.0}


class _Opinion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str = Field(max_length=1500)
    verdict: Verdict


def judge_digest() -> str:
    """Return a short fingerprint of the instructions given to a judging model."""
    return hashlib.sha256(CAUSE_JUDGING.encode()).hexdigest()[:12]


def judge_run(  # noqa: PLR0913
    report: RunReport,
    dataset: Dataset,
    judge: StructuredModel,
    *,
    judge_label: str,
    context: bool = False,
    workers: int = 1,
) -> RunReport:
    """Return the report with a judging model's opinion on every cause that has verified evidence.

    Verification in code proves that a quote exists; this asks a model whether
    it supports the cause. For each cause with at least one verified citation
    the judge gets the fault report, what was read from its photos, the device
    model, the cause, its verified quotes and the service records it cites,
    all as untrusted data, in one call per cause. A result scores the mean of
    its verdicts as ``quote_supports_cause``: 1 for supports, 0.5 for partly,
    0 for does not. No triage is run again.

    By default the judge sees the quotes alone, as a technician reading the
    result does, so a quote cut out of a fault-code entry without its code is
    judged without it. With ``context`` each quote comes with the paragraph of
    the page it stands in, which asks instead whether the document supports
    the cause. The two answer different questions and are not comparable; the
    report records which was asked.

    ``workers`` is how many calls run at once; the judge must then be safe to
    call from several threads.

    A cause for which the judge refuses, is cut off, answers outside the
    schema or cannot be reached is recorded as unjudged, with that as the
    reason, and left out of the score. A judgement is an opinion, not a
    measurement: it differs between judges and between two calls of one
    judge, so the report names the judge and the fingerprint of its
    instructions.

    Raises:
        InvalidRequestError: If the report was not made on this dataset, or
            ``workers`` is less than one.
        ModelCallError: If there were causes to judge and the judge answered
            for none of them.
    """
    if report.dataset_digest != dataset.digest:
        message = "the report was made on another dataset than the one given"
        raise InvalidRequestError(message)
    if workers < 1:
        message = "workers must be at least 1"
        raise InvalidRequestError(message)
    cases = {case.id: case for case in dataset.cases}
    asked = [
        (repeat, number, _request(case, documents, repeat.result, cause))
        for outcome in report.cases
        for case in [cases[outcome.case_id]]
        for documents in [dataset.documents_for(case) if context else ()]
        for repeat in outcome.repeats
        for number, cause in enumerate(repeat.result.causes, start=1)
        if cause.is_grounded
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(lambda job: _ask(judge, job[2]), asked))
    if answers and all(answer.output is None for answer in answers):
        message = (
            f"the judge answered for none of the {len(answers)} causes;"
            f" the last one: {_judgement(1, answers[-1]).reason}"
        )
        raise ModelCallError(message)
    opinions: dict[int, list[tuple[int, Generation[_Opinion]]]] = {}
    for (repeat, number, _), answer in zip(asked, answers, strict=True):
        opinions.setdefault(id(repeat), []).append((number, answer))
    return report.model_copy(
        update={
            "judge": judge_label,
            "judge_digest": judge_digest(),
            "judge_context": context,
            "cases": tuple(
                outcome.model_copy(
                    update={
                        "repeats": tuple(
                            _judged(repeat, opinions.get(id(repeat), []))
                            for repeat in outcome.repeats
                        )
                    }
                )
                for outcome in report.cases
            ),
        }
    )


def _judged(
    repeat: RepeatOutcome, answers: Sequence[tuple[int, Generation[_Opinion]]]
) -> RepeatOutcome:
    judgements = [_judgement(number, answer) for number, answer in answers]
    usages: list[Usage | None] = [answer.usage for _, answer in answers]
    worth = [_WORTH[judgement.verdict] for judgement in judgements if judgement.verdict]
    scores: dict[Metric, float] = {
        metric: value
        for metric, value in repeat.scores.items()
        if metric is not Metric.QUOTE_SUPPORTS_CAUSE
    }
    if worth:
        scores[Metric.QUOTE_SUPPORTS_CAUSE] = statistics.fmean(worth)
    return repeat.model_copy(
        update={
            "scores": scores,
            "judgements": tuple(judgements),
            "judge_usage": total_usage(usages),
        }
    )


def _judgement(number: int, answer: Generation[_Opinion]) -> Judgement:
    if answer.output is None:
        return Judgement(cause=number, reason=f"no verdict: {answer.detail or answer.status}")
    return Judgement(cause=number, verdict=answer.output.verdict, reason=answer.output.reason)


def _ask(judge: StructuredModel, request: GenerationRequest) -> Generation[_Opinion]:
    try:
        return judge.generate(request, _Opinion)
    except ModelCallError as error:
        return Generation[_Opinion](status=GenerationStatus.INVALID, detail=str(error))


def _request(
    case: EvalCase, documents: Sequence[Document], result: TriageResult, cause: Cause
) -> GenerationRequest:
    records = {record.id: record for record in case.records}
    blocks = [UntrustedBlock(label="fault report", content=case.report.description)]
    blocks.extend(
        UntrustedBlock(label="read from a photo of the report", content=reading)
        for reading in result.photo_readings
    )
    if case.device_model is not None:
        blocks.append(UntrustedBlock(label="device model", content=case.device_model))
    blocks.append(UntrustedBlock(label="proposed cause", content=f"{cause.title}\n{cause.why}"))
    for citation in cause.citations:
        if not citation.verified:
            continue
        if isinstance(citation, DocumentCitation):
            blocks.append(UntrustedBlock(label="quoted passage", content=citation.quote))
            blocks.extend(
                UntrustedBlock(label="paragraph the quote stands in", content=paragraph)
                for paragraph in _paragraph(documents, citation)
            )
        else:
            record = records[citation.record_id]
            parts = ", ".join(record.parts_replaced)
            blocks.append(
                UntrustedBlock(
                    label="service record",
                    content=f"{record.summary}\nParts replaced: {parts or 'none'}",
                )
            )
    return GenerationRequest(instructions=CAUSE_JUDGING, untrusted=tuple(blocks))


def _paragraph(documents: Sequence[Document], citation: DocumentCitation) -> list[str]:
    pages = [
        page
        for document in documents
        if document.id == citation.document_id
        for page in document.pages
        if page.number == citation.page
    ]
    return [
        paragraph
        for page in pages
        for paragraph in page.text.split("\n\n")
        if contains_quote(paragraph, citation.quote)
    ][:1]
