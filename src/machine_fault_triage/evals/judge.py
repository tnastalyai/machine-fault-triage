"""A judging model's opinion on whether the cited evidence supports each cause."""

import hashlib
import statistics
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from machine_fault_triage.errors import InvalidRequestError, ModelCallError
from machine_fault_triage.evals.cases import Dataset, EvalCase
from machine_fault_triage.evals.report import Judgement, RepeatOutcome, RunReport, Verdict
from machine_fault_triage.evals.scoring import Metric
from machine_fault_triage.inputs import ServiceRecord
from machine_fault_triage.models import Cause, DocumentCitation, TriageResult, total_usage
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

    reason: str = Field(max_length=600)
    verdict: Verdict


def judge_digest() -> str:
    """Return a short fingerprint of the instructions given to a judging model."""
    return hashlib.sha256(CAUSE_JUDGING.encode()).hexdigest()[:12]


def judge_run(
    report: RunReport, dataset: Dataset, judge: StructuredModel, *, judge_label: str
) -> RunReport:
    """Return the report with a judging model's opinion on every cause that has verified evidence.

    Verification in code proves that a quote exists; this asks a model whether
    it supports the cause. For each cause with at least one verified citation
    the judge gets the fault report, what was read from its photos, the device
    model, the cause, its verified quotes and the service records it cites,
    all as untrusted data, in one call per cause. It sees the quotes alone, not
    the pages they stand on, as a technician reading the result does; a quote
    cut out of a fault-code entry without its code is judged without it. A result scores the mean of
    its verdicts as ``quote_supports_cause``: 1 for supports, 0.5 for partly,
    0 for does not. No triage is run again.

    A cause for which the judge refuses, is cut off, answers outside the
    schema or cannot be reached is recorded as unjudged and left out of the
    score. A judgement is an opinion, not a measurement: it differs between
    judges and between two calls of one judge, so the report names the judge
    and the fingerprint of its instructions.

    Raises:
        InvalidRequestError: If the report was not made on this dataset.
    """
    if report.dataset_digest != dataset.digest:
        message = "the report was made on another dataset than the one given"
        raise InvalidRequestError(message)
    cases = {case.id: case for case in dataset.cases}
    return report.model_copy(
        update={
            "judge": judge_label,
            "judge_digest": judge_digest(),
            "cases": tuple(
                outcome.model_copy(
                    update={
                        "repeats": tuple(
                            _judged(repeat, cases[outcome.case_id], judge)
                            for repeat in outcome.repeats
                        )
                    }
                )
                for outcome in report.cases
            ),
        }
    )


def _judged(repeat: RepeatOutcome, case: EvalCase, judge: StructuredModel) -> RepeatOutcome:
    records = {record.id: record for record in case.records}
    judgements = []
    usages = []
    for number, cause in enumerate(repeat.result.causes, start=1):
        if not cause.is_grounded:
            continue
        generation = _ask(judge, _request(case, repeat.result, cause, records))
        opinion = generation.output
        usages.append(generation.usage)
        judgements.append(
            Judgement(cause=number)
            if opinion is None
            else Judgement(cause=number, verdict=opinion.verdict, reason=opinion.reason)
        )
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


def _ask(judge: StructuredModel, request: GenerationRequest) -> Generation[_Opinion]:
    try:
        return judge.generate(request, _Opinion)
    except ModelCallError:
        return Generation[_Opinion](status=GenerationStatus.INVALID)


def _request(
    case: EvalCase, result: TriageResult, cause: Cause, records: Mapping[str, ServiceRecord]
) -> GenerationRequest:
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
