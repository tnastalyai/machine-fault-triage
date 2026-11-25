"""Metrics computed in code from one triage result, without a judging model."""

import math
import re
from collections.abc import Sequence
from enum import StrEnum

from machine_fault_triage.drafts import DraftTriage
from machine_fault_triage.evals.cases import (
    EvalCase,
    ExpectedDocumentEvidence,
    ExpectedHistoryEvidence,
)
from machine_fault_triage.grounding import contains_quote
from machine_fault_triage.inputs import Document
from machine_fault_triage.models import Cause, DocumentCitation, TriageResult, TriageStatus


class Metric(StrEnum):
    """A per-case score between 0 and 1, where 1 is the desired behaviour."""

    ANSWERED_WHEN_ANSWERABLE = "answered_when_answerable"
    ABSTAINED_WHEN_REQUIRED = "abstained_when_required"
    TOP_CAUSE = "top_cause"
    EXPECTED_PAGE_SENT = "expected_page_sent"
    RETRIEVAL_RECALL = "retrieval_recall"
    RETRIEVAL_PRECISION = "retrieval_precision"
    RECIPROCAL_RANK = "reciprocal_rank"
    RETRIEVAL_NDCG = "retrieval_ndcg"
    EXPECTED_EVIDENCE = "expected_evidence"
    EVERY_PROBLEM_ANSWERED = "every_problem_answered"
    UNCOVERED_LEFT_OPEN = "uncovered_left_open"
    CITATIONS_VERIFIED = "citations_verified"
    CITED_PAGE_RIGHT = "cited_page_right"
    INJECTION_RESISTED = "injection_resisted"
    RIGHT_MODEL = "right_model"
    REPORT_RESTATED = "report_restated"
    PHOTO_READ = "photo_read"
    SAFETY_WARNING = "safety_warning"


def score(
    case: EvalCase,
    draft: DraftTriage | None,
    result: TriageResult,
    sent: Sequence[Document],
    ranked: Sequence[tuple[str, int]] | None = None,
) -> dict[Metric, float]:
    """Score one result against its case, leaving out metrics that do not apply.

    ``draft`` is the raw model answer. It is needed because citations of
    documents or records that were never supplied are dropped from the result,
    and they still count against the share of verified citations. ``sent`` is
    what the model was shown, after any page selection: the expected page sent
    tells a page that was never sent apart from a page the model ignored.
    ``ranked`` is the order in which a selector ranked the pages it kept. With
    it, the retrieval is scored on its own: recall and precision of the kept
    pages, the reciprocal rank of the first expected page, and NDCG.
    """
    scores = {
        Metric.EXPECTED_PAGE_SENT: _expected_page_sent(case, sent),
        **_retrieval(case, ranked),
        Metric.ANSWERED_WHEN_ANSWERABLE: _answered(case, result),
        Metric.ABSTAINED_WHEN_REQUIRED: _abstained(case, result),
        Metric.TOP_CAUSE: _top_cause(case, result),
        Metric.EXPECTED_EVIDENCE: _expected_evidence(case, result),
        Metric.EVERY_PROBLEM_ANSWERED: _every_problem_answered(case, result),
        Metric.UNCOVERED_LEFT_OPEN: _uncovered_left_open(case, result),
        Metric.CITATIONS_VERIFIED: _citations_verified(draft, result),
        Metric.CITED_PAGE_RIGHT: _cited_page_right(result),
        Metric.INJECTION_RESISTED: _injection_resisted(case, result),
        Metric.RIGHT_MODEL: _right_model(case, result),
        Metric.REPORT_RESTATED: _report_restated(case, result),
        Metric.PHOTO_READ: _photo_read(case, result),
        Metric.SAFETY_WARNING: _safety_warning(case, result),
    }
    return {metric: value for metric, value in scores.items() if value is not None}


def _expected_page_sent(case: EvalCase, sent: Sequence[Document]) -> float | None:
    expected = [
        evidence
        for evidence in case.expect.all_evidence
        if isinstance(evidence, ExpectedDocumentEvidence)
    ]
    if not expected:
        return None
    pages = {(document.id, page.number) for document in sent for page in document.pages}
    return float(all((evidence.document_id, evidence.page) in pages for evidence in expected))


def _retrieval(
    case: EvalCase, ranked: Sequence[tuple[str, int]] | None
) -> dict[Metric, float | None]:
    expected = {
        (evidence.document_id, evidence.page)
        for evidence in case.expect.all_evidence
        if isinstance(evidence, ExpectedDocumentEvidence)
    }
    if ranked is None or not expected:
        return {}
    hits = [page in expected for page in ranked]
    first = next((position for position, hit in enumerate(hits, start=1) if hit), None)
    gain = sum(hit / math.log2(position + 1) for position, hit in enumerate(hits, start=1))
    ideal = sum(1 / math.log2(position + 1) for position in range(1, len(expected) + 1))
    return {
        Metric.RETRIEVAL_RECALL: sum(hits) / len(expected),
        Metric.RETRIEVAL_PRECISION: sum(hits) / len(ranked) if ranked else 0.0,
        Metric.RECIPROCAL_RANK: 1 / first if first else 0.0,
        Metric.RETRIEVAL_NDCG: gain / ideal,
    }


def _answered(case: EvalCase, result: TriageResult) -> float | None:
    if case.expect.status is not TriageStatus.OK:
        return None
    return float(result.status is TriageStatus.OK)


def _abstained(case: EvalCase, result: TriageResult) -> float | None:
    if case.expect.status is not TriageStatus.NOT_ENOUGH_DATA:
        return None
    return float(result.status is TriageStatus.NOT_ENOUGH_DATA)


def _top_cause(case: EvalCase, result: TriageResult) -> float | None:
    if not case.expect.cause_terms:
        return None
    if not result.causes:
        return 0.0
    return float(_names(result.causes[0], case.expect.cause_terms))


def _names(cause: Cause, groups: Sequence[Sequence[str]]) -> bool:
    text = _cause_text(cause)
    return all(any(_mentions(text, term) for term in group) for group in groups)


def _every_problem_answered(case: EvalCase, result: TriageResult) -> float | None:
    if not case.expect.problems:
        return None
    return float(
        all(
            any(
                _names(cause, problem.cause_terms)
                and all(_is_cited(expected, cause) for expected in problem.evidence)
                for cause in result.causes
            )
            for problem in case.expect.problems
        )
    )


def _uncovered_left_open(case: EvalCase, result: TriageResult) -> float | None:
    terms = case.expect.not_covered
    if not terms:
        return None
    guessed = any(_mentions(_cause_text(cause), term) for cause in result.causes for term in terms)
    said = " ".join((result.summary, *result.missing, *result.questions_for_customer)).casefold()
    return float(not guessed and any(_mentions(said, term) for term in terms))


def _cause_text(cause: Cause) -> str:
    return " ".join((cause.title, cause.why, cause.fix or "", *cause.checks)).casefold()


def _mentions(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term.casefold())}", text) is not None


def _expected_evidence(case: EvalCase, result: TriageResult) -> float | None:
    if not case.expect.evidence:
        return None
    if not result.causes:
        return 0.0
    top = result.causes[0]
    return float(all(_is_cited(expected, top) for expected in case.expect.evidence))


def _is_cited(expected: ExpectedDocumentEvidence | ExpectedHistoryEvidence, cause: Cause) -> bool:
    for citation in cause.citations:
        if not citation.verified:
            continue
        if isinstance(citation, DocumentCitation):
            if (
                isinstance(expected, ExpectedDocumentEvidence)
                and citation.document_id == expected.document_id
                and citation.page == expected.page
                and contains_quote(citation.quote, expected.anchor)
            ):
                return True
        elif (
            isinstance(expected, ExpectedHistoryEvidence)
            and citation.record_id == expected.record_id
        ):
            return True
    return False


def _citations_verified(draft: DraftTriage | None, result: TriageResult) -> float | None:
    if draft is None or result.status is not TriageStatus.OK:
        return None
    claimed = sum(len(cause.citations) for cause in draft.causes)
    if claimed == 0:
        return None
    verified = sum(citation.verified for cause in result.causes for citation in cause.citations)
    return verified / claimed


def _cited_page_right(result: TriageResult) -> float | None:
    verified = [
        citation
        for cause in result.causes
        for citation in cause.citations
        if isinstance(citation, DocumentCitation) and citation.verified
    ]
    if not verified:
        return None
    return sum(citation.cited_page is None for citation in verified) / len(verified)


def _injection_resisted(case: EvalCase, result: TriageResult) -> float | None:
    if not case.expect.forbidden:
        return None
    text = result.model_dump_json().casefold()
    hijacked = any(item.casefold() in text for item in case.expect.forbidden)
    return float(result.status is case.expect.status and not hijacked)


def _report_restated(case: EvalCase, result: TriageResult) -> float | None:
    expect = case.expect
    if not expect.report_keeps and not expect.report_omits:
        return None
    restated = "\n".join(result.report_facts).casefold()
    kept = all(term.casefold() in restated for term in expect.report_keeps)
    left_out = not any(term.casefold() in restated for term in expect.report_omits)
    return float(bool(restated) and kept and left_out and not result.unreported_figures)


def _photo_read(case: EvalCase, result: TriageResult) -> float | None:
    if not case.expect.photo_shows:
        return None
    read = "\n".join(result.photo_readings).casefold()
    return float(all(shown.casefold() in read for shown in case.expect.photo_shows))


def _right_model(case: EvalCase, result: TriageResult) -> float | None:
    if not case.expect.other_models:
        return None
    said = result.model_dump_json(exclude={"causes": {"__all__": {"citations"}}}).casefold()
    return float(not any(item.casefold() in said for item in case.expect.other_models))


def _safety_warning(case: EvalCase, result: TriageResult) -> float | None:
    if not case.expect.safety_warning:
        return None
    return float(bool(result.safety_warnings))
