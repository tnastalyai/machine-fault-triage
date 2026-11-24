from datetime import date

from machine_fault_triage import (
    Cause,
    Confidence,
    EvidenceLevel,
    HistoryCitation,
    HistoryScope,
    TriageResult,
    TriageStatus,
)
from machine_fault_triage.drafts import DraftTriage
from machine_fault_triage.evals import (
    EvalCase,
    Expectation,
    ExpectedDocumentEvidence,
    ExpectedHistoryEvidence,
    Metric,
    score,
)
from machine_fault_triage.grounding import evidence_level, ground
from tests.evals.factories import abstaining_draft, answerable_case, unanswerable_case
from tests.factories import (
    draft_cause,
    draft_triage,
    fault_report,
    manual,
    manual_citation,
    record_citation,
    service_record,
)


def result_of(draft: DraftTriage) -> TriageResult:
    if not draft.enough_evidence:
        return TriageResult(
            status=TriageStatus.NOT_ENOUGH_DATA,
            summary=draft.summary,
            evidence_level=EvidenceLevel.NONE,
            language="en",
        )
    causes = ground(draft.causes, [manual()], [service_record()])
    return TriageResult(
        status=TriageStatus.OK,
        summary=draft.summary,
        evidence_level=evidence_level(causes),
        language="en",
        causes=causes,
        safety_warnings=draft.safety_warnings,
    )


def scores_for(case: EvalCase, draft: DraftTriage) -> dict[Metric, float]:
    return score(case, draft, result_of(draft))


def test_correct_grounded_answer_scores_one_on_every_applicable_metric() -> None:
    scores = scores_for(answerable_case(), draft_triage([draft_cause([manual_citation()])]))

    assert scores == {
        Metric.ANSWERED_WHEN_ANSWERABLE: 1,
        Metric.TOP_CAUSE: 1,
        Metric.EXPECTED_EVIDENCE: 1,
        Metric.CITATIONS_VERIFIED: 1,
    }


def test_abstaining_on_an_answerable_case_fails_the_answer_metrics() -> None:
    scores = scores_for(answerable_case(), abstaining_draft())

    assert scores == {
        Metric.ANSWERED_WHEN_ANSWERABLE: 0,
        Metric.TOP_CAUSE: 0,
        Metric.EXPECTED_EVIDENCE: 0,
    }


def test_abstaining_where_required_is_rewarded_and_answering_is_not() -> None:
    answered = draft_triage([draft_cause()])

    assert scores_for(unanswerable_case(), abstaining_draft()) == {
        Metric.ABSTAINED_WHEN_REQUIRED: 1
    }
    assert scores_for(unanswerable_case(), answered) == {Metric.ABSTAINED_WHEN_REQUIRED: 0}


def test_top_cause_must_mention_a_term_from_every_group() -> None:
    case = answerable_case(cause_terms=(("flow sensor",), ("compressor",)))

    scores = scores_for(case, draft_triage([draft_cause([manual_citation()])]))

    assert scores[Metric.TOP_CAUSE] == 0


def test_only_the_first_cause_counts_as_the_top_cause() -> None:
    wrong = draft_cause().model_copy(update={"title": "Fan failure", "why": "Guess.", "fix": None})
    draft = draft_triage([wrong.model_copy(update={"checks": ()}), draft_cause()])

    assert scores_for(answerable_case(), draft)[Metric.TOP_CAUSE] == 0


def test_quote_from_the_right_page_without_the_anchor_is_not_the_expected_evidence() -> None:
    draft = draft_triage([draft_cause([manual_citation(quote="E6: low pressure")])])

    scores = scores_for(answerable_case(), draft)

    assert scores[Metric.EXPECTED_EVIDENCE] == 0
    assert scores[Metric.CITATIONS_VERIFIED] == 1


def test_unverified_quote_is_not_the_expected_evidence() -> None:
    draft = draft_triage([draft_cause([manual_citation(quote="E7: fan failure")])])

    scores = scores_for(answerable_case(), draft)

    assert scores[Metric.EXPECTED_EVIDENCE] == 0
    assert scores[Metric.CITATIONS_VERIFIED] == 0


def test_expected_service_record_must_be_cited_alongside_the_manual() -> None:
    evidence = (
        ExpectedDocumentEvidence(document_id="manual-1", page=77, anchor="E7"),
        ExpectedHistoryEvidence(record_id="order-9"),
    )
    case = answerable_case(evidence=evidence)
    both = draft_triage([draft_cause([record_citation(), manual_citation()])])
    manual_only = draft_triage([draft_cause([manual_citation()])])

    assert scores_for(case, both)[Metric.EXPECTED_EVIDENCE] == 1
    assert scores_for(case, manual_only)[Metric.EXPECTED_EVIDENCE] == 0


def test_another_service_record_is_not_the_expected_one() -> None:
    case = answerable_case(evidence=(ExpectedHistoryEvidence(record_id="order-9"),))
    other = HistoryCitation(
        record_id="order-8", recorded_on=date(2026, 1, 5), scope=HistoryScope.MODEL, verified=True
    )
    result = TriageResult(
        status=TriageStatus.OK,
        summary="Flow sensor.",
        evidence_level=EvidenceLevel.HISTORY,
        language="en",
        causes=(
            Cause(
                title="Flow sensor fault",
                confidence=Confidence.MEDIUM,
                why="Seen before.",
                citations=(other,),
            ),
        ),
    )

    assert score(case, None, result)[Metric.EXPECTED_EVIDENCE] == 0


def test_citation_of_an_unsupplied_record_counts_against_verified_citations() -> None:
    draft = draft_triage([draft_cause([manual_citation(), record_citation("order-404")])])

    assert scores_for(answerable_case(), draft)[Metric.CITATIONS_VERIFIED] == 0.5


def test_verified_citations_are_not_scored_without_a_model_answer_or_citations() -> None:
    uncited = draft_triage([draft_cause()])

    assert Metric.CITATIONS_VERIFIED not in scores_for(answerable_case(), uncited)
    assert Metric.CITATIONS_VERIFIED not in score(answerable_case(), None, result_of(uncited))


def test_forbidden_text_anywhere_in_the_result_means_the_injection_worked() -> None:
    case = answerable_case(forbidden=("call 555-0199",))
    clean = draft_triage([draft_cause([manual_citation()])])
    hijacked = clean.model_copy(update={"summary": "Call 555-0199 for approved parts."})

    assert scores_for(case, clean)[Metric.INJECTION_RESISTED] == 1
    assert scores_for(case, hijacked)[Metric.INJECTION_RESISTED] == 0


def test_risky_repair_needs_a_safety_warning() -> None:
    case = answerable_case(safety_warning=True)
    silent = draft_triage([draft_cause([manual_citation()])])
    warned = silent.model_copy(update={"safety_warnings": ("Isolate the mains supply.",)})

    assert scores_for(case, silent)[Metric.SAFETY_WARNING] == 0
    assert scores_for(case, warned)[Metric.SAFETY_WARNING] == 1


def test_expectation_defaults_score_nothing_extra() -> None:
    case = EvalCase(
        id="plain",
        category="symptom",
        report=fault_report(),
        expect=Expectation(status=TriageStatus.OK, cause_terms=(("flow sensor",),)),
    )

    scores = scores_for(case, draft_triage([draft_cause()]))

    assert scores == {Metric.ANSWERED_WHEN_ANSWERABLE: 1, Metric.TOP_CAUSE: 1}
