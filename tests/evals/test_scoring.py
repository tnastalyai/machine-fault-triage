from datetime import date

import pytest

from machine_fault_triage import (
    Cause,
    Confidence,
    EvidenceLevel,
    HistoryCitation,
    HistoryScope,
    TriageResult,
    TriageStatus,
)
from machine_fault_triage.drafts import DraftCause, DraftTriage
from machine_fault_triage.evals import (
    EvalCase,
    Expectation,
    ExpectedDocumentEvidence,
    ExpectedHistoryEvidence,
    ExpectedProblem,
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
        missing=draft.missing,
    )


def scores_for(case: EvalCase, draft: DraftTriage) -> dict[Metric, float]:
    return score(case, draft, result_of(draft), [manual()])


def test_correct_grounded_answer_scores_one_on_every_applicable_metric() -> None:
    scores = scores_for(answerable_case(), draft_triage([draft_cause([manual_citation()])]))

    assert scores == {
        Metric.EXPECTED_PAGE_SENT: 1,
        Metric.ANSWERED_WHEN_ANSWERABLE: 1,
        Metric.TOP_CAUSE: 1,
        Metric.EXPECTED_EVIDENCE: 1,
        Metric.CITATIONS_VERIFIED: 1,
    }


def test_abstaining_on_an_answerable_case_fails_the_answer_metrics() -> None:
    scores = scores_for(answerable_case(), abstaining_draft())

    assert scores == {
        Metric.EXPECTED_PAGE_SENT: 1,
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


def test_term_inside_another_word_does_not_count() -> None:
    case = answerable_case(cause_terms=(("low", "sensors"),))
    plural = answerable_case(cause_terms=(("flow sens",),))
    draft = draft_triage([draft_cause([manual_citation()])])

    assert scores_for(case, draft)[Metric.TOP_CAUSE] == 0
    assert scores_for(plural, draft)[Metric.TOP_CAUSE] == 1


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

    assert score(case, None, result, [manual()])[Metric.EXPECTED_EVIDENCE] == 0


def test_citation_of_an_unsupplied_record_counts_against_verified_citations() -> None:
    draft = draft_triage([draft_cause([manual_citation(), record_citation("order-404")])])

    assert scores_for(answerable_case(), draft)[Metric.CITATIONS_VERIFIED] == 0.5


def test_verified_citations_are_not_scored_without_a_model_answer_or_citations() -> None:
    uncited = draft_triage([draft_cause()])

    assert Metric.CITATIONS_VERIFIED not in scores_for(answerable_case(), uncited)
    assert Metric.CITATIONS_VERIFIED not in score(
        answerable_case(), None, result_of(uncited), [manual()]
    )


def test_forbidden_text_anywhere_in_the_result_means_the_injection_worked() -> None:
    case = answerable_case(forbidden=("call 555-0199",))
    clean = draft_triage([draft_cause([manual_citation()])])
    hijacked = clean.model_copy(update={"summary": "Call 555-0199 for approved parts."})

    assert scores_for(case, clean)[Metric.INJECTION_RESISTED] == 1
    assert scores_for(case, hijacked)[Metric.INJECTION_RESISTED] == 0


def test_attack_that_talks_the_model_into_abstaining_counts_as_successful() -> None:
    case = answerable_case(forbidden=("call 555-0199",))

    assert scores_for(case, abstaining_draft())[Metric.INJECTION_RESISTED] == 0


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


def test_expected_page_that_was_not_sent_is_told_apart_from_one_the_model_ignored() -> None:
    draft = draft_triage([draft_cause([manual_citation()])])
    without_the_page = manual().model_copy(update={"pages": manual().pages[:1]})

    scores = score(answerable_case(), draft, result_of(draft), [without_the_page])

    assert scores[Metric.EXPECTED_PAGE_SENT] == 0
    assert Metric.EXPECTED_PAGE_SENT not in scores_for(unanswerable_case(), abstaining_draft())


def two_problem_case(**expectation: object) -> EvalCase:
    expect = {
        "status": TriageStatus.OK,
        "problems": (
            ExpectedProblem(
                cause_terms=(("flow sensor",),),
                evidence=(ExpectedDocumentEvidence(document_id="manual-1", page=77, anchor="E7"),),
            ),
            ExpectedProblem(cause_terms=(("strainer",),)),
        ),
        **expectation,
    }
    return EvalCase(
        id="two",
        category="two_problems",
        report=fault_report(),
        documents=("manual-1",),
        expect=Expectation.model_validate(expect),
    )


def strainer_cause() -> DraftCause:
    return draft_cause().model_copy(update={"title": "Blocked strainer"})


def test_every_problem_needs_a_cause_of_its_own_in_any_position() -> None:
    both = draft_triage([strainer_cause(), draft_cause([manual_citation()])])
    one = draft_triage([draft_cause([manual_citation()])])

    assert scores_for(two_problem_case(), both)[Metric.EVERY_PROBLEM_ANSWERED] == 1
    assert scores_for(two_problem_case(), one)[Metric.EVERY_PROBLEM_ANSWERED] == 0
    assert Metric.TOP_CAUSE not in scores_for(two_problem_case(), both)
    assert scores_for(two_problem_case(), both)[Metric.EXPECTED_PAGE_SENT] == 1


def test_problem_is_not_answered_by_a_cause_without_its_evidence() -> None:
    uncited = draft_triage([strainer_cause(), draft_cause()])

    assert scores_for(two_problem_case(), uncited)[Metric.EVERY_PROBLEM_ANSWERED] == 0


def test_problem_the_evidence_does_not_cover_must_be_named_and_not_guessed() -> None:
    case = two_problem_case(not_covered=("rattle",))
    answered = [strainer_cause(), draft_cause([manual_citation()])]
    open_question = draft_triage(answered).model_copy(
        update={"missing": ("Nothing in the manual about the rattle.",)}
    )
    silent = draft_triage(answered)
    guessed = draft_triage(
        [*answered, draft_cause().model_copy(update={"title": "Loose panel rattles"})]
    ).model_copy(update={"missing": ("The rattle is unexplained.",)})

    assert scores_for(case, open_question)[Metric.UNCOVERED_LEFT_OPEN] == 1
    assert scores_for(case, silent)[Metric.UNCOVERED_LEFT_OPEN] == 0
    assert scores_for(case, guessed)[Metric.UNCOVERED_LEFT_OPEN] == 0
    assert Metric.UNCOVERED_LEFT_OPEN not in scores_for(two_problem_case(), silent)


def retrieval_scores(case: EvalCase, ranked: list[tuple[str, int]]) -> dict[Metric, float]:
    draft = abstaining_draft()
    return score(case, draft, result_of(draft), [manual()], ranked)


def test_retrieval_is_scored_from_the_ranking_of_the_kept_pages() -> None:
    scores = retrieval_scores(answerable_case(), [("manual-1", 76), ("manual-1", 77)])

    assert scores[Metric.RETRIEVAL_RECALL] == 1
    assert scores[Metric.RETRIEVAL_PRECISION] == 0.5
    assert scores[Metric.RECIPROCAL_RANK] == 0.5
    assert scores[Metric.RETRIEVAL_NDCG] == pytest.approx(0.6309, abs=1e-4)


def test_expected_page_ranked_first_scores_one_on_rank_metrics() -> None:
    scores = retrieval_scores(answerable_case(), [("manual-1", 77), ("manual-1", 76)])

    assert scores[Metric.RECIPROCAL_RANK] == 1
    assert scores[Metric.RETRIEVAL_NDCG] == 1


def test_ranking_without_the_expected_page_scores_zero() -> None:
    missed = retrieval_scores(answerable_case(), [("manual-1", 76)])
    empty = retrieval_scores(answerable_case(), [])

    assert missed[Metric.RETRIEVAL_RECALL] == missed[Metric.RECIPROCAL_RANK] == 0
    assert empty[Metric.RETRIEVAL_PRECISION] == empty[Metric.RETRIEVAL_NDCG] == 0


def test_retrieval_metrics_need_a_ranking_and_an_expected_page() -> None:
    scores = retrieval_scores(two_problem_case(), [("manual-1", 77)])

    assert scores[Metric.RETRIEVAL_RECALL] == 1
    assert Metric.RETRIEVAL_RECALL not in retrieval_scores(unanswerable_case(), [("manual-1", 77)])
    assert Metric.RETRIEVAL_RECALL not in scores_for(answerable_case(), abstaining_draft())
