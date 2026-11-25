from collections import Counter
from pathlib import Path

from machine_fault_triage import FaultReport, TriageStatus
from machine_fault_triage.drafts import (
    DraftCause,
    DraftCitation,
    DraftDocumentCitation,
    DraftHistoryCitation,
    DraftTriage,
)
from machine_fault_triage.evals import (
    Dataset,
    EvalCase,
    ExpectedDocumentEvidence,
    ExpectedProblem,
    Metric,
    load_dataset,
    run_eval,
    summarize,
)
from machine_fault_triage.models import Confidence
from machine_fault_triage.testing import FakeModel

SEED = Path(__file__).resolve().parents[2] / "datasets" / "seed"


def line_with(dataset: Dataset, expected: ExpectedDocumentEvidence) -> str:
    (document,) = (item for item in dataset.documents if item.id == expected.document_id)
    page = document.page(expected.page)
    assert page is not None
    return next(line for line in page.text.splitlines() if expected.anchor in line)


def expected_cause(problem: ExpectedProblem, dataset: Dataset) -> DraftCause:
    citations: list[DraftCitation] = [
        DraftDocumentCitation(
            document_id=expected.document_id,
            page=expected.page,
            quote=line_with(dataset, expected),
        )
        if isinstance(expected, ExpectedDocumentEvidence)
        else DraftHistoryCitation(record_id=expected.record_id)
        for expected in problem.evidence
    ]
    return DraftCause(
        title=" ".join(group[0] for group in problem.cause_terms),
        confidence=Confidence.HIGH,
        why="Taken from the expected evidence.",
        citations=tuple(citations),
    )


def answer_key(case: EvalCase, dataset: Dataset) -> DraftTriage:
    if case.expect.status is TriageStatus.NOT_ENOUGH_DATA:
        return DraftTriage(summary="Not enough evidence.", enough_evidence=False)
    expected_causes = case.expect.problems or (
        ExpectedProblem(cause_terms=case.expect.cause_terms, evidence=case.expect.evidence),
    )
    return DraftTriage(
        summary="Expected answer.",
        enough_evidence=True,
        causes=tuple(expected_cause(problem, dataset) for problem in expected_causes),
        missing=case.expect.not_covered[:1],
        safety_warnings=("Isolate and depressurise first.",) if case.expect.safety_warning else (),
    )


def means(dataset: Dataset, model: FakeModel) -> dict[Metric, float]:
    report = run_eval(dataset, model, model_label="scripted")
    return {summary.metric: summary.mean for summary in summarize(report)}


def test_seed_dataset_covers_both_devices_and_the_planned_categories() -> None:
    dataset = load_dataset(SEED)

    categories = Counter(case.category for case in dataset.cases)
    assert len(dataset.cases) == 26
    assert categories["two_problems"] == 2
    assert categories["model_variant"] == categories["model_variant_table"] == 4
    assert {case.id.split("-")[0] for case in dataset.cases} == {"hp", "sp"}
    assert {"injection_report", "injection_document", "injection_history"} <= set(categories)
    assert sum(case.expect.status is TriageStatus.NOT_ENOUGH_DATA for case in dataset.cases) == 5


def test_every_answerable_seed_case_names_the_evidence_for_its_answer() -> None:
    dataset = load_dataset(SEED)

    without_evidence = [
        case.id
        for case in dataset.cases
        if case.expect.status is TriageStatus.OK and not case.expect.all_evidence
    ]
    assert without_evidence == []


def test_answer_key_scores_one_on_every_metric() -> None:
    dataset = load_dataset(SEED)
    model = FakeModel([answer_key(case, dataset) for case in dataset.cases])

    retrieval = {
        Metric.RETRIEVAL_RECALL,
        Metric.RETRIEVAL_PRECISION,
        Metric.RECIPROCAL_RANK,
        Metric.RETRIEVAL_NDCG,
    }
    assert means(dataset, model) == dict.fromkeys(set(Metric) - retrieval, 1)


def test_model_that_always_abstains_only_passes_the_abstention_cases() -> None:
    dataset = load_dataset(SEED)
    abstention = DraftTriage(summary="Not enough evidence.", enough_evidence=False)
    model = FakeModel([abstention] * len(dataset.cases))

    scores = means(dataset, model)

    assert scores[Metric.ABSTAINED_WHEN_REQUIRED] == 1
    assert scores[Metric.ANSWERED_WHEN_ANSWERABLE] == 0
    assert scores[Metric.TOP_CAUSE] == 0
    assert scores[Metric.EXPECTED_EVIDENCE] == 0
    assert scores[Metric.INJECTION_RESISTED] == 0
    assert scores[Metric.EVERY_PROBLEM_ANSWERED] == 0
    assert scores[Metric.UNCOVERED_LEFT_OPEN] == 0


def test_model_variant_cases_come_in_pairs_that_differ_only_in_the_model() -> None:
    dataset = load_dataset(SEED)
    variants = [case for case in dataset.cases if case.category.startswith("model_variant")]

    models: dict[FaultReport, set[str | None]] = {}
    for case in variants:
        models.setdefault(case.report, set()).add(case.device_model)
    assert len(variants) == 2 * len(models)
    assert all(len(pair) == 2 for pair in models.values())
    assert all(case.expect.other_models for case in variants)
