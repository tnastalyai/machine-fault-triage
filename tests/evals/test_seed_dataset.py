from collections import Counter
from pathlib import Path

from machine_fault_triage import TriageStatus
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
    Metric,
    load_dataset,
    run_eval,
    summarize,
)
from machine_fault_triage.models import Confidence
from machine_fault_triage.testing import FakeModel

SEED = Path(__file__).resolve().parents[2] / "datasets" / "seed"


def answer_key(case: EvalCase) -> DraftTriage:
    if case.expect.status is TriageStatus.NOT_ENOUGH_DATA:
        return DraftTriage(summary="Not enough evidence.", enough_evidence=False)
    citations: list[DraftCitation] = [
        DraftDocumentCitation(
            document_id=expected.document_id, page=expected.page, quote=expected.anchor
        )
        if isinstance(expected, ExpectedDocumentEvidence)
        else DraftHistoryCitation(record_id=expected.record_id)
        for expected in case.expect.evidence
    ]
    cause = DraftCause(
        title=" ".join(group[0] for group in case.expect.cause_terms),
        confidence=Confidence.HIGH,
        why="Taken from the expected evidence.",
        citations=tuple(citations),
    )
    return DraftTriage(
        summary="Expected answer.",
        enough_evidence=True,
        causes=(cause,),
        safety_warnings=("Isolate and depressurise first.",) if case.expect.safety_warning else (),
    )


def means(dataset: Dataset, model: FakeModel) -> dict[Metric, float]:
    report = run_eval(dataset, model, model_label="scripted")
    return {summary.metric: summary.mean for summary in summarize(report)}


def test_seed_dataset_covers_both_devices_and_the_planned_categories() -> None:
    dataset = load_dataset(SEED)

    categories = Counter(case.category for case in dataset.cases)
    assert len(dataset.cases) == 15
    assert {case.id.split("-")[0] for case in dataset.cases} == {"hp", "sp"}
    assert {"injection_report", "injection_document", "injection_history"} <= set(categories)
    assert sum(case.expect.status is TriageStatus.NOT_ENOUGH_DATA for case in dataset.cases) == 4


def test_every_answerable_seed_case_names_the_evidence_for_its_answer() -> None:
    dataset = load_dataset(SEED)

    without_evidence = [
        case.id
        for case in dataset.cases
        if case.expect.status is TriageStatus.OK and not case.expect.evidence
    ]
    assert without_evidence == []


def test_answer_key_scores_one_on_every_metric() -> None:
    dataset = load_dataset(SEED)
    model = FakeModel([answer_key(case) for case in dataset.cases])

    assert means(dataset, model) == dict.fromkeys(Metric, 1)


def test_model_that_always_abstains_only_passes_the_abstention_cases() -> None:
    dataset = load_dataset(SEED)
    abstention = DraftTriage(summary="Not enough evidence.", enough_evidence=False)
    model = FakeModel([abstention] * len(dataset.cases))

    scores = means(dataset, model)

    assert scores[Metric.ABSTAINED_WHEN_REQUIRED] == 1
    assert scores[Metric.ANSWERED_WHEN_ANSWERABLE] == 0
    assert scores[Metric.TOP_CAUSE] == 0
    assert scores[Metric.EXPECTED_EVIDENCE] == 0
