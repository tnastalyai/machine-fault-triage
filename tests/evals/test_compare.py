import pytest

from machine_fault_triage.errors import IncomparableRunsError
from machine_fault_triage.evals import Metric, compare_runs, run_eval
from machine_fault_triage.evals.compare import render_comparison
from machine_fault_triage.testing import FakeModel
from tests.evals.factories import (
    abstaining_draft,
    answerable_case,
    correct_draft,
    dataset,
    unanswerable_case,
)

CASES = [answerable_case("a"), answerable_case("b"), unanswerable_case("c")]


def test_comparison_pairs_cases_and_names_the_ones_that_moved() -> None:
    baseline = run_eval(
        dataset(CASES),
        FakeModel([correct_draft(), abstaining_draft(), abstaining_draft()]),
        model_label="old",
    )
    candidate = run_eval(
        dataset(CASES),
        FakeModel([abstaining_draft(), correct_draft(), abstaining_draft()]),
        model_label="new",
    )

    comparison = compare_runs(baseline, candidate)

    deltas = {delta.metric: delta for delta in comparison.deltas}
    answered = deltas[Metric.ANSWERED_WHEN_ANSWERABLE]
    assert (answered.cases, answered.baseline, answered.candidate) == (2, 0.5, 0.5)
    assert answered.difference == 0
    assert answered.standard_error == pytest.approx(1)
    assert (answered.improved, answered.regressed) == (("b",), ("a",))
    assert deltas[Metric.ABSTAINED_WHEN_REQUIRED].standard_error is None
    assert Metric.CITATIONS_VERIFIED not in deltas
    rendered = render_comparison(comparison)
    assert "baseline old  candidate new" in rendered
    assert "improved  answered_when_answerable: b" in rendered
    assert "regressed answered_when_answerable: a" in rendered


def test_runs_over_different_datasets_are_not_compared() -> None:
    baseline = run_eval(dataset(CASES[:1]), FakeModel([correct_draft()]), model_label="old")
    candidate = run_eval(
        dataset(CASES[:1], digest="digest-2"), FakeModel([correct_draft()]), model_label="new"
    )

    with pytest.raises(IncomparableRunsError):
        compare_runs(baseline, candidate)
