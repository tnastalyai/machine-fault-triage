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
    assert "different prompts" not in rendered


def test_comparison_says_when_the_runs_used_different_prompts() -> None:
    current = run_eval(dataset(CASES[:1]), FakeModel([correct_draft()]), model_label="new")
    older = current.model_copy(update={"prompt_version": None, "prompt_digest": None})

    rendered = render_comparison(compare_runs(older, current))

    assert f"the runs used different prompts: unknown and {current.prompt_digest}" in rendered


def test_comparison_says_when_the_runs_selected_pages_differently() -> None:
    limited = run_eval(
        dataset(CASES[:1]),
        FakeModel([correct_draft()]),
        model_label="new",
        page_selection="keyword, at most 5 pages",
    )
    full = limited.model_copy(update={"page_selection": None})

    rendered = render_comparison(compare_runs(full, limited))

    assert "the runs selected pages differently: all and keyword, at most 5 pages" in rendered
    assert "differently" not in render_comparison(compare_runs(limited, limited))


def test_runs_over_different_datasets_are_not_compared() -> None:
    baseline = run_eval(dataset(CASES[:1]), FakeModel([correct_draft()]), model_label="old")
    candidate = run_eval(
        dataset(CASES[:1], digest="digest-2"), FakeModel([correct_draft()]), model_label="new"
    )

    with pytest.raises(IncomparableRunsError):
        compare_runs(baseline, candidate)


def test_comparison_says_when_the_runs_read_photos_differently() -> None:
    baseline = run_eval(dataset([answerable_case()]), FakeModel([correct_draft()]), model_label="x")
    candidate = baseline.model_copy(update={"photo_reader": "reader-2"})

    assert "the runs read photos differently: no reader and reader-2" in render_comparison(
        compare_runs(baseline, candidate)
    )
