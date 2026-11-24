import pytest

from machine_fault_triage import GenerationStatus, InvalidRequestError, TriageStatus, Usage
from machine_fault_triage.evals import Metric, Pricing, RunReport, run_eval, summarize
from machine_fault_triage.evals.report import render_summary
from machine_fault_triage.testing import FakeModel
from tests.evals.factories import (
    abstaining_draft,
    answerable_case,
    correct_draft,
    dataset,
    unanswerable_case,
)


def test_every_case_is_run_with_its_own_evidence_and_scored() -> None:
    model = FakeModel([correct_draft(), abstaining_draft()])

    report = run_eval(
        dataset([answerable_case(), unanswerable_case()]), model, model_label="scripted"
    )

    answered, abstained = report.cases
    assert (report.dataset, report.dataset_digest, report.model) == ("tiny", "digest-1", "scripted")
    assert answered.repeats[0].result.status is TriageStatus.OK
    assert answered.repeats[0].scores[Metric.EXPECTED_EVIDENCE] == 1
    assert abstained.repeats[0].scores == {Metric.ABSTAINED_WHEN_REQUIRED: 1}
    assert [len(request.untrusted) for request in model.requests] == [3, 1]


def test_refusal_counts_as_a_missed_answer() -> None:
    report = run_eval(
        dataset([answerable_case()]), FakeModel([GenerationStatus.REFUSED]), model_label="scripted"
    )

    assert report.cases[0].repeats[0].scores[Metric.ANSWERED_WHEN_ANSWERABLE] == 0


def test_summary_reports_mean_spread_and_cases_that_passed_every_repeat() -> None:
    model = FakeModel([correct_draft(), correct_draft(), correct_draft(), abstaining_draft()])

    report = run_eval(
        dataset([answerable_case("steady"), answerable_case("flaky")]),
        model,
        model_label="scripted",
        repeats=2,
    )

    answered = next(
        summary
        for summary in summarize(report)
        if summary.metric is Metric.ANSWERED_WHEN_ANSWERABLE
    )
    assert (answered.cases, answered.mean, answered.passed_every_repeat) == (2, 0.75, 1)
    assert answered.standard_error == pytest.approx(0.25)
    assert Metric.ABSTAINED_WHEN_REQUIRED not in {summary.metric for summary in summarize(report)}


def test_single_case_has_no_standard_error() -> None:
    report = run_eval(dataset([answerable_case()]), FakeModel([correct_draft()]), model_label="x")

    assert {summary.standard_error for summary in summarize(report)} == {None}


def test_reported_tokens_are_totalled_and_priced() -> None:
    model = FakeModel(
        [correct_draft(), GenerationStatus.REFUSED],
        usage=Usage(input_tokens=2_000_000, output_tokens=500_000),
    )

    report = run_eval(
        dataset([answerable_case()]),
        model,
        model_label="scripted",
        repeats=2,
        pricing=Pricing(input=2, output=10),
    )

    assert report.usage() == Usage(input_tokens=4_000_000, output_tokens=1_000_000)
    assert report.cost() == 18
    assert all(run.seconds >= 0 for run in report.runs())
    assert "4000000 input and 1000000 output tokens" in render_summary(report)
    assert "$18.00 in total, $9.0000 per run" in render_summary(report)


def test_cost_is_unknown_without_usage_or_without_pricing() -> None:
    silent = run_eval(
        dataset([answerable_case()]),
        FakeModel([correct_draft()]),
        model_label="x",
        pricing=Pricing(input=2, output=10),
    )
    unpriced = run_eval(
        dataset([answerable_case()]),
        FakeModel([correct_draft()], usage=Usage(input_tokens=10, output_tokens=5)),
        model_label="x",
    )

    assert (silent.usage(), silent.cost()) == (None, None)
    assert unpriced.cost() is None
    assert "tokens" not in render_summary(silent)
    assert "$" not in render_summary(unpriced)


def test_report_round_trips_through_json() -> None:
    report = run_eval(dataset([answerable_case()]), FakeModel([correct_draft()]), model_label="x")

    assert RunReport.model_validate_json(report.model_dump_json()) == report


def test_fewer_than_one_repeat_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        run_eval(dataset([answerable_case()]), FakeModel([]), model_label="x", repeats=0)
