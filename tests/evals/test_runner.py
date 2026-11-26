from collections.abc import Sequence
from pathlib import Path

import pytest

from machine_fault_triage import (
    CachedEmbedder,
    Document,
    FaultReport,
    GenerationStatus,
    HybridPageSelector,
    InvalidRequestError,
    KeywordPageSelector,
    Selection,
    TriageStatus,
    Usage,
)
from machine_fault_triage.evals import Metric, Pricing, RunReport, run_eval, summarize
from machine_fault_triage.evals.report import render_summary
from machine_fault_triage.prompts import instructions, prompt_digest
from machine_fault_triage.testing import FakeEmbedder, FakeModel, FakePhotoReader
from tests.evals.factories import (
    abstaining_draft,
    answerable_case,
    correct_draft,
    dataset,
    unanswerable_case,
    with_photo,
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


def test_stored_page_vectors_cost_no_embedding_tokens_on_a_later_run(tmp_path: Path) -> None:
    embedder = CachedEmbedder(
        FakeEmbedder({"E7": [1.0, 0.0]}, tokens_per_text=5), tmp_path / "v.sqlite", model="m1"
    )
    tiny = dataset([answerable_case()])
    written = sum(bool(page.text.strip()) for doc in tiny.documents for page in doc.pages)

    first, later = (
        run_eval(tiny, FakeModel([correct_draft()]), model_label="x", pages=_OnlySelect(embedder))
        for _ in range(2)
    )

    assert written > 1
    assert first.embedding_tokens() == 5 * (written + 1)
    assert later.embedding_tokens() == 5


class _OnlySelect:
    def __init__(self, embedder: CachedEmbedder) -> None:
        self._selector = HybridPageSelector(embedder, 1)

    def select(self, report: FaultReport, documents: Sequence[Document]) -> Selection:
        return self._selector.select(report, documents)


class FirstPages:
    def select(self, report: FaultReport, documents: Sequence[Document]) -> Selection:
        del report
        first = [
            document.model_copy(update={"pages": document.pages[:1]}) for document in documents
        ]
        return Selection(first, 1_000_000)


def test_cached_and_embedding_tokens_are_priced_at_their_own_rates() -> None:
    usage = Usage(input_tokens=3_000_000, output_tokens=1_000_000, cached_input_tokens=2_000_000)

    report = run_eval(
        dataset([answerable_case()]),
        FakeModel([correct_draft()], usage=usage),
        model_label="x",
        pages=FirstPages(),
        pricing=Pricing(input=2, output=10, cached_input=0.5, embedding=0.1),
    )
    one_price = report.model_copy(update={"pricing": Pricing(input=2, output=10)})

    assert report.embedding_tokens() == 1_000_000
    assert report.cost() == pytest.approx(13.1)
    assert one_price.cost() == 16
    assert "2000000 of the input read from cache" in render_summary(report)
    assert "1000000 embedding tokens, $" in render_summary(report)
    assert "1000000 embedding tokens, not in the cost" in render_summary(one_price)


def test_report_round_trips_through_json() -> None:
    report = run_eval(dataset([answerable_case()]), FakeModel([correct_draft()]), model_label="x")

    assert RunReport.model_validate_json(report.model_dump_json()) == report


def test_summary_names_the_prompt_the_run_used() -> None:
    report = run_eval(dataset([answerable_case()]), FakeModel([correct_draft()]), model_label="x")
    older = report.model_copy(update={"prompt_version": None, "prompt_digest": None})

    assert f"prompt v{report.prompt_version} ({report.prompt_digest})" in render_summary(report)
    assert "prompt unknown" in render_summary(older)


def test_page_selection_is_applied_to_every_case_and_recorded() -> None:
    model = FakeModel([correct_draft()])
    pages = KeywordPageSelector(1)

    report = run_eval(
        dataset([answerable_case()]),
        model,
        model_label="x",
        pages=pages,
        page_selection=pages.description,
    )

    (request,) = model.requests
    assert sum("page=" in block.label for block in request.untrusted) == 1
    assert report.page_selection == "keyword, at most 1 pages"
    assert "pages: keyword, at most 1 pages" in render_summary(report)
    assert "pages: all" in render_summary(report.model_copy(update={"page_selection": None}))


def test_older_prompt_version_is_run_and_recorded() -> None:
    model = FakeModel([correct_draft()])

    report = run_eval(dataset([answerable_case()]), model, model_label="x", prompt_version=1)

    assert model.requests[0].instructions == instructions(1)
    assert (report.prompt_version, report.prompt_digest) == (1, prompt_digest(1))


def test_case_over_the_evidence_limit_stops_the_run_before_any_model_call() -> None:
    model = FakeModel([correct_draft()])

    with pytest.raises(InvalidRequestError, match="more than the limit of 50"):
        run_eval(
            dataset([unanswerable_case(), answerable_case()]),
            model,
            model_label="x",
            evidence_limit=50,
        )

    assert model.requests == []


def test_report_over_the_limit_stops_the_run_before_any_model_call() -> None:
    model = FakeModel([correct_draft()])

    with pytest.raises(InvalidRequestError, match="more than the limit of 5; shorten it"):
        run_eval(
            dataset([unanswerable_case(), answerable_case()]),
            model,
            model_label="x",
            pages=KeywordPageSelector(1),
            report_limit=5,
        )

    assert model.requests == []


@pytest.mark.parametrize(("repeats", "retries"), [(0, 1), (1, -1)])
def test_fewer_than_one_repeat_or_negative_retries_are_rejected(repeats: int, retries: int) -> None:
    with pytest.raises(InvalidRequestError):
        run_eval(
            dataset([answerable_case()]),
            FakeModel([]),
            model_label="x",
            repeats=repeats,
            retries=retries,
        )


def test_summary_counts_the_runs_that_asked_the_model_again() -> None:
    replies = (GenerationStatus.TRUNCATED, correct_draft(), correct_draft(), correct_draft())

    report = run_eval(
        dataset([answerable_case()]), FakeModel(replies), model_label="x", repeats=3, retries=1
    )
    patient = run_eval(
        dataset([answerable_case()]), FakeModel([correct_draft()]), model_label="x", retries=0
    )

    assert [run.result.attempts for run in report.runs()] == [2, 1, 1]
    assert "1 of 3 runs asked the model again" in render_summary(report)
    assert "asked the model again" not in render_summary(patient)


def test_device_model_of_a_case_is_sent_and_counted_in_the_check_up_front() -> None:
    case = answerable_case(device_model="AquaHeat 12 Compact")
    model = FakeModel([correct_draft()])

    run_eval(dataset([case]), model, model_label="x")

    (request,) = model.requests
    assert request.untrusted[1].content == "AquaHeat 12 Compact"
    sent = sum(len(block.label) + len(block.content) for block in request.untrusted)
    with pytest.raises(InvalidRequestError, match=f"holds {sent} characters"):
        run_eval(dataset([case]), FakeModel([]), model_label="x", evidence_limit=sent - 1)


def test_case_with_photos_and_no_reader_stops_the_run_before_any_model_call() -> None:
    model = FakeModel([correct_draft()])

    with pytest.raises(InvalidRequestError, match="no photo_reader was given"):
        run_eval(dataset([with_photo(answerable_case())]), model, model_label="x")

    assert model.requests == []


def test_photos_of_a_case_are_read_scored_and_their_tokens_reported() -> None:
    case = with_photo(answerable_case(), photo_shows=("E7",))
    reader = FakePhotoReader(
        ["Display text: E7", "Display text: 37"], usage=Usage(input_tokens=400, output_tokens=30)
    )

    report = run_eval(
        dataset([case]),
        FakeModel([correct_draft(), correct_draft()]),
        model_label="x",
        repeats=2,
        photo_reader=reader,
        photo_reader_label="scripted reader",
    )

    first, second = report.runs()
    assert (first.scores[Metric.PHOTO_READ], second.scores[Metric.PHOTO_READ]) == (1, 0)
    assert report.photo_reader == "scripted reader"
    assert report.photo_usage() == Usage(input_tokens=800, output_tokens=60)
    assert (
        "800 input and 60 output tokens for reading photos with scripted reader, not in the cost"
        in render_summary(report)
    )


def test_run_without_a_reader_names_none_in_the_report() -> None:
    report = run_eval(
        dataset([answerable_case()]),
        FakeModel([correct_draft()]),
        model_label="x",
        photo_reader_label="ignored",
    )

    assert (report.photo_reader, report.photo_usage()) == (None, None)


def test_code_seen_only_on_a_photo_decides_which_pages_are_sent_and_ranked() -> None:
    shown = with_photo(answerable_case())
    case = shown.model_copy(
        update={
            "report": shown.report.model_copy(
                update={"description": "See the photo of the display."}
            )
        }
    )

    report = run_eval(
        dataset([case]),
        FakeModel([correct_draft()]),
        model_label="x",
        pages=KeywordPageSelector(1),
        photo_reader=FakePhotoReader(["E7: flow sensor open or short circuit"]),
    )

    (run,) = report.runs()
    assert run.scores[Metric.EXPECTED_PAGE_SENT] == 1
    assert run.scores[Metric.RECIPROCAL_RANK] == 1
