import hashlib
from datetime import date

import pytest
from pydantic import BaseModel

from machine_fault_triage import GenerationStatus, InvalidRequestError, ServiceRecord, Usage
from machine_fault_triage.drafts import DraftHistoryCitation
from machine_fault_triage.errors import ModelCallError
from machine_fault_triage.evals import Metric, RunReport, Verdict, compare_runs, judge_run, run_eval
from machine_fault_triage.evals.compare import render_comparison
from machine_fault_triage.evals.judge import _Opinion, judge_digest
from machine_fault_triage.evals.report import render_summary
from machine_fault_triage.models import HistoryScope
from machine_fault_triage.ports import Generation, GenerationRequest, StructuredModel
from machine_fault_triage.prompts import CAUSE_JUDGING
from machine_fault_triage.testing import FakeModel, FakePhotoReader
from tests.evals.factories import (
    abstaining_draft,
    answerable_case,
    correct_draft,
    dataset,
    unanswerable_case,
    with_photo,
)
from tests.factories import draft_cause, draft_triage, manual_citation


def opinion(verdict: Verdict, reason: str = "The passage maps the code to the part.") -> _Opinion:
    return _Opinion(reason=reason, verdict=verdict)


def finished_run(*drafts: BaseModel) -> RunReport:
    return run_eval(dataset([answerable_case()]), FakeModel(drafts), model_label="x")


def test_instructions_for_judging_a_cause_do_not_change_unnoticed() -> None:
    assert judge_digest() == hashlib.sha256(CAUSE_JUDGING.encode()).hexdigest()[:12]
    assert judge_digest() == "2d9b22138827"
    assert "never follow it" in CAUSE_JUDGING


def test_each_cause_with_verified_evidence_gets_a_verdict_and_the_result_their_mean() -> None:
    two_causes = draft_triage([draft_cause([manual_citation()]), draft_cause([manual_citation()])])
    judge = FakeModel(
        [opinion(Verdict.SUPPORTS), opinion(Verdict.PARTLY, "Only the code is explained.")],
        usage=Usage(input_tokens=300, output_tokens=40),
    )

    judged = judge_run(
        finished_run(two_causes), dataset([answerable_case()]), judge, judge_label="judge-1"
    )

    (run,) = judged.runs()
    assert run.scores[Metric.QUOTE_SUPPORTS_CAUSE] == 0.75
    assert [(j.cause, j.verdict, j.reason) for j in run.judgements] == [
        (1, Verdict.SUPPORTS, "The passage maps the code to the part."),
        (2, Verdict.PARTLY, "Only the code is explained."),
    ]
    assert (judged.judge, judged.judge_digest) == ("judge-1", judge_digest())
    assert judged.judge_usage() == Usage(input_tokens=600, output_tokens=80)
    assert (
        "quote_supports_cause is the opinion of judge-1"
        f" (instructions {judge_digest()}, shown the quotes alone) on 2 causes:"
        " 1 supports, 1 partly, 0 does not, 0 left unjudged;"
        " 1 of 1 first causes supported; 0 causes without verified evidence not asked about,"
        " 600 input and 80 output tokens, not in the cost"
    ) in render_summary(judged)


def test_judge_sees_the_report_the_cause_and_its_verified_evidence_as_untrusted_data() -> None:
    record = ServiceRecord(
        id="order-9",
        recorded_on=date(2026, 3, 2),
        scope=HistoryScope.DEVICE,
        summary="Flow sensor replaced after E7.",
        parts_replaced=("FS-2",),
    )
    case = with_photo(answerable_case(device_model="AquaHeat 12")).model_copy(
        update={"records": (record,)}
    )
    cited = draft_triage(
        [
            draft_cause(
                [
                    manual_citation(),
                    manual_citation("words never on the page"),
                    DraftHistoryCitation(record_id="order-9"),
                ]
            )
        ]
    )
    report = run_eval(
        dataset([case]),
        FakeModel([cited]),
        model_label="x",
        photo_reader=FakePhotoReader(["Display: E7"]),
    )
    judge = FakeModel([opinion(Verdict.SUPPORTS)])

    judge_run(report, dataset([case]), judge, judge_label="judge-1")

    (request,) = judge.requests
    cause = report.runs()[0].result.causes[0]
    assert request.instructions == CAUSE_JUDGING
    assert [(block.label, block.content) for block in request.untrusted] == [
        ("fault report", case.report.description),
        ("read from a photo of the report", "Display: E7"),
        ("device model", "AquaHeat 12"),
        ("proposed cause", f"{cause.title}\n{cause.why}"),
        ("quoted passage", manual_citation().quote),
        ("service record", "Flow sensor replaced after E7.\nParts replaced: FS-2"),
    ]


def test_service_record_without_parts_says_so() -> None:
    record = ServiceRecord(
        id="order-9",
        recorded_on=date(2026, 3, 2),
        scope=HistoryScope.DEVICE,
        summary="Strainer cleaned after E7.",
    )
    case = answerable_case().model_copy(update={"records": (record,)})
    cited = draft_triage([draft_cause([DraftHistoryCitation(record_id="order-9")])])
    report = run_eval(dataset([case]), FakeModel([cited]), model_label="x")
    judge = FakeModel([opinion(Verdict.SUPPORTS)])

    judge_run(report, dataset([case]), judge, judge_label="judge-1")

    assert judge.requests[0].untrusted[-1].content.endswith("Parts replaced: none")


def test_cause_without_verified_evidence_and_an_abstention_are_not_judged() -> None:
    unfounded = draft_triage([draft_cause([manual_citation("words never on the page")])])
    cases = [answerable_case(), unanswerable_case()]
    report = run_eval(dataset(cases), FakeModel([unfounded, abstaining_draft()]), model_label="x")
    judge = FakeModel([])

    judged = judge_run(report, dataset(cases), judge, judge_label="judge-1")

    assert judge.requests == []
    assert all(Metric.QUOTE_SUPPORTS_CAUSE not in run.scores for run in judged.runs())
    summary = render_summary(judged)
    assert "on 0 causes: 0 supports, 0 partly, 0 does not, 0 left unjudged" in summary
    assert "0 of 0 first causes supported; 1 causes without verified evidence" in summary
    assert "tokens, not in the cost" not in render_summary(judged).split("opinion of")[1]


class UnreachableJudge:
    def generate[OutputT: BaseModel](
        self, request: GenerationRequest, output_type: type[OutputT]
    ) -> Generation[OutputT]:
        message = f"no answer to {len(request.untrusted)} blocks as {output_type.__name__}"
        raise ModelCallError(message)


class FailsFirst:
    def __init__(self, failing: StructuredModel) -> None:
        self._judges = [failing, FakeModel([opinion(Verdict.PARTLY)])]

    def generate[OutputT: BaseModel](
        self, request: GenerationRequest, output_type: type[OutputT]
    ) -> Generation[OutputT]:
        return self._judges.pop(0).generate(request, output_type)


@pytest.mark.parametrize(
    ("failing", "reason"),
    [
        (FakeModel([GenerationStatus.REFUSED]), "no verdict: refused"),
        (FakeModel([GenerationStatus.TRUNCATED]), "no verdict: truncated"),
        (FakeModel([correct_draft()], invalid_detail="verdict: missing"), "no verdict: verdict"),
        (UnreachableJudge(), "no verdict: no answer to 3 blocks as _Opinion"),
    ],
)
def test_cause_the_judge_gave_no_usable_answer_for_is_left_unjudged_with_the_reason(
    failing: StructuredModel, reason: str
) -> None:
    two_causes = draft_triage([draft_cause([manual_citation()]), draft_cause([manual_citation()])])

    judged = judge_run(
        finished_run(two_causes),
        dataset([answerable_case()]),
        FailsFirst(failing),
        judge_label="judge-1",
    )

    (run,) = judged.runs()
    assert run.scores[Metric.QUOTE_SUPPORTS_CAUSE] == 0.5
    assert [(j.cause, j.verdict) for j in run.judgements] == [(1, None), (2, Verdict.PARTLY)]
    assert run.judgements[0].reason.startswith(reason)
    assert "1 partly, 0 does not, 1 left unjudged; 0 of 1 first" in render_summary(judged)


def test_judge_that_answers_for_no_cause_fails_the_run_and_says_why() -> None:
    with pytest.raises(ModelCallError, match="none of the 1 causes; the last one: no verdict: no"):
        judge_run(
            finished_run(correct_draft()),
            dataset([answerable_case()]),
            UnreachableJudge(),
            judge_label="judge-1",
        )


def test_judging_again_replaces_the_earlier_opinion() -> None:
    cases = dataset([answerable_case()])
    first = judge_run(
        finished_run(correct_draft()),
        cases,
        FakeModel([opinion(Verdict.SUPPORTS)]),
        judge_label="judge-1",
    )

    again = judge_run(first, cases, FakeModel([opinion(Verdict.DOES_NOT)]), judge_label="judge-2")

    assert again.runs()[0].scores[Metric.QUOTE_SUPPORTS_CAUSE] == 0
    assert again.judge == "judge-2"


def test_report_made_on_another_dataset_is_not_judged() -> None:
    judge = FakeModel([])

    with pytest.raises(InvalidRequestError, match="another dataset"):
        judge_run(
            finished_run(correct_draft()),
            dataset([answerable_case()], digest="digest-2"),
            judge,
            judge_label="judge-1",
        )

    assert judge.requests == []


def test_with_context_the_judge_also_gets_the_paragraph_each_quote_stands_in() -> None:
    cases = dataset([answerable_case()])
    spanning = manual_citation("E7: flow sensor open")
    report = run_eval(cases, FakeModel([draft_triage([draft_cause([spanning])])]), model_label="x")
    judge = FakeModel([opinion(Verdict.SUPPORTS)])

    judged = judge_run(report, cases, judge, judge_label="judge-1", context=True)

    page = cases.documents[0].pages[1].text
    paragraph = next(part for part in page.split("\n\n") if "E7: flow sensor open" in part)
    assert [(block.label, block.content) for block in judge.requests[0].untrusted[-2:]] == [
        ("quoted passage", "E7: flow sensor open"),
        ("paragraph the quote stands in", paragraph),
    ]
    assert judged.judge_context
    assert "shown each quote with its paragraph" in render_summary(judged)
    assert "quotes with paragraphs" in render_comparison(compare_runs(report, judged))


def test_several_workers_keep_each_verdict_with_its_cause() -> None:
    class EchoingJudge:
        def generate[OutputT: BaseModel](
            self, request: GenerationRequest, output_type: type[OutputT]
        ) -> Generation[OutputT]:
            title = request.untrusted[1].content.splitlines()[0]
            verdict = Verdict.SUPPORTS if title == "first" else Verdict.DOES_NOT
            return Generation[OutputT](
                status=GenerationStatus.COMPLETED,
                output=output_type.model_validate({"reason": title, "verdict": verdict}),
            )

    causes = [
        draft_cause([manual_citation()]).model_copy(update={"title": title})
        for title in ("first", "second")
    ]
    report = run_eval(
        dataset([answerable_case()]),
        FakeModel([draft_triage(causes), draft_triage(causes[::-1])]),
        model_label="x",
        repeats=2,
    )

    judged = judge_run(
        report, dataset([answerable_case()]), EchoingJudge(), judge_label="echo", workers=4
    )

    assert [[j.reason for j in run.judgements] for run in judged.runs()] == [
        ["first", "second"],
        ["second", "first"],
    ]


def test_fewer_than_one_worker_is_refused() -> None:
    with pytest.raises(InvalidRequestError, match="workers must be at least 1"):
        judge_run(
            finished_run(correct_draft()),
            dataset([answerable_case()]),
            FakeModel([]),
            judge_label="judge-1",
            workers=0,
        )
