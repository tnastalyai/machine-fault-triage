import hashlib
from datetime import date

import pytest
from pydantic import BaseModel

from machine_fault_triage import GenerationStatus, InvalidRequestError, ServiceRecord, Usage
from machine_fault_triage.drafts import DraftHistoryCitation
from machine_fault_triage.errors import ModelCallError
from machine_fault_triage.evals import Metric, RunReport, Verdict, judge_run, run_eval
from machine_fault_triage.evals.judge import _Opinion, judge_digest
from machine_fault_triage.evals.report import render_summary
from machine_fault_triage.models import HistoryScope
from machine_fault_triage.ports import Generation, GenerationRequest
from machine_fault_triage.prompts import CAUSE_JUDGING
from machine_fault_triage.testing import FakeModel, FakePhotoReader
from tests.evals.factories import (
    abstaining_draft,
    answerable_case,
    correct_draft,
    dataset,
    unanswerable_case,
)
from tests.evals.test_runner import with_photo
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
        f" (instructions {judge_digest()}) on 2 causes:"
        " 1 supports, 1 partly, 0 does not, 0 left unjudged,"
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
    assert "on 0 causes: 0 supports, 0 partly, 0 does not, 0 left unjudged" in render_summary(
        judged
    )
    assert "tokens, not in the cost" not in render_summary(judged).split("opinion of")[1]


class UnreachableJudge:
    def generate[OutputT: BaseModel](
        self, request: GenerationRequest, output_type: type[OutputT]
    ) -> Generation[OutputT]:
        message = f"no answer to {len(request.untrusted)} blocks as {output_type.__name__}"
        raise ModelCallError(message)


@pytest.mark.parametrize(
    "judge",
    [
        FakeModel([GenerationStatus.REFUSED]),
        FakeModel([GenerationStatus.TRUNCATED]),
        FakeModel([correct_draft()]),
        UnreachableJudge(),
    ],
)
def test_judge_that_gives_no_usable_answer_leaves_the_cause_unjudged(
    judge: FakeModel | UnreachableJudge,
) -> None:
    judged = judge_run(
        finished_run(correct_draft()), dataset([answerable_case()]), judge, judge_label="judge-1"
    )

    (run,) = judged.runs()
    assert Metric.QUOTE_SUPPORTS_CAUSE not in run.scores
    assert [(j.cause, j.verdict) for j in run.judgements] == [(1, None)]
    assert "on 1 causes: 0 supports, 0 partly, 0 does not, 1 left unjudged" in render_summary(
        judged
    )


def test_judging_again_replaces_the_earlier_opinion() -> None:
    cases = dataset([answerable_case()])
    first = judge_run(
        finished_run(correct_draft()),
        cases,
        FakeModel([opinion(Verdict.SUPPORTS)]),
        judge_label="judge-1",
    )

    again = judge_run(first, cases, FakeModel([GenerationStatus.REFUSED]), judge_label="judge-2")

    assert Metric.QUOTE_SUPPORTS_CAUSE not in again.runs()[0].scores
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
