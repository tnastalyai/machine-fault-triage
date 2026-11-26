from collections.abc import Sequence
from pathlib import Path

from machine_fault_triage.drafts import DraftTriage
from machine_fault_triage.evals import (
    Dataset,
    EvalCase,
    Expectation,
    ExpectedDocumentEvidence,
    Verdict,
)
from machine_fault_triage.evals.judge import _Opinion
from machine_fault_triage.inputs import Photo
from machine_fault_triage.models import TriageStatus
from machine_fault_triage.testing import FakeEmbedder, FakeModel, FakePhotoReader
from tests.factories import (
    MANUAL_ID,
    draft_cause,
    draft_triage,
    fault_report,
    manual,
    manual_citation,
)

DOCUMENT_TOML = """\
id = "manual-1"
title = "AquaHeat 12 service manual"

[[pages]]
number = 77
text = "7.3 Error codes\\nE7: flow sensor open or short circuit"
"""

CASE_TOML = """\
id = "e7"
category = "code_in_manual"
documents = ["manual-1"]

[report]
description = "Display shows E7."
language = "en"

[[records]]
id = "order-9"
recorded_on = 2026-03-11
scope = "device"
summary = "E7 on display, flow sensor replaced"

[expect]
status = "ok"
cause_terms = [["flow sensor"]]

[[expect.evidence]]
type = "document"
document_id = "manual-1"
page = 77
anchor = "E7"

[[expect.evidence]]
type = "history"
record_id = "order-9"
"""


def write_dataset(root: Path, *, document: str = DOCUMENT_TOML, case: str = CASE_TOML) -> Path:
    (root / "documents").mkdir(parents=True)
    (root / "cases").mkdir()
    (root / "documents" / "manual.toml").write_text(document, encoding="utf-8")
    (root / "cases" / "e7.toml").write_text(case, encoding="utf-8")
    return root


def answerable_case(
    case_id: str = "e7", *, device_model: str | None = None, **expectation: object
) -> EvalCase:
    expect = {
        "status": TriageStatus.OK,
        "cause_terms": (("flow sensor", "flow switch"),),
        "evidence": (ExpectedDocumentEvidence(document_id=MANUAL_ID, page=77, anchor="E7"),),
        **expectation,
    }
    return EvalCase(
        id=case_id,
        category="code_in_manual",
        report=fault_report(),
        device_model=device_model,
        documents=(MANUAL_ID,),
        expect=Expectation.model_validate(expect),
    )


def unanswerable_case(case_id: str = "vague") -> EvalCase:
    return EvalCase(
        id=case_id,
        category="no_basis",
        report=fault_report(),
        expect=Expectation(status=TriageStatus.NOT_ENOUGH_DATA),
    )


def dataset(cases: Sequence[EvalCase], digest: str = "digest-1") -> Dataset:
    return Dataset(name="tiny", digest=digest, documents=(manual(),), cases=tuple(cases))


def correct_draft() -> DraftTriage:
    return draft_triage([draft_cause([manual_citation()])])


def abstaining_draft() -> DraftTriage:
    return DraftTriage(summary="Nothing identifies the fault.", enough_evidence=False)


def correct_model() -> FakeModel:
    return FakeModel([correct_draft()])


def abstaining_model() -> FakeModel:
    return FakeModel([abstaining_draft()])


def configured_embedder(word: str = "sensor") -> FakeEmbedder:
    return FakeEmbedder({word: [1.0, 0.0]})


def configured_model(answer: str) -> FakeModel:
    return correct_model() if answer == "correct" else abstaining_model()


def configured_reader(text: str = "E7") -> FakePhotoReader:
    return FakePhotoReader([text] * 20)


def configured_judge(verdict: str = "supports") -> FakeModel:
    return FakeModel([_Opinion(reason="The passage names the part.", verdict=Verdict(verdict))])


def with_photo(case: EvalCase, **expectation: object) -> EvalCase:
    photo = Photo(media_type="image/png", data=b"pixels")
    return case.model_copy(
        update={
            "report": case.report.model_copy(update={"photos": (photo,)}),
            "expect": case.expect.model_copy(update=expectation),
        }
    )
