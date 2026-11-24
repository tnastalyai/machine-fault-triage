from pathlib import Path

import pytest
from pydantic import ValidationError

from machine_fault_triage import TriageStatus
from machine_fault_triage.errors import DatasetError
from machine_fault_triage.evals import Expectation, ExpectedHistoryEvidence, load_dataset
from tests.evals.factories import CASE_TOML, DOCUMENT_TOML, write_dataset


def test_dataset_directory_loads_documents_and_cases(tmp_path: Path) -> None:
    dataset = load_dataset(write_dataset(tmp_path / "tiny"))

    (case,) = dataset.cases
    assert dataset.name == "tiny"
    assert case.report.description == "Display shows E7."
    assert case.records[0].id == "order-9"
    assert [document.id for document in dataset.documents_for(case)] == ["manual-1"]


def test_digest_changes_when_a_case_changes(tmp_path: Path) -> None:
    original = load_dataset(write_dataset(tmp_path / "a"))
    edited = load_dataset(
        write_dataset(tmp_path / "b", case=CASE_TOML.replace("shows E7", "shows E7 again"))
    )
    copy = load_dataset(write_dataset(tmp_path / "c"))

    assert original.digest != edited.digest
    assert original.digest == copy.digest


@pytest.mark.parametrize(
    ("case", "problem"),
    [
        (CASE_TOML.replace('documents = ["manual-1"]', 'documents = ["manual-2"]'), "unknown"),
        (CASE_TOML.replace('documents = ["manual-1"]', "documents = []"), "does not supply"),
        (CASE_TOML.replace("page = 77", "page = 78"), "does not supply"),
        (CASE_TOML.replace('anchor = "E7"', 'anchor = "E9"'), "does not supply"),
        (CASE_TOML.replace('record_id = "order-9"', 'record_id = "order-1"'), "does not supply"),
        (CASE_TOML.replace('status = "ok"', 'status = "okay"'), "status"),
        (CASE_TOML.replace("[report]", "[report"), "e7.toml"),
    ],
)
def test_case_that_contradicts_its_evidence_is_rejected(
    tmp_path: Path, case: str, problem: str
) -> None:
    with pytest.raises(DatasetError, match=problem):
        load_dataset(write_dataset(tmp_path, case=case))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    root = write_dataset(tmp_path)
    (root / "documents" / "copy.toml").write_text(DOCUMENT_TOML, encoding="utf-8")

    with pytest.raises(DatasetError, match="duplicate document ids: manual-1"):
        load_dataset(root)


def test_directory_without_cases_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)


@pytest.mark.parametrize(
    "expectation",
    [
        {"status": TriageStatus.REFUSED},
        {"status": TriageStatus.OK},
        {"status": TriageStatus.OK, "cause_terms": ((),)},
        {"status": TriageStatus.NOT_ENOUGH_DATA, "cause_terms": (("pump",),)},
        {
            "status": TriageStatus.NOT_ENOUGH_DATA,
            "evidence": (ExpectedHistoryEvidence(record_id="order-9"),),
        },
    ],
)
def test_expectation_that_cannot_be_scored_is_rejected(expectation: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Expectation.model_validate(expectation)
