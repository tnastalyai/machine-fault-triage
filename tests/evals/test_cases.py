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
        {
            "status": TriageStatus.OK,
            "cause_terms": (("flow",),),
            "problems": ({"cause_terms": (("strainer",),)},),
        },
        {"status": TriageStatus.OK, "cause_terms": (("flow",),), "not_covered": ("rattle",)},
    ],
)
def test_expectation_that_cannot_be_scored_is_rejected(expectation: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Expectation.model_validate(expectation)


def test_values_of_other_models_need_a_case_that_names_its_model(tmp_path: Path) -> None:
    without_model = CASE_TOML.replace("[expect]", '[expect]\nother_models = ["16 l/min"]')
    with_model = 'device_model = "AquaHeat 12 Compact"\n' + without_model

    with pytest.raises(DatasetError, match="must name the device model"):
        load_dataset(write_dataset(tmp_path / "unnamed", case=without_model))

    (case,) = load_dataset(write_dataset(tmp_path / "named", case=with_model)).cases
    assert case.device_model == "AquaHeat 12 Compact"


def test_blank_device_model_is_rejected_when_the_dataset_loads(tmp_path: Path) -> None:
    blank = 'device_model = "  "\n' + CASE_TOML

    with pytest.raises(DatasetError, match="a device model must not be blank"):
        load_dataset(write_dataset(tmp_path / "blank", case=blank))


PHOTO_CASE = CASE_TOML.replace(
    'language = "en"', 'language = "en"\nphotos = ["photos/display.png"]'
)


def test_photos_named_by_a_case_are_read_and_count_towards_the_digest(tmp_path: Path) -> None:
    root = write_dataset(tmp_path, case=PHOTO_CASE)
    (root / "photos").mkdir()
    (root / "photos" / "display.png").write_bytes(b"first picture")

    first = load_dataset(root)
    (root / "photos" / "display.png").write_bytes(b"second picture")

    (photo,) = first.cases[0].report.photos
    assert (photo.media_type, photo.data) == ("image/png", b"first picture")
    assert load_dataset(root).digest != first.digest


@pytest.mark.parametrize(
    "name", ["photos/absent.png", "photos/display.gif", "../outside.png", "photos/empty.png"]
)
def test_photo_that_cannot_be_used_is_rejected_when_the_dataset_loads(
    tmp_path: Path, name: str
) -> None:
    root = write_dataset(tmp_path / "set", case=PHOTO_CASE.replace("photos/display.png", name))
    (root / "photos").mkdir()
    (root / "photos" / "display.gif").write_bytes(b"picture")
    (root / "photos" / "empty.png").write_bytes(b"")
    (tmp_path / "outside.png").write_bytes(b"picture")

    with pytest.raises(DatasetError):
        load_dataset(root)


def test_case_that_says_what_its_photos_show_must_have_photos(tmp_path: Path) -> None:
    case = CASE_TOML.replace("[expect]", '[expect]\nphoto_shows = ["E7"]')

    with pytest.raises(DatasetError, match="must have photos"):
        load_dataset(write_dataset(tmp_path, case=case))
