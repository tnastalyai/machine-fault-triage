import pytest
from pydantic import ValidationError

from machine_fault_triage import Generation, GenerationRequest, GenerationStatus, Photo, Usage
from machine_fault_triage.drafts import DraftCause, DraftTriage
from machine_fault_triage.testing import FakeModel, FakeModelExhaustedError, FakePhotoReader
from tests.factories import draft_cause, draft_triage

REQUEST = GenerationRequest(instructions="Diagnose the fault.")


def test_fake_model_returns_scripted_replies_in_order_and_records_requests() -> None:
    first, second = draft_triage([draft_cause()]), draft_triage([])
    model = FakeModel([first, second])

    assert model.generate(REQUEST, DraftTriage).output == first
    assert model.generate(REQUEST, DraftTriage).output == second
    assert model.requests == [REQUEST, REQUEST]


def test_fake_model_can_script_a_refusal() -> None:
    generation = FakeModel([GenerationStatus.REFUSED]).generate(REQUEST, DraftTriage)

    assert generation.status is GenerationStatus.REFUSED
    assert generation.output is None


def test_fake_model_reports_the_usage_it_was_given() -> None:
    usage = Usage(input_tokens=120, output_tokens=30)
    model = FakeModel([draft_triage([]), GenerationStatus.REFUSED, draft_cause()], usage=usage)

    reported = [model.generate(REQUEST, DraftTriage).usage for _ in range(3)]

    assert reported == [usage, usage, usage]


def test_scripted_reply_of_another_type_counts_as_schema_invalid() -> None:
    generation = FakeModel([draft_cause()]).generate(REQUEST, DraftTriage)

    assert generation.status is GenerationStatus.INVALID


def test_fake_model_fails_loudly_when_called_more_often_than_scripted() -> None:
    with pytest.raises(FakeModelExhaustedError):
        FakeModel([]).generate(REQUEST, DraftTriage)


def test_generation_cannot_complete_without_output() -> None:
    with pytest.raises(ValidationError, match="exactly when it completed"):
        Generation[DraftCause](status=GenerationStatus.COMPLETED)


def test_generation_cannot_carry_output_unless_completed() -> None:
    with pytest.raises(ValidationError, match="exactly when it completed"):
        Generation[DraftCause](status=GenerationStatus.REFUSED, output=draft_cause())


def test_fake_photo_reader_hands_out_readings_in_order_and_fails_when_it_runs_out() -> None:
    photo = Photo(media_type="image/png", data=b"pixels")
    reader = FakePhotoReader(["E7", "E8", "E9"], usage=Usage(input_tokens=5, output_tokens=1))

    assert reader.read([photo, photo]).texts == ["E7", "E8"]
    assert reader.read([photo]).usage == Usage(input_tokens=5, output_tokens=1)
    with pytest.raises(FakeModelExhaustedError):
        reader.read([photo])
