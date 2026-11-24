from datetime import date

import pytest
from pydantic import ValidationError

from machine_fault_triage import (
    Cause,
    Confidence,
    DocumentSource,
    EvidenceLevel,
    HistoryScope,
    HistorySource,
    TriageResult,
    TriageStatus,
)


def verified_quote() -> DocumentSource:
    return DocumentSource(
        document_id="manual-1",
        document_title="Service manual",
        page=77,
        quote="E7: flow sensor",
        verified=True,
    )


def unverified_quote() -> DocumentSource:
    return DocumentSource(
        document_id="manual-1", document_title="Service manual", page=77, quote="E7: flow sensor"
    )


def service_record() -> HistorySource:
    return HistorySource(
        record_id="order-9", recorded_on=date(2026, 3, 11), scope=HistoryScope.DEVICE
    )


def grounded_cause() -> Cause:
    return Cause(
        title="Flow sensor fault",
        confidence=Confidence.HIGH,
        why="The manual maps E7 to the flow sensor.",
        sources=(verified_quote(),),
    )


def test_cause_backed_by_a_verified_quote_is_grounded() -> None:
    assert grounded_cause().is_grounded


def test_cause_backed_by_a_service_record_is_grounded() -> None:
    cause = Cause(
        title="Flow sensor fault",
        confidence=Confidence.MEDIUM,
        why="The same fault was repaired before.",
        sources=(service_record(),),
    )

    assert cause.is_grounded


def test_cause_with_only_an_unverified_quote_is_not_grounded() -> None:
    cause = Cause(
        title="Flow sensor fault",
        confidence=Confidence.LOW,
        why="Quoted text could not be found in the manual.",
        sources=(unverified_quote(),),
    )

    assert not cause.is_grounded


@pytest.mark.parametrize("confidence", [Confidence.HIGH, Confidence.MEDIUM])
def test_ungrounded_cause_cannot_claim_more_than_low_confidence(confidence: Confidence) -> None:
    with pytest.raises(ValidationError, match="must have low confidence"):
        Cause(title="Guess", confidence=confidence, why="General knowledge only.")


def test_ok_result_requires_a_cause() -> None:
    with pytest.raises(ValidationError, match="at least one cause"):
        TriageResult(
            status=TriageStatus.OK,
            summary="Nothing found.",
            evidence_level=EvidenceLevel.MANUAL,
            language="en",
        )


@pytest.mark.parametrize(
    "status", [TriageStatus.NOT_ENOUGH_DATA, TriageStatus.REFUSED, TriageStatus.FAILED]
)
def test_result_without_ok_status_carries_no_causes(status: TriageStatus) -> None:
    with pytest.raises(ValidationError, match="only an ok result"):
        TriageResult(
            status=status,
            summary="No diagnosis.",
            evidence_level=EvidenceLevel.MANUAL,
            language="en",
            causes=(grounded_cause(),),
        )


def test_result_with_causes_states_its_evidence_level() -> None:
    with pytest.raises(ValidationError, match="must state its evidence level"):
        TriageResult(
            status=TriageStatus.OK,
            summary="Flow sensor.",
            evidence_level=EvidenceLevel.NONE,
            language="en",
            causes=(grounded_cause(),),
        )


def test_document_source_names_the_document_page_and_quote() -> None:
    source = verified_quote()

    assert (source.document_title, source.page, source.quote) == (
        "Service manual",
        77,
        "E7: flow sensor",
    )


def test_not_enough_data_result_lists_what_is_missing() -> None:
    result = TriageResult(
        status=TriageStatus.NOT_ENOUGH_DATA,
        summary="No manual and no service history for this device.",
        evidence_level=EvidenceLevel.NONE,
        language="en",
        questions_for_customer=("Which code does the display show?",),
        missing=("manual",),
    )

    assert result.causes == ()
    assert result.missing == ("manual",)


def test_result_round_trips_through_json() -> None:
    result = TriageResult(
        status=TriageStatus.OK,
        summary="Flow sensor fault is the most likely cause.",
        evidence_level=EvidenceLevel.MANUAL,
        language="en",
        causes=(
            Cause(
                title="Flow sensor fault",
                confidence=Confidence.HIGH,
                why="Manual and history agree.",
                sources=(verified_quote(), service_record()),
            ),
        ),
    )

    assert TriageResult.model_validate_json(result.model_dump_json()) == result


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        DocumentSource.model_validate(
            {
                "document_id": "manual-1",
                "document_title": "Service manual",
                "page": 1,
                "quote": "text",
                "url": "https://example.test",
            }
        )


def test_result_is_immutable() -> None:
    cause = grounded_cause()

    with pytest.raises(ValidationError):
        cause.title = "Changed"  # type: ignore[misc]
