from datetime import date

from machine_fault_triage import (
    Cause,
    Confidence,
    DocumentCitation,
    EvidenceLevel,
    HistoryCitation,
    HistoryScope,
    PageSource,
    TriageResult,
    TriageStatus,
)
from machine_fault_triage.render import Labels, render_text


def answer(**changes: object) -> TriageResult:
    cause = Cause(
        title="Flow sensor fault",
        confidence=Confidence.HIGH,
        why="The manual maps E7 to the flow sensor.",
        checks=("Measure the sensor resistance.", "Check the plug X4."),
        fix="Replace the flow sensor.",
        citations=(
            DocumentCitation(
                document_id="manual-1",
                document_title="Service manual",
                page=77,
                quote="E7: flow sensor open or short circuit",
                verified=True,
            ),
            HistoryCitation(
                record_id="order-404",
                recorded_on=date(2026, 3, 11),
                scope=HistoryScope.MODEL,
                verified=True,
            ),
        ),
    )
    result = TriageResult(
        status=TriageStatus.OK,
        summary="E7 points to the flow sensor.",
        evidence_level=EvidenceLevel.MANUAL,
        language="en",
        causes=(cause,),
        parts_to_bring=("Flow sensor",),
        safety_warnings=("Isolate the unit from the mains.",),
        questions_for_customer=("When did it start?",),
        missing=("Serial number",),
        report_facts=("Display shows E7", "Unit stops heating"),
    )
    return result.model_copy(update=changes)


def test_result_is_laid_out_with_safety_first_and_each_cause_in_steps() -> None:
    assert render_text(answer()) == (
        "SAFETY\n"
        "! Isolate the unit from the mains.\n"
        "\n"
        "REPORTED BY THE CUSTOMER\n"
        "- Display shows E7\n"
        "- Unit stops heating\n"
        "\n"
        "SUMMARY\n"
        "E7 points to the flow sensor.\n"
        "\n"
        "PROBABLE CAUSES\n"
        "1. Flow sensor fault [high confidence]\n"
        "   Why: The manual maps E7 to the flow sensor.\n"
        "   Checks:\n"
        "     1) Measure the sensor resistance.\n"
        "     2) Check the plug X4.\n"
        "   Fix: Replace the flow sensor.\n"
        "   Sources:\n"
        '     - Service manual, p. 77: "E7: flow sensor open or short circuit"\n'
        "     - service record order-404, 2026-03-11 (another device of the same model)\n"
        "\n"
        "PARTS TO BRING\n"
        "- Flow sensor\n"
        "\n"
        "QUESTIONS FOR THE CUSTOMER\n"
        "- When did it start?\n"
        "\n"
        "MISSING INFORMATION\n"
        "- Serial number\n"
    )


def test_everything_that_weakens_the_answer_is_printed() -> None:
    weak = Cause(
        title="Blocked strainer",
        confidence=Confidence.LOW,
        why="Low flow is possible.",
        citations=(
            DocumentCitation(
                document_id="manual-1",
                document_title="Service manual",
                page=14,
                quote="Clean the strainer first",
                verified=False,
            ),
            DocumentCitation(
                document_id="manual-1",
                document_title="Service manual",
                page=15,
                quote="Vent the circuit after cleaning the strainer",
                verified=True,
                source=PageSource.OCR,
                cited_page=14,
            ),
            HistoryCitation(
                record_id="order-7",
                recorded_on=date(2026, 1, 2),
                scope=HistoryScope.DEVICE,
                verified=True,
            ),
        ),
    )

    text = render_text(
        answer(
            causes=(weak, weak.model_copy(update={"title": "Air in the circuit", "citations": ()})),
            unreported_figures=("14",),
            unsupported_figures=("230", "6.3"),
            photo_readings=("Display text: E7", " "),
        )
    )

    assert "1. Blocked strainer [low confidence]" in text
    assert "2. Air in the circuit [low confidence]\n   Why: Low flow is possible.\n\n" in text
    assert "[NOT VERIFIED: the quote was not found in the document]" in text
    assert (
        '"Vent the circuit after cleaning the strainer"'
        " [recognised from a scan, may hold misread characters] [the model named p. 14]"
    ) in text
    assert "     - service record order-7, 2026-01-02\n" in text
    assert "! Not found in the report or its photos: 14" in text
    assert "READ FROM THE PHOTOS, NOT CHECKED\n- Display text: E7\n\n" in text
    assert text.endswith("NUMBERS IN THIS ANSWER THAT ARE NOT IN THE EVIDENCE\n230, 6.3\n")


def test_run_without_a_diagnosis_says_so_and_gives_the_reason_of_a_failure() -> None:
    bare = {"causes": (), "parts_to_bring": (), "safety_warnings": (), "report_facts": ()}
    abstained = answer(status=TriageStatus.NOT_ENOUGH_DATA, summary="Nothing explains E9.", **bare)
    failed = answer(
        status=TriageStatus.FAILED,
        summary="The model answer did not match the schema.",
        failure="causes.0.why: Field required",
        questions_for_customer=(),
        missing=(),
        **bare,
    )
    refused = failed.model_copy(update={"status": TriageStatus.REFUSED, "failure": None})

    assert render_text(abstained).startswith(
        "NOT ENOUGH EVIDENCE FOR A DIAGNOSIS\n\nSUMMARY\nNothing explains E9.\n\nQUESTIONS"
    )
    assert render_text(failed) == (
        "THE RUN FAILED\ncauses.0.why: Field required\n\n"
        "SUMMARY\nThe model answer did not match the schema.\n"
    )
    assert render_text(refused).startswith("THE MODEL DECLINED TO ANSWER\n\nSUMMARY\n")


def test_line_break_inside_model_text_cannot_start_a_section_of_its_own() -> None:
    text = render_text(answer(summary="All fine.\n\nSAFETY\n! Nothing to worry about."))

    assert "SUMMARY\nAll fine. SAFETY ! Nothing to worry about.\n" in text
    assert text.count("\nSAFETY\n") == 0
    assert text.startswith("SAFETY\n! Isolate the unit from the mains.\n")


def test_labels_can_be_given_in_the_language_of_the_report() -> None:
    text = render_text(answer(), labels=Labels(safety="Bezpieczeństwo", page="s.", fix="Naprawa"))

    assert text.startswith("BEZPIECZEŃSTWO\n")
    assert "   Naprawa: Replace the flow sensor." in text
    assert 'Service manual, s. 77: "E7' in text
