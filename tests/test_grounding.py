import pytest

from machine_fault_triage import (
    Confidence,
    Document,
    DocumentCitation,
    EvidenceLevel,
    HistoryCitation,
    Page,
    PageSource,
)
from machine_fault_triage.drafts import DraftDocumentCitation
from machine_fault_triage.grounding import (
    contains_quote,
    evidence_level,
    ground,
    unreported_figures,
    unsupported_figures,
)
from tests.factories import (
    MANUAL_ID,
    QUOTE,
    draft_cause,
    manual,
    manual_citation,
    record_citation,
    service_record,
)


def test_quote_found_on_the_cited_page_is_verified() -> None:
    (cause,) = ground([draft_cause([manual_citation()])], [manual()], [])

    (citation,) = cause.citations
    assert isinstance(citation, DocumentCitation)
    assert citation.verified
    assert (citation.document_title, citation.page) == ("AquaHeat 12 service manual", 77)
    assert cause.confidence is Confidence.HIGH


@pytest.mark.parametrize(
    "quote",
    ["e7:  FLOW sensor open\nor short circuit", "  E7: flow sensor open or short circuit  "],
)
def test_quote_matches_despite_case_and_whitespace_differences(quote: str) -> None:
    (cause,) = ground([draft_cause([manual_citation(quote=quote)])], [manual()], [])

    assert cause.citations[0].verified


def test_quote_absent_from_the_page_is_kept_unverified_and_lowers_confidence() -> None:
    (cause,) = ground(
        [draft_cause([manual_citation(quote="E7: compressor overload")])], [manual()], []
    )

    assert not cause.citations[0].verified
    assert cause.confidence is Confidence.LOW


def test_quote_found_on_another_page_is_verified_there_and_keeps_the_page_the_model_named() -> None:
    (cause,) = ground([draft_cause([manual_citation(page=76)])], [manual()], [])

    (citation,) = cause.citations
    assert isinstance(citation, DocumentCitation)
    assert (citation.verified, citation.page, citation.cited_page) == (True, 77, 76)
    assert cause.confidence is Confidence.HIGH


def test_quote_on_the_cited_page_names_no_other_page() -> None:
    (cause,) = ground([draft_cause([manual_citation()])], [manual()], [])

    assert getattr(cause.citations[0], "cited_page", 0) is None


def test_quote_citing_a_page_the_document_lacks_is_verified_where_it_stands() -> None:
    (cause,) = ground([draft_cause([manual_citation(page=500)])], [manual()], [])

    (citation,) = cause.citations
    assert isinstance(citation, DocumentCitation)
    assert (citation.verified, citation.page, citation.cited_page) == (True, 77, 500)


def test_short_quote_is_not_looked_for_on_other_pages() -> None:
    (cause,) = ground(
        [draft_cause([manual_citation(quote="E6: low pressure", page=76)])], [manual()], []
    )

    (citation,) = cause.citations
    assert isinstance(citation, DocumentCitation)
    assert (citation.verified, citation.page, citation.cited_page) == (False, 76, None)


def test_quote_repeated_on_several_pages_is_placed_on_the_nearest_and_then_the_earlier() -> None:
    warning = "Isolate the unit from the mains before opening"
    repeated = Document(
        id=MANUAL_ID,
        title="Service manual",
        pages=tuple(
            Page(number=number, text=warning if number in {2, 6, 9} else "Contents")
            for number in range(1, 10)
        ),
    )

    pages = [
        ground([draft_cause([manual_citation(quote=warning, page=cited)])], [repeated], [])[0]
        .citations[0]
        .model_dump()["page"]
        for cited in (1, 4, 8)
    ]

    assert pages == [2, 2, 9]


def test_quote_from_another_document_is_not_moved_to_it() -> None:
    other = Document(id="other-manual", title="Other", pages=(Page(number=1, text=QUOTE),))
    lacking = Document(id=MANUAL_ID, title="Manual", pages=(Page(number=1, text="Contents"),))

    (cause,) = ground([draft_cause([manual_citation(page=1)])], [lacking, other], [])

    assert not cause.citations[0].verified


@pytest.mark.parametrize(
    "citation",
    [
        DraftDocumentCitation(document_id="unknown-manual", page=77, quote="E7"),
        DraftDocumentCitation(document_id="manual-1", page=0, quote="E7"),
        DraftDocumentCitation(document_id="manual-1", page=77, quote="   "),
    ],
)
def test_unusable_document_citation_is_dropped(citation: DraftDocumentCitation) -> None:
    (cause,) = ground([draft_cause([citation])], [manual()], [])

    assert cause.citations == ()
    assert cause.confidence is Confidence.LOW


def test_supplied_service_record_is_verified_with_its_own_date_and_scope() -> None:
    record = service_record()

    (cause,) = ground([draft_cause([record_citation()])], [], [record])

    (citation,) = cause.citations
    assert isinstance(citation, HistoryCitation)
    assert citation.verified
    assert (citation.recorded_on, citation.scope) == (record.recorded_on, record.scope)


def test_invented_service_record_is_dropped() -> None:
    (cause,) = ground([draft_cause([record_citation("order-404")])], [], [service_record()])

    assert cause.citations == ()
    assert cause.confidence is Confidence.LOW


def test_cause_without_citations_becomes_low_confidence_general_knowledge() -> None:
    causes = ground([draft_cause()], [manual()], [service_record()])

    assert causes[0].confidence is Confidence.LOW
    assert evidence_level(causes) is EvidenceLevel.GENERAL_KNOWLEDGE


def test_evidence_level_prefers_a_verified_manual_quote_over_history() -> None:
    causes = ground(
        [draft_cause([record_citation()]), draft_cause([manual_citation()])],
        [manual()],
        [service_record()],
    )

    assert evidence_level(causes) is EvidenceLevel.MANUAL


def test_evidence_level_is_history_when_only_a_service_record_is_verified() -> None:
    causes = ground([draft_cause([record_citation()])], [], [service_record()])

    assert evidence_level(causes) is EvidenceLevel.HISTORY


def test_evidence_level_is_none_without_causes() -> None:
    assert evidence_level(()) is EvidenceLevel.NONE


@pytest.mark.parametrize("quote", ["E7", "flow sensor", "sensor open"])
def test_quote_shorter_than_three_words_proves_nothing(quote: str) -> None:
    (cause,) = ground([draft_cause([manual_citation(quote=quote)])], [manual()], [])

    assert not cause.citations[0].verified
    assert cause.confidence is Confidence.LOW


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        ("Set the pressure to 15 bar.", "5 bar"),
        ("Resistance at 25 degrees: 10,000 ohm", "5 degrees: 10,000 ohm"),
        ("Check the strainers yearly.", "check the strainer"),
        ("Anything at all.", "   "),
    ],
)
def test_quote_cut_out_of_the_middle_of_a_word_or_number_is_not_found(
    text: str, quote: str
) -> None:
    assert not contains_quote(text, quote)


@pytest.mark.parametrize("quote", ["15 bar", "(auto): flow switch", "pressure to 15 bar."])
def test_quote_of_whole_words_is_found_whatever_its_punctuation(quote: str) -> None:
    assert contains_quote("A12 (auto): flow switch open. Set the pressure to 15 bar.", quote)


def test_numbers_missing_from_the_evidence_are_listed_once_in_order() -> None:
    answer = ["Expect about 25.39 kOhm at 5 degrees.", "Tighten to 70 Nm, then 25,39 again."]
    evidence = ["Tightening torque 70 Nm.", "At 5 degrees: 25,388 kOhm. Record job-0312."]

    assert unsupported_figures(answer, evidence) == ("25.39",)


def test_decimal_comma_and_point_name_the_same_number() -> None:
    assert unsupported_figures(["Fill to 1,5 bar."], ["Fill pressure 1.5 bar"]) == ()


def test_single_digits_are_not_checked() -> None:
    assert unsupported_figures(["Check 3 things in step 2."], ["No numbers here."]) == ()


def test_citation_says_when_its_page_was_recognised_from_an_image() -> None:
    printed = manual()
    scanned = printed.model_copy(
        update={
            "pages": tuple(
                page.model_copy(update={"source": PageSource.OCR}) for page in printed.pages
            )
        }
    )

    sources = [
        ground([draft_cause([manual_citation()])], [document], [])[0].citations[0]
        for document in (printed, scanned)
    ]

    assert [(c.verified, getattr(c, "source", None)) for c in sources] == [
        (True, PageSource.TEXT_LAYER),
        (True, PageSource.OCR),
    ]


def test_code_or_number_the_customer_did_not_report_is_listed_once() -> None:
    reported = ["hello, a12 came up and the flow is 23.5 l/min on my AW 90", "Display text: 21 C"]
    facts = [
        "Display shows A-12 and 21 °C",
        "Flow 23,5 l/min, limit 14 l/min",
        "Model AW-90, switch FS1 open for 3 days",
        "Limit 14 l/min again",
    ]

    assert unreported_figures(facts, reported) == ("14", "FS1")
