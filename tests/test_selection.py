import pytest

from machine_fault_triage import (
    Document,
    FaultReport,
    HybridPageSelector,
    InvalidRequestError,
    KeywordPageSelector,
    Page,
)
from machine_fault_triage.testing import FakeEmbedder

MANUAL = Document(
    id="manual-1",
    title="Service manual",
    pages=(
        Page(number=1, text="Safety notes. Disconnect the unit from the mains before any work."),
        Page(number=2, text="Start-up sequence. The circulation pump runs for one minute."),
        Page(number=3, text="Message A.11: system pressure outside the set range."),
        Page(number=4, text="Expansion vessel. Check the pre-charge pressure once a year."),
        Page(number=5, text="Cleaning the casing with a damp cloth."),
    ),
)
BULLETIN = Document(
    id="bulletin-7",
    title="Service bulletin",
    pages=(Page(number=1, text="Bus cable routing. Keep the bus cable away from mains cables."),),
)


def report(description: str) -> FaultReport:
    return FaultReport(description=description, language="en")


def selected(limit: int, description: str) -> list[tuple[str, int]]:
    documents, _ = KeywordPageSelector(limit).select(report(description), [MANUAL, BULLETIN])
    return [(document.id, page.number) for document in documents for page in document.pages]


def test_pages_that_share_words_with_the_report_are_kept_in_page_order() -> None:
    assert selected(2, "The pressure keeps dropping, vessel was checked.") == [
        ("manual-1", 3),
        ("manual-1", 4),
    ]


def test_fault_code_is_matched_as_a_whole() -> None:
    assert selected(1, "The display shows A.11 again.") == [("manual-1", 3)]


def test_inflected_word_still_finds_its_page() -> None:
    assert selected(1, "Something about cleaned casings.") == [("manual-1", 5)]


def test_pages_are_ranked_across_all_documents_of_the_device() -> None:
    assert selected(1, "Where should the bus cable go?") == [("bulletin-7", 1)]


def test_report_that_matches_nothing_selects_no_pages() -> None:
    assert selected(3, "Zzz qqq.") == []


def test_device_without_documents_selects_nothing() -> None:
    assert KeywordPageSelector().select(report("The display shows A.11."), []).documents == []


def test_documents_that_fit_within_the_limit_are_sent_whole() -> None:
    assert len(selected(6, "Zzz qqq.")) == len(MANUAL.pages) + len(BULLETIN.pages)


def test_limit_below_one_is_rejected() -> None:
    with pytest.raises(InvalidRequestError):
        KeywordPageSelector(0)


def pages_of(documents: object) -> list[tuple[str, int]]:
    assert isinstance(documents, list)
    return [(document.id, page.number) for document in documents for page in document.pages]


def test_chosen_page_brings_its_neighbours_within_the_limit() -> None:
    selector = KeywordPageSelector(3, neighbours=1)

    documents, _ = selector.select(report("The display shows A.11 again."), [MANUAL, BULLETIN])

    assert pages_of(documents) == [("manual-1", 2), ("manual-1", 3), ("manual-1", 4)]
    assert selector.description == "keyword, at most 3 pages, 1 neighbouring on each side"


def test_neighbours_never_push_the_selection_past_the_limit() -> None:
    documents, _ = KeywordPageSelector(2, neighbours=1).select(
        report("The display shows A.11 again."), [MANUAL, BULLETIN]
    )

    assert len(pages_of(documents)) == 2
    assert ("manual-1", 3) in pages_of(documents)


def test_neighbour_is_the_next_page_number_not_the_next_page_in_the_file() -> None:
    sparse = Document(
        id="sparse",
        title="Excerpts",
        pages=(
            Page(number=3, text="Safety notes."),
            Page(number=8, text="Message A.11: system pressure."),
            Page(number=12, text="Spare parts."),
        ),
    )

    documents, _ = KeywordPageSelector(2, neighbours=1).select(report("A.11 again."), [sparse])

    assert pages_of(documents) == [("sparse", 8)]


def test_page_with_the_same_meaning_is_found_without_a_shared_word() -> None:
    embedder = FakeEmbedder({"dark": [1.0, 0.0], "mains": [0.9, 0.1], "Cleaning": [0.0, 1.0]})
    selector = HybridPageSelector(embedder, 1, embedder_name="fake")

    documents, _ = selector.select(report("The screen stays dark."), [MANUAL, BULLETIN])

    assert pages_of(documents)[0][0] in {"manual-1", "bulletin-7"}
    assert "mains" in documents[0].pages[0].text
    assert selector.description == "hybrid with fake, at most 1 pages"
    assert embedder.queries == ["The screen stays dark."]


def test_exact_fault_code_still_wins_in_a_hybrid_selection() -> None:
    embedder = FakeEmbedder({"A.11": [1.0, 0.0], "pressure": [1.0, 0.0], "Cleaning": [0.0, 1.0]})

    documents, _ = HybridPageSelector(embedder, 1).select(
        report("The display shows A.11 again."), [MANUAL, BULLETIN]
    )

    assert pages_of(documents) == [("manual-1", 3)]


def test_blank_pages_are_not_embedded_and_unknown_text_matches_nothing() -> None:
    blank = Document(id="blank", title="Blank", pages=(Page(number=1, text="  "),))
    embedder = FakeEmbedder({"Cleaning": [0.0, 1.0]})

    documents, _ = HybridPageSelector(embedder, 1).select(report("Zzz qqq."), [MANUAL, blank])

    assert all(text.strip() for call in embedder.page_calls for text in call)
    assert len(pages_of(documents)) == 1


def test_hybrid_selection_sends_small_documents_whole_without_embedding() -> None:
    embedder = FakeEmbedder({})

    selection = HybridPageSelector(embedder, 20).select(report("A.11"), [MANUAL])

    assert selection == ([MANUAL], 0)
    assert embedder.page_calls == []


def test_hybrid_selection_reports_the_tokens_of_the_pages_and_the_report() -> None:
    embedder = FakeEmbedder({"dark": [1.0, 0.0], "mains": [0.9, 0.1]}, tokens_per_text=10)
    written = sum(bool(page.text.strip()) for doc in (MANUAL, BULLETIN) for page in doc.pages)

    selection = HybridPageSelector(embedder, 1).select(
        report("The screen stays dark."), [MANUAL, BULLETIN]
    )

    assert selection.embedding_tokens == 10 * (written + 1)


@pytest.mark.parametrize(("limit", "neighbours"), [(0, 0), (1, -1)])
def test_hybrid_selection_rejects_the_same_bad_limits(limit: int, neighbours: int) -> None:
    with pytest.raises(InvalidRequestError):
        HybridPageSelector(FakeEmbedder({}), limit, neighbours=neighbours)


def test_negative_neighbours_are_rejected() -> None:
    with pytest.raises(InvalidRequestError, match="neighbours"):
        KeywordPageSelector(5, neighbours=-1)


@pytest.mark.parametrize("typed", ["A11", "a11", "A-11", "A.11"])
def test_fault_code_is_found_however_the_customer_typed_it(typed: str) -> None:
    documents, _ = KeywordPageSelector(1).select(
        report(f"Error {typed} again."), [MANUAL, BULLETIN]
    )

    assert pages_of(documents) == [("manual-1", 3)]


class LineSplitter:
    description = "lines"

    def split(self, text: str) -> list[str]:
        return [line for line in text.splitlines() if line.strip()]


def test_rank_lists_the_kept_pages_best_first() -> None:
    ranked = KeywordPageSelector(2).rank(
        report("The display shows A.11 again."), [MANUAL, BULLETIN]
    )

    assert ranked[0] == ("manual-1", 3)
    assert len(ranked) <= 2


def test_page_is_as_good_as_its_best_piece_and_is_kept_whole() -> None:
    long_page = Page(
        number=9,
        text="General notes on storage and transport.\n" * 30 + "Message A.11: system pressure.",
    )
    document = Document(
        id="long",
        title="Long pages",
        pages=(long_page, Page(number=10, text="Storage and transport of spare parts.")),
    )
    selector = KeywordPageSelector(1, splitter=LineSplitter())

    documents, _ = selector.select(report("A.11 again."), [document])

    assert documents[0].pages == (long_page,)
    assert selector.description == "keyword, at most 1 pages, ranked by lines"


def test_page_without_pieces_is_still_ranked_as_a_whole() -> None:
    class NoPieces:
        description = "nothing"

        def split(self, text: str) -> list[str]:
            del text
            return []

    documents, _ = KeywordPageSelector(1, splitter=NoPieces()).select(
        report("The display shows A.11 again."), [MANUAL, BULLETIN]
    )

    assert pages_of(documents) == [("manual-1", 3)]


def test_hybrid_selection_embeds_pieces_when_given_a_splitter() -> None:
    embedder = FakeEmbedder({"dark": [1.0, 0.0], "mains": [0.9, 0.1], "Cleaning": [0.0, 1.0]})
    selector = HybridPageSelector(embedder, 1, embedder_name="fake", splitter=LineSplitter())

    documents, _ = selector.select(report("The screen stays dark."), [MANUAL, BULLETIN])

    assert "mains" in documents[0].pages[0].text
    assert all("\n" not in text for call in embedder.page_calls for text in call)
    assert selector.description == "hybrid with fake, at most 1 pages, ranked by lines"
