import pytest

from machine_fault_triage import InvalidRequestError
from machine_fault_triage.adapters.chonkie import ChonkieSplitter

TEXT = (
    "7.1 Ice in the base tray\n"
    "Ice building up in the base tray means that defrost water cannot drain. "
    "Check that the condensate drain hole is clear.\n"
    "7.2 Unit runs but the flow temperature does not rise\n"
    "Check whether the four-way valve Y1 has switched.\n"
)


def test_recursive_pieces_are_verbatim_parts_that_cover_the_text() -> None:
    splitter = ChonkieSplitter(120)

    pieces = splitter.split(TEXT)

    assert len(pieces) > 1
    assert all(len(piece) <= 120 and piece in TEXT for piece in pieces)
    assert "".join(pieces) == TEXT
    assert splitter.description == "recursive pieces of 120 characters"


def test_fixed_pieces_repeat_the_end_of_the_piece_before() -> None:
    splitter = ChonkieSplitter(100, overlap=20)

    first, second, *_ = splitter.split(TEXT)

    assert first in TEXT
    assert second.startswith(first[-20:])
    assert splitter.description == "fixed pieces of 100 characters, 20 overlapping"


def test_blank_text_has_no_pieces() -> None:
    assert ChonkieSplitter(100).split("  \n ") == []


@pytest.mark.parametrize(("chunk_size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_sizes_that_cannot_work_are_rejected(chunk_size: int, overlap: int) -> None:
    with pytest.raises(InvalidRequestError, match="chunk_size"):
        ChonkieSplitter(chunk_size, overlap=overlap)
