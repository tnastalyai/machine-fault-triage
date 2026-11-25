"""RapidOCR behind the ``OcrEngine`` port. Needs the ``ocr`` extra."""

from collections.abc import Sequence
from importlib.metadata import version
from statistics import median
from typing import Any

import numpy as np
import rapidocr

from machine_fault_triage.ports import PageImage

_LAYOUT_VERSION = 1
_GUTTER_ZONE = (0.35, 0.65)
_GUTTER_STEPS = 200
_MOST_CROSSING = 0.15
_FEWEST_PER_COLUMN = 3
_NARROWEST_COLUMN_LINE = 0.15

type _Span = tuple[float, float]


class RapidOcrEngine:
    """Recognises page text on this machine with RapidOCR and its default models.

    The default models read Latin scripts with their diacritics, Polish and
    German among them. A page set in two columns of running text is read
    column by column: the gutter is looked for in the middle third of the
    page, and a line that crosses it, such as a heading, is kept between the
    lines above and below it. A side made of short pieces, as in a table or
    a table of contents, is not taken for a column. Any other layout comes
    back top to bottom across the page width: the lines of a table cell are
    mixed with those of its neighbours, and three columns interleave. A table
    of two wide text columns is read column by column, which separates each
    cell from the one beside it. Nothing leaves the machine.
    """

    def __init__(self, *, min_score: float = 0.5, engine: Any = None) -> None:  # noqa: ANN401
        """Choose the confidence below which a recognised line is dropped.

        ``engine`` replaces the ``rapidocr.RapidOCR`` instance, for tests.
        """
        self._min_score = float(min_score)
        self._engine: Any = engine or rapidocr.RapidOCR(
            params={"Global.log_level": "error", "Global.text_score": self._min_score}
        )

    @property
    def fingerprint(self) -> str:
        """The package version and the settings that decide what text comes back."""
        return (
            f"rapidocr {version('rapidocr')}, default models,"
            f" min score {self._min_score}, layout {_LAYOUT_VERSION}"
        )

    def recognize(self, image: PageImage) -> str:
        """Return the recognised lines of the page, or an empty string for a blank page."""
        pixels = np.frombuffer(image.rgb, dtype=np.uint8).reshape(image.height, image.width, 3)
        found = self._engine(np.ascontiguousarray(pixels[:, :, ::-1]))
        if found.txts is None:
            return ""
        spans = [
            (float(min(x for x, _ in box)), float(max(x for x, _ in box))) for box in found.boxes
        ]
        return "\n".join(_by_column(list(found.txts), spans, image.width))


def _by_column(lines: Sequence[str], spans: Sequence[_Span], width: int) -> list[str]:
    gutter = _gutter(spans, width)
    if gutter is None:
        return list(lines)
    ordered: list[str] = []
    left: list[str] = []
    right: list[str] = []
    for line, (start, end) in zip(lines, spans, strict=True):
        if end <= gutter:
            left.append(line)
        elif start >= gutter:
            right.append(line)
        else:
            ordered += [*left, *right, line]
            left, right = [], []
    return [*ordered, *left, *right]


def _gutter(spans: Sequence[_Span], width: int) -> float | None:
    first, last = (round(edge * _GUTTER_STEPS) for edge in _GUTTER_ZONE)
    grid = [width * step / _GUTTER_STEPS for step in range(first, last + 1)]
    crossing = [sum(start < x < end for start, end in spans) for x in grid]
    fewest = min(crossing)
    best: tuple[int, int] = (0, 0)
    run = 0
    for index, count in enumerate(crossing):
        run = run + 1 if count == fewest else 0
        best = max(best, (run, index))
    length, end_index = best
    gutter = (grid[end_index - length + 1] + grid[end_index]) / 2
    columns = (
        [end - start for start, end in spans if end <= gutter],
        [end - start for start, end in spans if start >= gutter],
    )
    if fewest > _MOST_CROSSING * len(spans) or min(map(len, columns)) < _FEWEST_PER_COLUMN:
        return None
    if min(map(median, columns)) < _NARROWEST_COLUMN_LINE * width:
        return None
    return gutter
