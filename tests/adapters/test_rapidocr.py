from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from machine_fault_triage.adapters.rapidocr import RapidOcrEngine
from machine_fault_triage.ports import PageImage


class RecordingEngine:
    def __init__(self, lines: Mapping[str, tuple[float, float]]) -> None:
        self._lines = lines
        self.arrays: list[np.ndarray] = []

    def __call__(self, pixels: np.ndarray) -> SimpleNamespace:
        self.arrays.append(pixels)
        if not self._lines:
            return SimpleNamespace(txts=None, boxes=None)
        boxes = [
            [(start, 0), (end, 0), (end, 9), (start, 9)] for start, end in self._lines.values()
        ]
        return SimpleNamespace(txts=tuple(self._lines), boxes=np.array(boxes, dtype=np.float32))


def read(lines: Mapping[str, tuple[float, float]], width: int = 1000) -> list[str]:
    image = PageImage(width=width, height=1, rgb=bytes(3 * width))
    return RapidOcrEngine(engine=RecordingEngine(lines)).recognize(image).splitlines()


def test_recognised_lines_are_joined_one_per_line() -> None:
    lines = {"6 Fault codes": (100, 400), "A12: flow switch open": (100, 900)}

    assert read(lines) == ["6 Fault codes", "A12: flow switch open"]


def test_two_columns_are_read_one_after_the_other_around_a_heading_that_spans_both() -> None:
    lines = {
        "Clean the strainer and": (100, 480),
        "Never open the housing": (520, 900),
        "vent the circuit.": (100, 380),
        "while the unit is live.": (520, 880),
        "then check pump P1.": (100, 420),
        "Wait five minutes.": (520, 800),
        "7 Maintenance of both circuits": (100, 700),
        "Left one": (100, 300),
        "Right one": (520, 700),
        "Left two": (100, 300),
        "Right two": (520, 700),
        "Left three": (100, 300),
        "Right three": (520, 700),
    }

    assert read(lines) == [
        "Clean the strainer and",
        "vent the circuit.",
        "then check pump P1.",
        "Never open the housing",
        "while the unit is live.",
        "Wait five minutes.",
        "7 Maintenance of both circuits",
        "Left one",
        "Left two",
        "Left three",
        "Right one",
        "Right two",
        "Right three",
    ]


def test_table_with_a_narrow_first_column_keeps_its_rows() -> None:
    lines = {
        "A.11": (100, 160),
        "Pressure outside the set range": (300, 900),
        "A.12": (100, 160),
        "Flow switch open": (300, 700),
        "A.13": (100, 160),
        "Sensor fault": (300, 600),
    }

    assert read(lines) == list(lines)


def test_table_split_down_the_middle_keeps_each_code_beside_its_meaning() -> None:
    lines = {
        "C003": (70, 130),
        "Tank temperature": (140, 300),
        "Shown in orange when a fault is active": (520, 900),
        "C004": (70, 130),
        "Flow temperature": (140, 300),
        "Shown in orange above the limit": (520, 900),
        "C005": (70, 130),
        "Return temperature": (140, 300),
        "Shown in orange below the limit": (520, 900),
    }

    assert read(lines) == list(lines)


def test_page_with_too_few_lines_beside_each_other_keeps_its_order() -> None:
    lines = {"Left": (100, 300), "Right": (600, 900), "Below": (100, 300)}

    assert read(lines) == ["Left", "Right", "Below"]


def test_page_on_which_nothing_is_found_gives_an_empty_text() -> None:
    engine = RapidOcrEngine(engine=RecordingEngine({}))

    assert engine.recognize(PageImage(width=1, height=1, rgb=bytes(3))) == ""


def test_pixels_reach_the_engine_as_rows_with_blue_first() -> None:
    recording = RecordingEngine({})
    red_then_blue = bytes([255, 0, 0, 0, 0, 255])

    RapidOcrEngine(engine=recording).recognize(PageImage(width=1, height=2, rgb=red_then_blue))

    (pixels,) = recording.arrays
    assert pixels.tolist() == [[[0, 0, 255]], [[255, 0, 0]]]


def test_fingerprint_changes_with_the_setting_that_changes_the_text() -> None:
    strict = RapidOcrEngine(min_score=0.9, engine=RecordingEngine({})).fingerprint
    default = RapidOcrEngine(engine=RecordingEngine({})).fingerprint

    assert strict != default
    assert default.startswith("rapidocr ")
    assert "min score 0.5" in default


def test_real_engine_reads_a_fault_code_and_a_figure_from_a_drawn_page() -> None:
    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=56)
    draw.text((40, 30), "A12: flow switch open", fill="black", font=font)
    draw.text((40, 140), "Minimum flow 14 l/min", fill="black", font=font)

    text = RapidOcrEngine().recognize(PageImage(image.width, image.height, image.tobytes()))

    assert text.splitlines() == ["A12: flow switch open", "Minimum flow 14 l/min"]
