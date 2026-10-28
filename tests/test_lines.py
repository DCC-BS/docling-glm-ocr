"""Tests for splitting a crop's text into line cells placed on the crop's ink."""

from __future__ import annotations

import pytest
from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image, ImageDraw

from docling_glm_ocr.lines import split_into_lines

# Page rect of the crop: 200 x 100 page units, rendered at 2 px per unit
RECT = BoundingBox(l=100, t=50, r=300, b=150, coord_origin=CoordOrigin.TOPLEFT)


def _page(blocks: list[tuple[int, int, int, int]], size: tuple[int, int] = (400, 200)) -> Image.Image:
    """White crop with black boxes standing in for words, given in pixels (l, t, r, b)."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for box in blocks:
        draw.rectangle(box, fill="black")
    return image


def _boxes(cells: list[tuple[str, BoundingBox]]) -> list[tuple[str, tuple[float, float, float, float]]]:
    return [(text, (round(b.l, 1), round(b.t, 1), round(b.r, 1), round(b.b, 1))) for text, b in cells]


def test_one_cell_per_line_on_its_ink():
    image = _page([(20, 20, 300, 40), (20, 80, 200, 100)])

    cells = split_into_lines("Erste Zeile\nZweite", image, RECT)

    assert _boxes(cells) == [
        ("Erste Zeile", (110.0, 60.0, 250.5, 70.5)),
        ("Zweite", (110.0, 90.0, 200.5, 100.5)),
    ]


def test_tight_box_leaves_margin_of_region_free():
    # The region reaches into the next line of PDF text; the cell must stop at the ink
    image = _page([(20, 20, 300, 60), (20, 196, 300, 199)])

    cells = split_into_lines("Jahresrückblick 2025", image, RECT)

    assert _boxes(cells) == [("Jahresrückblick 2025", (110.0, 60.0, 250.5, 80.5))]


def test_table_row_cells_split_at_column_gaps():
    image = _page([(10, 20, 60, 40), (150, 20, 220, 40), (330, 20, 390, 40)])

    cells = split_into_lines("Datum\tLeistung\tBetrag", image, RECT)

    assert [text for text, _ in cells] == ["Datum", "Leistung", "Betrag"]
    assert [round(b.l) for _, b in cells] == [105, 175, 265]


def test_table_row_falls_back_to_one_cell_when_columns_do_not_match():
    image = _page([(10, 20, 390, 40)])

    cells = split_into_lines("Datum\tLeistung\tBetrag", image, RECT)

    assert [text for text, _ in cells] == ["Datum Leistung Betrag"]


def test_cells_spread_over_fewer_columns():
    # Two header cells almost touch: three cells, two columns
    image = _page([(10, 20, 60, 40), (250, 20, 318, 40), (322, 20, 390, 40)])

    cells = split_into_lines("Datum\tSelbstbehalt\tVergütung", image, RECT)

    assert [text for text, _ in cells] == ["Datum", "Selbstbehalt Vergütung"]


def test_row_without_tabs_split_by_words_at_column_gaps():
    image = _page([(10, 20, 60, 40), (150, 20, 260, 40), (330, 20, 390, 40)])

    cells = split_into_lines("08.01.2026 Praxis Dr. Beispiel 142.30", image, RECT)

    assert [text for text, _ in cells] == ["08.01.2026", "Praxis Dr. Beispiel", "142.30"]


def test_table_rules_do_not_hide_column_gaps():
    # A ruled table row: lines above, below and between the cells
    image = _page([
        *[(10, 20, 60, 40), (150, 20, 260, 40), (330, 20, 390, 40)],
        *[(0, 10, 399, 11), (0, 48, 399, 49), (100, 0, 101, 199), (300, 0, 301, 199)],
    ])

    cells = split_into_lines("Datum\tLeistung\tBetrag", image, RECT)

    assert [text for text, _ in cells] == ["Datum", "Leistung", "Betrag"]
    assert [round(b.l) for _, b in cells] == [105, 175, 265]
    assert [(round(b.t), round(b.b)) for _, b in cells] == [(60, 70)] * 3


@pytest.mark.parametrize("angle", [-1.2, 0.8])
def test_skewed_scan_rows_and_columns_found(angle):
    rows = 8
    blocks = [
        box
        for i in range(rows)
        for box in (
            (40, 30 + i * 18, 200, 40 + i * 18),
            (400, 30 + i * 18, 560, 40 + i * 18),
            (760, 30 + i * 18, 900, 40 + i * 18),
        )
    ]
    image = _page(blocks, size=(960, 200)).rotate(angle, fillcolor="white")
    text = "\n".join(f"a{i}\tb{i}\tc{i}" for i in range(rows))

    cells = split_into_lines(text, image, RECT)

    assert [t for t, _ in cells] == [f"{c}{i}" for i in range(rows) for c in "abc"]


def test_words_of_a_short_line_not_split_at_spaces():
    # One tall line sets the scale; the small line's word gaps are spaces, not columns
    image = _page([(20, 20, 300, 60), (20, 100, 60, 112), (68, 100, 120, 112)])

    cells = split_into_lines("Titel\nPostfach, 4000", image, RECT)

    assert [text for text, _ in cells] == ["Titel", "Postfach, 4000"]


def test_wrapped_paragraph_spread_over_its_lines():
    # GLM joins a paragraph wrapped over two lines into one line of text
    image = _page([(20, 20, 380, 40), (20, 60, 200, 80), (20, 140, 120, 160)])

    cells = split_into_lines("eins zwei drei vier fünf sechs\nEnde", image, RECT)

    texts = [text for text, _ in cells]
    assert texts[-1] == "Ende"
    assert " ".join(texts[:-1]) == "eins zwei drei vier fünf sechs"
    assert len(texts) == 3


def test_lines_the_model_left_out_are_skipped():
    # A three-column table and a paragraph below it that the model did not transcribe
    rows = [(40, 20 + i * 30, 200, 32 + i * 30) for i in range(4)]
    blocks = [(x, t, x + w, b) for _, t, _, b in rows for x, w in ((40, 120), (400, 160), (760, 140))]
    blocks += [(30, 150, 930, 164), (30, 175, 700, 189)]
    text = "\n".join(f"a{i}\tb{i}\tc{i}" for i in range(4))

    cells = split_into_lines(text, _page(blocks, size=(960, 200)), RECT)

    assert [t for t, _ in cells] == [f"{c}{i}" for i in range(4) for c in "abc"]
    assert all(box.b <= 50 + 140 / 2 for _, box in cells)


def test_more_lines_than_bands_keeps_whole_region():
    # The ink does not show the lines the model read (a frame or a drawing joins them):
    # the text keeps the whole region, as without line splitting
    image = _page([(20, 20, 380, 40)])

    cells = split_into_lines("a\nb", image, RECT)

    assert _boxes(cells) == [("a\nb", (100.0, 50.0, 300.0, 150.0))]


def test_text_lines_found_next_to_a_tall_chart():
    # A table above a bar chart: the chart's band must not make the text lines count as specks
    image = _page([(20, 10, 300, 25), (20, 40, 300, 55), (20, 70, 300, 85), (20, 120, 60, 190), (100, 150, 140, 190)])

    cells = split_into_lines("Erste\nZweite\nDritte", image, RECT)

    assert [text for text, _ in cells] == ["Erste", "Zweite", "Dritte"]
    assert [(round(b.t), round(b.b)) for _, b in cells] == [(55, 63), (70, 78), (85, 93)]


def test_label_and_value_stay_one_cell():
    # A form line: label and value with a wide gap are not table cells
    image = _page([(10, 20, 120, 40), (250, 20, 390, 40)])

    cells = split_into_lines("Vorname, Name: Helen Schmitz", image, RECT)

    assert [text for text, _ in cells] == ["Vorname, Name: Helen Schmitz"]


def test_no_contrast_returns_whole_region():
    cells = split_into_lines("Text", Image.new("RGB", (40, 20)), RECT)

    assert _boxes(cells) == [("Text", (100.0, 50.0, 300.0, 150.0))]


def test_light_text_on_dark_background():
    image = Image.new("RGB", (400, 200), "black")
    ImageDraw.Draw(image).rectangle((20, 20, 300, 40), fill="white")

    cells = split_into_lines("Weiss", image, RECT)

    assert _boxes(cells) == [("Weiss", (110.0, 60.0, 250.5, 70.5))]


def test_umlaut_dots_stay_with_their_line():
    image = _page([(20, 14, 26, 17), (20, 20, 300, 40)])

    cells = split_into_lines("Ärger", image, RECT)

    assert len(cells) == 1
    assert round(cells[0][1].t) == 57


def test_empty_text_gives_no_cells():
    assert split_into_lines(" \n ", _page([(20, 20, 300, 40)]), RECT) == []


def test_lines_next_to_a_box_of_lines_each_get_their_ink():
    # A form: a title over the whole width, field lines on the left and an address box on the
    # right whose lines share the rows of the fields; the model reads the left column first
    image = _page([
        (10, 10, 380, 30),
        (10, 60, 150, 75),
        (10, 90, 150, 105),
        (10, 120, 150, 135),
        (250, 62, 380, 77),
        (250, 92, 380, 107),
    ])

    cells = split_into_lines("Titel über alles\nA eins\nB zwei\nC drei\nLea Muster\nBasel", image, RECT)

    # Within a page unit: the crop is deskewed by a fraction of a degree
    assert [(text, pytest.approx(b.l, abs=1), pytest.approx(b.t, abs=1)) for text, b in cells] == [
        ("Titel über alles", 105, 55),
        ("A eins", 105, 80),
        ("B zwei", 105, 95),
        ("C drei", 105, 110),
        ("Lea Muster", 225, 81),
        ("Basel", 225, 96),
    ]


def test_scanner_speckle_is_not_a_line():
    lines = [(20, 20, 300, 40), (20, 80, 200, 100)]
    image = _page(lines)
    for x in range(3, 400, 7):
        for y in range(3, 200, 7):
            if not any(left - 3 <= x <= right + 3 and top - 3 <= y <= bottom + 3 for left, top, right, bottom in lines):
                image.putpixel((x, y), (0, 0, 0))

    cells = split_into_lines("Erste Zeile\nZweite", image, RECT)

    assert _boxes(cells) == [
        ("Erste Zeile", (110.0, 60.0, 250.5, 70.5)),
        ("Zweite", (110.0, 90.0, 200.5, 100.5)),
    ]


def test_dotted_line_next_to_the_text_is_left_out():
    # A receipt's dotted cut line at the right edge: its dashes neither widen the boxes nor are lines
    dashes = [(392, y, 393, y + 5) for y in range(0, 200, 10)]
    image = _page([(20, 20, 300, 40), (20, 80, 200, 100), *dashes])

    cells = split_into_lines("Erste Zeile\nZweite", image, RECT)

    assert _boxes(cells) == [
        ("Erste Zeile", (110.0, 60.0, 250.5, 70.5)),
        ("Zweite", (110.0, 90.0, 200.5, 100.5)),
    ]
