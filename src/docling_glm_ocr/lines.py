"""Place the text GLM-OCR read from a crop on the crop's text lines.

GLM-OCR returns the text of a whole crop without positions. A single cell covering the
whole crop misleads docling twice: a cell that touches any PDF text is dropped in favour of
the PDF text (so a padded title region loses its title), and a table region gets one cell
for all its cells (so table structure has nothing to match). This module finds the text
lines in the crop image from its ink and gives each line of text its own tight box; table
lines are split further at their column gaps, so table cells get boxes of their own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin

if TYPE_CHECKING:
    from PIL import Image

# Below this grey-level contrast between ink and background the crop is treated as blank.
_MIN_CONTRAST: Final = 40
# A pixel row counts as ink when it holds at least this many ink pixels (ignores specks).
_MIN_ROW_INK: Final = 2
# Relative to the line height: a band this small is an accent or a speck ...
_SMALL_BAND: Final = 0.25
# ... and joins a neighbouring band at most this far away; otherwise it is dropped.
_ATTACH_GAP: Final = 0.3
# Relative to the line height: a gap at least this wide separates two columns (a space is ~0.25).
_MIN_COLUMN_GAP: Final = 0.8
# Relative to the line height: ink narrower than this at either end of a band is a dash of a
# dotted line or a frame next to the text, not a letter; a band of only such ink is no line.
_MIN_INK_WIDTH: Final = 0.15
# Relative to the line height: a strip of ink at least this far from all other ink down the
# whole crop stands apart (wider than the space between two letters).
_MIN_STRIP_GAP: Final = 0.25
# A line without tabs is split into columns only from this many on: a label and its value
# ("Name: Helen Schmitz") have one wide gap too, but they are one cell.
_MIN_UNTABBED_COLUMNS: Final = 3
# Table rules: runs of ink at least this long (pixels, and share of the crop's side) ...
_MIN_RULE_PX: Final = 48
_MIN_RULE_SHARE: Final = 0.1
# ... and at most this thick (pixels); text never forms such long thin runs.
_MAX_RULE_THICKNESS: Final = 8
# An ink pixel with fewer ink pixels than this in its 3 x 3 neighbourhood (itself included)
# is scanner speckle: a stroke of text, even a one pixel thin one, has at least this many.
_MIN_NEIGHBOURS: Final = 3
# Bands lower than this (pixels) are left out when measuring the line height.
_MIN_LINE_PX: Final = 4
# Relative to the line height: blank space this wide or high separates blocks of text that
# stand side by side (an address box next to form fields, text next to a QR code) ...
_MIN_BLOCK_GAP: Final = 2.0
# ... and blocks above each other.
_MIN_BLOCK_SPACE: Final = 1.0
# Matching lines to bands: cost of leaving a band without text, of wrapping a line over
# one more band, and of a table row on a band with more or fewer columns than cells.
_SKIP_BAND_COST: Final = 1.2
_WRAP_COST: Final = 0.3
_EXTRA_COLUMNS_COST: Final = 0.3
_FEW_COLUMNS_COST: Final = 1.0
# A line of text is wrapped over at most this many bands, of about the same height (its font).
_MAX_WRAP: Final = 4
_MAX_WRAP_HEIGHT_RATIO: Final = 1.5
# Above this many lines or bands, matching them is not worth it.
_MAX_ITEMS: Final = 300

# Skew angles tried for scans, in degrees either way, and the step between them.
_MAX_SKEW_DEG: Final = 2.0
_SKEW_STEP_DEG: Final = 0.1

Band = tuple[int, int, int, int]  # l, t, r, b in pixels, r and b exclusive


def _long_runs(mask: np.ndarray, length: int, axis: int) -> np.ndarray:
    """Pixels of ``mask`` that lie on a run of at least ``length`` along ``axis``."""
    m = mask if axis == 1 else mask.T
    n = m.shape[1]
    out = np.zeros_like(m)
    if length <= n:
        sums = np.cumsum(np.pad(m, ((0, 0), (1, 0))), axis=1, dtype=np.int32)
        starts = np.zeros(m.shape, dtype=np.int32)
        starts[:, : n - length + 1] = (sums[:, length:] - sums[:, : n - length + 1]) == length
        covered = np.cumsum(np.pad(starts, ((0, 0), (1, 0))), axis=1)
        idx = np.arange(n)
        out = (covered[:, idx + 1] - covered[:, np.maximum(idx - length + 1, 0)]) > 0
    return out if axis == 1 else out.T


def _without_rules(mask: np.ndarray) -> np.ndarray:
    """Remove table rules and frames: long, thin runs of ink in either direction."""
    keep = mask.copy()
    for axis, side in ((1, mask.shape[1]), (0, mask.shape[0])):
        long = _long_runs(mask, max(_MIN_RULE_PX, int(_MIN_RULE_SHARE * side)), axis)
        thick = _long_runs(long, _MAX_RULE_THICKNESS + 1, 1 - axis)
        keep &= ~(long & ~thick)
    return keep


def _ink_mask(image: Image.Image) -> np.ndarray | None:
    grey = np.asarray(image.convert("L"), dtype=np.int16)
    if grey.size == 0:
        return None
    diff = np.abs(grey - int(np.median(grey)))
    peak = int(diff.max())
    if peak < _MIN_CONTRAST:
        return None
    return _without_speckle(_without_rules(diff > max(_MIN_CONTRAST // 2, peak // 2)))


def _without_speckle(mask: np.ndarray) -> np.ndarray:
    """Remove scanner speckle: ink pixels with too few ink neighbours."""
    padded = np.pad(mask, 1).astype(np.uint8)
    h, w = mask.shape
    neighbours = sum(padded[dy : dy + h, dx : dx + w] for dy in range(3) for dx in range(3))
    return mask & (neighbours >= _MIN_NEIGHBOURS)


def _deskew(mask: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Shear a skewed scan's ink so its text lines run horizontally.

    Returns the sheared mask, the slope removed (``y' = y - x * slope + offset``) and the
    offset. The slope is the one whose row profile is sharpest (largest sum of squares).
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return mask, 0.0, 0
    best_slope, best_score = 0.0, float(np.square(np.bincount(ys)).sum())
    steps = round(_MAX_SKEW_DEG / _SKEW_STEP_DEG)
    for slope in np.tan(np.radians(np.arange(-steps, steps + 1) * _SKEW_STEP_DEG)):
        if slope == 0:
            continue
        shifted = np.round(ys - xs * slope).astype(np.int64)
        score = float(np.square(np.bincount(shifted - shifted.min())).sum())
        if score > best_score:
            best_slope, best_score = float(slope), score
    if best_slope == 0:
        return mask, 0.0, 0
    shifted = np.round(ys - xs * best_slope).astype(np.int64)
    offset = int(-shifted.min())
    sheared = np.zeros((int(shifted.max()) + offset + 1, mask.shape[1]), dtype=bool)
    sheared[shifted + offset, xs] = True
    return sheared, best_slope, offset


def _unshear(box: Band, slope: float, offset: int, height: int) -> Band:
    """Map a box found on the sheared mask back onto the crop."""
    left, t, right, b = box
    if slope == 0:
        return box
    ends = (left * slope, right * slope)
    top = max(0, int(np.floor(t - offset + min(ends))))
    bottom = min(height, int(np.ceil(b - offset + max(ends))))
    return (left, top, right, bottom)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _line_height(boxes: list[Band]) -> float:
    """A typical text line's height: the tallest band would be a drawing or a chart next to the text.

    Each band counts by its width: specks and the dashes of a dotted line can outnumber the
    lines, but not outweigh them. Bands of a few pixels are left out.
    """
    lines = [box for box in boxes if box[3] - box[1] >= _MIN_LINE_PX] or boxes
    heights = np.array([box[3] - box[1] for box in lines], dtype=float)
    widths = np.array([max(1, box[2] - box[0]) for box in lines], dtype=float)
    order = np.argsort(heights)
    share = np.cumsum(widths[order]) / widths.sum()
    return float(heights[order][np.searchsorted(share, 0.75)])


def _row_boxes(mask: np.ndarray) -> list[Band]:
    """The runs of ink rows, each boxed by its ink's horizontal extent (counted in columns)."""
    boxes = []
    for t, b in _runs(mask.sum(axis=1) >= _MIN_ROW_INK):
        width = int(mask[t:b].any(axis=0).sum())
        boxes.append((0, t, width, b))
    return boxes


def _without_thin_columns(mask: np.ndarray) -> np.ndarray:
    """Remove a dotted or thin line running down the crop: a narrow strip of ink set apart.

    Its dashes would join the rows of the text lines next to them and widen their boxes.
    """
    boxes = _row_boxes(mask)
    if not boxes:
        return mask
    height = _line_height(boxes)
    keep = mask.copy()
    strips = _columns((0, 0, mask.shape[1], mask.shape[0]), mask, height, _MIN_STRIP_GAP)
    for left, _, right, _ in strips:
        if right - left < _MIN_INK_WIDTH * height:
            keep[:, left:right] = False
    return keep


def _bands(mask: np.ndarray) -> list[Band]:
    boxes = _row_boxes(mask)
    if not boxes:
        return []
    rows = [(t, b) for _, t, _, b in boxes]
    height = _line_height(boxes)
    lines = {i: [t, b] for i, (t, b) in enumerate(rows) if b - t >= _SMALL_BAND * height}
    for i, (t, b) in enumerate(rows):
        if i in lines:
            continue
        # An accent or a speck: it joins the nearest line if close enough, otherwise it is dropped
        gap, j = min((t - rows[j][1] if j < i else rows[j][0] - b, j) for j in lines)
        if gap <= _ATTACH_GAP * height:
            lines[j] = [min(lines[j][0], t), max(lines[j][1], b)]
    kept = [lines[i] for i in sorted(lines)]
    bands: list[Band] = []
    for t, b in kept:
        ink = [c for c in _columns((0, t, mask.shape[1], b), mask, height) if c[2] - c[0] >= _MIN_INK_WIDTH * height]
        if ink:
            bands.append((ink[0][0], t, ink[-1][2], b))
    return bands


def _gaps(ink: np.ndarray, size: float) -> list[tuple[int, int]]:
    """Blank runs at least ``size`` long between ink (not at the ends)."""
    return [(s, e) for s, e in _runs(~ink) if e - s >= size and s > 0 and e < len(ink)]


def _block_bands(mask: np.ndarray, height: float) -> list[list[Band]]:
    """The bands of each block of text, block by block in reading order.

    Rows across the whole crop merge lines that stand side by side: field lines and an address
    box next to them, or text next to a QR code. Cutting the crop at wide blank gaps, first
    between columns and else between rows (top to bottom, left to right), gives each its own.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return []
    top, left = int(rows[0]), int(cols[0])
    sub = mask[top : int(rows[-1]) + 1, left : int(cols[-1]) + 1]
    for axis, gap in ((0, _MIN_BLOCK_GAP), (1, _MIN_BLOCK_SPACE)):
        ink = sub.any(axis=axis) if axis == 0 else sub.sum(axis=1) >= _MIN_ROW_INK
        gaps = _gaps(ink, gap * height)
        if not gaps:
            continue
        edges = [0, *(x for g in gaps for x in g), len(ink)]
        blocks: list[list[Band]] = []
        for s, e in zip(edges[::2], edges[1::2], strict=True):
            part = np.zeros_like(sub)
            if axis == 0:
                part[:, s:e] = sub[:, s:e]
            else:
                part[s:e] = sub[s:e]
            blocks += _block_bands(part, height)
        return [[(x0 + left, y0 + top, x1 + left, y1 + top) for x0, y0, x1, y1 in block] for block in blocks]
    # A block of a few thin pixels is what is left of a frame or a rule, not a line
    found = [band for band in _bands(sub) if band[3] - band[1] >= _SMALL_BAND * height]
    return [[(x0 + left, y0 + top, x1 + left, y1 + top) for x0, y0, x1, y1 in found]] if found else []


def _group(weights: list[float], targets: list[float]) -> list[tuple[int, int]] | None:
    """Split ``weights`` into ``len(targets)`` contiguous groups whose shares best match the targets'."""
    n, m = len(weights), len(targets)
    if m == 0 or n < m or n > _MAX_ITEMS:
        return None
    total_w, total_t = sum(weights) or 1.0, sum(targets) or 1.0
    prefix = [0.0]
    for w in weights:
        prefix.append(prefix[-1] + w / total_w)
    inf = float("inf")
    cost = [[inf] * (n + 1) for _ in range(m + 1)]
    cut = [[0] * (n + 1) for _ in range(m + 1)]
    cost[0][0] = 0.0
    for j in range(1, m + 1):
        share = targets[j - 1] / total_t
        for i in range(j, n - (m - j) + 1):
            for k in range(j - 1, i):
                c = cost[j - 1][k] + abs(prefix[i] - prefix[k] - share)
                if c < cost[j][i]:
                    cost[j][i], cut[j][i] = c, k
    groups = []
    i = n
    for j in range(m, 0, -1):
        k = cut[j][i]
        groups.append((k, i))
        i = k
    return groups[::-1]


def _union(bands: list[Band]) -> Band:
    return (min(b[0] for b in bands), min(b[1] for b in bands), max(b[2] for b in bands), max(b[3] for b in bands))


def _spread_words(text: str, bands: list[Band]) -> list[tuple[str, Band]]:
    """Distribute the words of one text line over the bands it was wrapped onto."""
    words = text.split()
    if len(words) < len(bands):
        return [(" ".join(words), _union(bands))]
    widths = [b[2] - b[0] for b in bands]
    total_chars = sum(len(w) + 1 for w in words)
    bounds = np.cumsum(widths) / sum(widths)
    buckets: list[list[str]] = [[] for _ in bands]
    pos = 0
    for word in words:
        mid = (pos + len(word) / 2) / total_chars
        buckets[min(int(np.searchsorted(bounds, mid)), len(bands) - 1)].append(word)
        pos += len(word) + 1
    if any(not bucket for bucket in buckets):
        return [(" ".join(words), _union(bands))]
    return [(" ".join(bucket), band) for bucket, band in zip(buckets, bands, strict=True)]


def _columns(band: Band, mask: np.ndarray, line_height: float, gap: float = _MIN_COLUMN_GAP) -> list[Band]:
    """The band's columns: ink separated by gaps much wider than a space."""
    left, t, right, b = band
    columns: list[list[int]] = []
    for start, end in _runs(mask[t:b, left:right].any(axis=0)):
        if columns and start - columns[-1][1] < gap * line_height:
            columns[-1][1] = end
        else:
            columns.append([start, end])
    return [(left + start, t, left + end, b) for start, end in columns]


def _split_columns(line: str, band: Band, boxes: list[Band]) -> list[tuple[str, Band]]:
    """Split a line over its band's columns, one piece per column.

    Table cells separated by tabs go to the columns in order (neighbouring cells share a
    column if the band has fewer columns); untabbed words are spread over the columns.
    """
    plain = " ".join(line.replace("\t", " ").split())
    if len(boxes) < 2:  # noqa: PLR2004 -- one column: nothing to split
        return [(plain, band)]
    parts = [p.strip() for p in line.split("\t") if p.strip()]
    if len(parts) > 1:
        return _split_cells(parts, boxes) or [(plain, band)]
    return _split_words(plain, boxes) or [(plain, band)]


def _split_cells(parts: list[str], boxes: list[Band]) -> list[tuple[str, Band]] | None:
    """Tab-separated cells on the band's columns, or None if they cannot be matched."""
    if len(parts) > len(boxes):
        # Cells closer than a column gap share a column
        cell_groups = _group([float(len(p) + 1) for p in parts], [float(box[2] - box[0]) for box in boxes])
        if cell_groups is None:
            return None
        return [(" ".join(parts[s:e]), box) for (s, e), box in zip(cell_groups, boxes, strict=True)]
    # Keep the widest gaps, one fewer than there are cells
    gaps = sorted(range(len(boxes) - 1), key=lambda i: boxes[i + 1][0] - boxes[i][2], reverse=True)
    cuts = sorted(gaps[: len(parts) - 1])
    groups = list(zip([0, *(i + 1 for i in cuts)], [*(i + 1 for i in cuts), len(boxes)], strict=True))
    return [(part, _union(boxes[s:e])) for part, (s, e) in zip(parts, groups, strict=True)]


def _split_words(plain: str, boxes: list[Band]) -> list[tuple[str, Band]] | None:
    """The words of an untabbed line spread over the band's columns, or None to keep the line whole."""
    if len(boxes) < _MIN_UNTABBED_COLUMNS:
        return None
    words = plain.split()
    word_groups = _group([float(len(w) + 1) for w in words], [float(box[2] - box[0]) for box in boxes])
    if word_groups is None:
        return None
    return [(" ".join(words[s:e]), box) for (s, e), box in zip(word_groups, boxes, strict=True)]


def _match_cost(chars: int, cells: int, width: int, columns: int, density: float) -> float:
    """How badly a line fits a band: its length against the ink's width, its cells against the columns."""
    cost = abs(np.log((chars + 1) / (width * density + 1)))
    if cells > 1:
        cost += 0.0 if columns == cells else (_EXTRA_COLUMNS_COST if columns > cells else _FEW_COLUMNS_COST)
    return float(cost)


def _wrap_room(bands: list[Band], starts: frozenset[int]) -> list[int]:
    """For each end ``i``, how many bands before it one line could be wrapped over.

    A wrapped line keeps its font, so its bands are about as high as each other, and it stays
    in its block of text (``starts``: the first band of each block but the first).
    """
    room = [0]
    for i, band in enumerate(bands):
        if i == 0 or i in starts:
            room.append(1)
            continue
        low, high = sorted((band[3] - band[1], bands[i - 1][3] - bands[i - 1][1]))
        room.append(room[-1] + 1 if high <= _MAX_WRAP_HEIGHT_RATIO * max(low, 1) else 1)
    return room


def _align(
    lines: list[str], bands: list[Band], columns: list[int], starts: frozenset[int] = frozenset()
) -> list[tuple[int, int, int]] | None:
    """Match each line to one band, or to a few consecutive bands it was wrapped over.

    Bands may stay unmatched: text the model left out, or a sliver of the neighbouring
    region. A line is not wrapped into the next block of text (``starts``: its first bands).
    Returns ``(line, first band, end band)`` triples, or None if too large.
    """
    m, n = len(lines), len(bands)
    if n > _MAX_ITEMS or m > n:
        return None
    chars = [len(" ".join(line.split())) for line in lines]
    cells = [len([p for p in line.split("\t") if p.strip()]) for line in lines]
    words = [len(line.split()) for line in lines]
    room = _wrap_room(bands, starts)
    widths = [b[2] - b[0] for b in bands]
    # Medians: robust to the bands without text
    density = float(np.median(chars)) / max(1.0, float(np.median(widths)))
    path: list[tuple[int, int, int]] = []
    for _ in range(2):  # the second pass measures the density on the matched bands only
        cost = np.full((m + 1, n + 1), np.inf)
        cost[0] = np.arange(n + 1) * _SKIP_BAND_COST
        back: dict[tuple[int, int], int] = {}
        for j in range(1, m + 1):
            for i in range(j, n + 1):
                cost[j, i], back[j, i] = cost[j, i - 1] + _SKIP_BAND_COST, 0
                # A line wraps over at most as many bands as it has words
                for k in range(1, min(_MAX_WRAP, words[j - 1], i - j + 1, room[i]) + 1):
                    width = sum(widths[i - k : i])
                    c = cost[j - 1, i - k] + (k - 1) * _WRAP_COST
                    if k > 1 and cells[j - 1] > 1:
                        c += _FEW_COLUMNS_COST  # a table row is not wrapped over several lines
                    c += _match_cost(chars[j - 1], cells[j - 1] if k == 1 else 1, width, columns[i - 1], density)
                    if c < cost[j, i]:
                        cost[j, i], back[j, i] = c, k
        path, j, i = [], m, n
        while j > 0:
            k = back[j, i]
            if k:
                path.append((j - 1, i - k, i))
                j -= 1
            i -= k or 1
        path.reverse()
        matched = sum(sum(widths[s:e]) for _, s, e in path)
        density = sum(chars) / max(1, matched)
    return path


def _place(lines: list[str], bands: list[Band], mask: np.ndarray) -> list[tuple[str, Band]] | None:
    """Pieces of text with the band (or column) each was read from, or None if they do not match.

    Fewer bands than lines means the ink does not show the lines the model read: a frame or a
    drawing joins them, or the text is too faint. Placing the lines anyway would put text on
    the wrong ink, so the caller keeps the whole region instead.
    """
    starts: frozenset[int] = frozenset()
    if len(lines) > len(bands):
        blocks = _block_bands(mask, _line_height(bands))
        bands = [band for block in blocks for band in block]
        if len(lines) > len(bands):
            return None
        starts = frozenset(np.cumsum([len(block) for block in blocks[:-1]]).tolist())
    flat = [" ".join(line.replace("\t", " ").split()) for line in lines]
    height = _line_height(bands)
    columns = [_columns(band, mask, height) for band in bands]
    path = _align(lines, bands, [len(c) for c in columns], starts)
    if path is None:
        return None
    pieces: list[tuple[str, Band]] = []
    for j, s, e in path:
        if e - s == 1:
            pieces.extend(_split_columns(lines[j], bands[s], columns[s]))
        else:
            pieces.extend(_spread_words(flat[j], bands[s:e]))
    return pieces


def split_into_lines(text: str, image: Image.Image, rect: BoundingBox) -> list[tuple[str, BoundingBox]]:
    """Give each line of ``text`` the page box of the text line it was read from in ``image``.

    ``image`` is the crop of the page region ``rect`` (top-left origin). Lines in ``text``
    may hold table cells separated by tabs. When lines and image cannot be matched, the
    whole text keeps ``rect``, as without line splitting.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    whole = [("\n".join(" ".join(line.replace("\t", " ").split()) for line in lines), rect)]
    mask = _ink_mask(image)
    bands: list[Band] = []
    if mask is not None:
        mask, slope, offset = _deskew(mask)
        mask = _without_thin_columns(mask)
        bands = _bands(mask)
    if mask is None or not bands:
        return whole
    pieces = _place(lines, bands, mask)
    if pieces is None:
        return whole

    sx = (rect.r - rect.l) / image.width
    sy = (rect.b - rect.t) / image.height
    placed = []
    for piece, box in pieces:
        left, t, right, b = _unshear(box, slope, offset, image.height)
        placed.append((
            piece,
            BoundingBox(
                l=rect.l + left * sx,
                t=rect.t + t * sy,
                r=rect.l + right * sx,
                b=rect.t + b * sy,
                coord_origin=CoordOrigin.TOPLEFT,
            ),
        ))
    return placed
