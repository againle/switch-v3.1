"""Minimal, dependency-free PDF writer for text reports.

The runtime environment has no PDF library (no reportlab / fpdf / weasyprint),
so this module writes PDF 1.4 directly. It covers exactly what the session
report needs — headings, wrapped paragraphs, label/value rows, rules and page
numbers — using the base-14 Helvetica fonts, which need no embedding.

Deliberate limits:

* **Latin-1 / WinAnsi only.** Characters outside it (CJK, emoji, some symbols)
  are transliterated where common and replaced with ``?`` otherwise. Supporting
  more means embedding a TrueType font; this writer does not do that.
* **Text only.** No images, tables, links or Unicode line breaking.

The public surface is :class:`PdfDocument`; everything else is internal.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PdfDocument", "PAGE_SIZES"]

PAGE_SIZES: dict[str, tuple[float, float]] = {
    "A4": (595.276, 841.89),
    "LETTER": (612.0, 792.0),
}

Color = tuple[float, float, float]

BLACK: Color = (0.10, 0.10, 0.10)
MUTED: Color = (0.42, 0.42, 0.42)
ACCENT: Color = (0.07, 0.37, 0.33)
RULE: Color = (0.80, 0.82, 0.84)

# ── Adobe core-14 font metrics (1/1000 em) for chars 32..126 ─────────
# Needed for accurate wrapping; a uniform "average width" approximation
# would either overflow the margin or leave ragged gaps.
_HELVETICA: tuple[int, ...] = (
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584,
)

_HELVETICA_BOLD: tuple[int, ...] = (
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,
    975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,
    333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
    611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584,
)

_FIRST_CHAR = 32
_LAST_CHAR = 126

#: Common typographic characters mapped to ASCII, so text pulled from an LLM
#: does not turn into a row of "?" in the rendered report.
_TRANSLITERATE: dict[int, str] = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"',
    0x2013: "-", 0x2014: "-", 0x2015: "-", 0x2212: "-",
    0x2026: "...", 0x2022: "-", 0x00B7: "-",
    0x00A0: " ", 0x2009: " ", 0x202F: " ", 0x200B: "",
    0x2192: "->", 0x2190: "<-", 0x00D7: "x", 0x2265: ">=", 0x2264: "<=",
}


def _sanitize(text: str) -> str:
    """Transliterate, then drop anything WinAnsi cannot represent."""
    return str(text).translate(_TRANSLITERATE).encode("latin-1", "replace").decode("latin-1")


def _escape(text: str) -> str:
    """Escape a PDF literal string."""
    return (
        text.replace("\\", r"\\")
        .replace("(", r"\(")
        .replace(")", r"\)")
        .replace("\r", "")
        .replace("\n", " ")
    )


def _text_width(text: str, font: str, size: float) -> float:
    """Width of *text* in points."""
    table = _HELVETICA_BOLD if font == "bold" else _HELVETICA
    total = 0
    for char in text:
        code = ord(char)
        if _FIRST_CHAR <= code <= _LAST_CHAR:
            total += table[code - _FIRST_CHAR]
        else:
            total += table[ord("?") - _FIRST_CHAR]
    return total * size / 1000.0


def wrap_text(text: str, max_width: float, font: str = "regular", size: float = 10.0) -> list[str]:
    """Wrap *text* to *max_width* points, preserving explicit newlines.

    Words longer than the available width are hard-broken so they cannot
    overflow the page margin.
    """
    if max_width <= 1:
        return [text]

    lines: list[str] = []
    for paragraph in _sanitize(text).split("\n"):
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if _text_width(candidate, font, size) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            # Hard-break a single word that is too wide on its own.
            while _text_width(word, font, size) > max_width and len(word) > 1:
                cut = len(word)
                while cut > 1 and _text_width(word[:cut], font, size) > max_width:
                    cut -= 1
                lines.append(word[:cut])
                word = word[cut:]
            current = word
        lines.append(current)
    return lines


def _text_op(x: float, y: float, text: str, font: str, size: float, color: Color) -> str:
    name = "/F2" if font == "bold" else "/F1"
    red, green, blue = color
    return (
        f"BT {red:.3f} {green:.3f} {blue:.3f} rg {name} {size:.2f} Tf "
        f"1 0 0 1 {x:.2f} {y:.2f} Tm ({_escape(text)}) Tj ET"
    )


def _rect_op(x: float, y: float, w: float, h: float, color: Color) -> str:
    red, green, blue = color
    return f"{red:.3f} {green:.3f} {blue:.3f} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f"


class PdfDocument:
    """A growing multi-page text document.

    Layout is a simple downward cursor: callers add blocks and the document
    starts a new page whenever the next block would cross the bottom margin.
    """

    def __init__(
        self,
        *,
        title: str = "Report",
        page_size: str = "A4",
        margin: float = 48.0,
        footer: bool = True,
    ) -> None:
        self.width, self.height = PAGE_SIZES.get(page_size.upper(), PAGE_SIZES["A4"])
        self.margin = margin
        self.title = title
        self.footer = footer
        self._pages: list[list[str]] = []
        self._y: float = 0.0
        self.new_page()

    # ── Page plumbing ───────────────────────────────────────────────

    @property
    def content_width(self) -> float:
        return self.width - 2 * self.margin

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def new_page(self) -> None:
        self._pages.append([])
        self._y = self.height - self.margin

    def _bottom(self) -> float:
        return self.margin + (20.0 if self.footer else 0.0)

    def _ensure(self, height: float) -> None:
        if self._y - height < self._bottom():
            self.new_page()

    def move(self, dy: float) -> None:
        """Advance the cursor by *dy* points (negative = upwards)."""
        self._y -= dy

    def rule(self, *, width: float | None = None, thickness: float = 0.7, color: Color = RULE,
             space_before: float = 2.0, space_after: float = 6.0) -> None:
        self._y -= space_before
        self._ensure(thickness + space_after)
        self._y -= thickness
        self._pages[-1].append(
            _rect_op(self.margin, self._y, width if width is not None else self.content_width,
                     thickness, color)
        )
        self._y -= space_after

    # ── Blocks ──────────────────────────────────────────────────────

    def text(
        self,
        text: str,
        *,
        font: str = "regular",
        size: float = 10.0,
        indent: float = 0.0,
        color: Color = BLACK,
        leading: float | None = None,
        space_after: float = 0.0,
    ) -> None:
        """Add a wrapped paragraph."""
        leading = leading if leading is not None else size * 1.38
        for line in wrap_text(text, self.content_width - indent, font, size):
            self._ensure(leading)
            self._y -= leading
            self._pages[-1].append(
                _text_op(self.margin + indent, self._y, line, font, size, color)
            )
        self._y -= space_after

    def heading(
        self,
        text: str,
        *,
        size: float = 15.0,
        color: Color = ACCENT,
        space_before: float = 12.0,
        space_after: float = 6.0,
        underline: bool = False,
    ) -> None:
        """Add a bold heading, optionally underlined."""
        self._y -= space_before
        self.text(text, font="bold", size=size, color=color, space_after=0.0)
        if underline:
            self.rule(space_before=3.0, space_after=0.0)
        self._y -= space_after

    def kv(
        self,
        label: str,
        value: str,
        *,
        label_width: float = 116.0,
        size: float = 9.5,
        indent: float = 0.0,
        gap: float = 3.0,
    ) -> None:
        """Add a bold ``label:`` with a wrapped value in a second column."""
        value_indent = indent + label_width
        leading = size * 1.36
        first = True
        for line in wrap_text(str(value), self.content_width - value_indent, "regular", size):
            self._ensure(leading)
            self._y -= leading
            x = self.margin
            if first:
                self._pages[-1].append(
                    _text_op(x + indent, self._y, f"{label}:", "bold", size, MUTED)
                )
                first = False
            self._pages[-1].append(
                _text_op(x + value_indent, self._y, line, "regular", size, BLACK)
            )
        self._y -= gap

    # ── Output ──────────────────────────────────────────────────────

    def _footer_ops(self, page_no: int, total: int) -> list[str]:
        if not self.footer:
            return []
        y = self.margin - 16.0
        right = f"Page {page_no} of {total}"
        right_width = _text_width(right, "regular", 8.0)
        return [
            _text_op(self.margin, y, self.title, "regular", 8.0, MUTED),
            _text_op(self.width - self.margin - right_width, y, right, "regular", 8.0, MUTED),
        ]

    def save(self, path: str | Path) -> Path:
        """Serialize the document to *path* (parents created as needed)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        total = len(self._pages)
        streams = ["\n".join(page + self._footer_ops(i, total)) for i, page in enumerate(self._pages, 1)]

        # Object ids: 1 catalog, 2 pages, 3/4 fonts, then page/content pairs.
        page_ids = [5 + 2 * i for i in range(total)]

        objects: list[tuple[int, bytes]] = []
        objects.append((1, b"<< /Type /Catalog /Pages 2 0 R >>"))
        kids = " ".join(f"{oid} 0 R" for oid in page_ids)
        objects.append((2, f"<< /Type /Pages /Kids [{kids}] /Count {total} >>".encode("latin-1")))
        objects.append((3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
                           b"/Encoding /WinAnsiEncoding >>"))
        objects.append((4, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
                           b"/Encoding /WinAnsiEncoding >>"))

        for index, stream in enumerate(streams):
            page_id = page_ids[index]
            content_id = page_id + 1
            payload = stream.encode("latin-1", "replace")
            objects.append((
                content_id,
                b"<< /Length " + str(len(payload)).encode("ascii") + b" >>\nstream\n"
                + payload + b"\nendstream",
            ))
            objects.append((
                page_id,
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.2f} {self.height:.2f}] "
                    f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                    f"/Contents {content_id} 0 R >>"
                ).encode("latin-1"),
            ))

        objects.sort(key=lambda item: item[0])

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: dict[int, int] = {}
        for oid, body in objects:
            offsets[oid] = len(out)
            out += f"{oid} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

        max_id = max(offsets)
        start_xref = len(out)
        out += f"xref\n0 {max_id + 1}\n".encode("ascii")
        out += b"0000000000 65535 f \n"
        for oid in range(1, max_id + 1):
            out += f"{offsets.get(oid, 0):010d} 00000 n \n".encode("ascii")
        out += (
            f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{start_xref}\n%%EOF\n"
        ).encode("ascii")

        target.write_bytes(bytes(out))
        return target
