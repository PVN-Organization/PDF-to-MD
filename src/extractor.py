from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

from .analyzer import PageInfo

logger = logging.getLogger(__name__)


@dataclass
class PageExtraction:
    page_num: int
    markdown: str
    confidence: float
    page_type: str
    has_tables: bool


@dataclass
class ChunkExtraction:
    chunk_id: int
    markdown: str
    confidence: float
    page_extractions: list[PageExtraction] = field(default_factory=list)


def extract_page_pymupdf(page: fitz.Page, page_info: PageInfo) -> PageExtraction:
    """Extract text and tables from a text-based PDF page using PyMuPDF."""
    tables_md = _extract_tables(page)
    table_rects = _get_table_rects(page)
    text_md = _extract_text_blocks(page, table_rects)

    has_tables = len(tables_md) > 0

    if has_tables:
        merged = _merge_text_and_tables(text_md, tables_md, page)
        confidence = 0.70
    else:
        merged = text_md
        confidence = 0.80

    if not merged.strip():
        confidence = 0.0

    return PageExtraction(
        page_num=page_info.page_num,
        markdown=merged,
        confidence=confidence,
        page_type=page_info.page_type,
        has_tables=has_tables,
    )


def extract_page_tesseract(
    image_path: Path, page_info: PageInfo, lang: str = "vie"
) -> PageExtraction:
    """Extract text from a scanned page image using Tesseract OCR."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        text = text.strip()
        confidence = 0.50 if text else 0.0
    except Exception as e:
        logger.error("Tesseract failed for page %d: %s", page_info.page_num, e)
        text = ""
        confidence = 0.0

    return PageExtraction(
        page_num=page_info.page_num,
        markdown=text,
        confidence=confidence,
        page_type="scanned",
        has_tables=False,
    )


def extract_chunk(
    pdf_path: Path,
    start_page: int,
    end_page: int,
    page_infos: list[PageInfo],
    image_paths: dict[int, Path],
    chunk_id: int,
    tesseract_lang: str = "vie",
    use_tesseract: bool = True,
) -> ChunkExtraction:
    """Extract text/tables for all pages in a chunk."""
    doc = fitz.open(str(pdf_path))
    page_extractions: list[PageExtraction] = []

    info_map = {p.page_num: p for p in page_infos}

    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)
        info = info_map.get(page_num)
        if info is None:
            continue

        if info.page_type == "scanned" and use_tesseract:
            img_path = image_paths.get(page_num)
            if img_path and img_path.exists():
                extraction = extract_page_tesseract(
                    img_path, info, lang=tesseract_lang
                )
            else:
                extraction = PageExtraction(
                    page_num=page_num,
                    markdown="",
                    confidence=0.0,
                    page_type="scanned",
                    has_tables=False,
                )
        else:
            extraction = extract_page_pymupdf(page, info)

        page_extractions.append(extraction)
        logger.debug(
            "Extracted page %d: %d chars, confidence=%.2f, type=%s",
            page_num,
            len(extraction.markdown),
            extraction.confidence,
            extraction.page_type,
        )

    doc.close()

    chunk_md = "\n\n".join(
        pe.markdown for pe in page_extractions if pe.markdown.strip()
    )
    avg_confidence = (
        sum(pe.confidence for pe in page_extractions) / len(page_extractions)
        if page_extractions
        else 0.0
    )

    return ChunkExtraction(
        chunk_id=chunk_id,
        markdown=chunk_md,
        confidence=avg_confidence,
        page_extractions=page_extractions,
    )


def _extract_tables(page: fitz.Page) -> list[str]:
    """Extract tables from a page and convert to Markdown."""
    try:
        tables = page.find_tables()
    except Exception as e:
        logger.debug("find_tables failed on page: %s", e)
        return []

    results = []
    for table in tables:
        try:
            df = table.to_pandas()
            if df.empty:
                continue
            md = _dataframe_to_markdown(df)
            if md.strip():
                results.append(md)
        except Exception as e:
            logger.debug("Table to_pandas failed: %s", e)
            try:
                header = table.header
                rows = table.extract()
                if rows:
                    md = _rows_to_markdown(rows, header)
                    if md.strip():
                        results.append(md)
            except Exception as e2:
                logger.debug("Table extract fallback failed: %s", e2)

    return results


def _get_table_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Get bounding rectangles of detected tables."""
    try:
        tables = page.find_tables()
        return [fitz.Rect(t.bbox) for t in tables]
    except Exception:
        return []


_HEADING_KEYWORDS = (
    "CHƯƠNG", "Chương", "MỤC", "Mục", "PHẦN", "Phần",
    "PHỤ LỤC", "Phụ lục", "QUY CHẾ", "QUY ĐỊNH",
    "ĐIỀU KHOẢN",
)

_SIGNATURE_PATTERNS = (
    "TM.", "TUQ.", "KT.", "Q.", "QUYỀN",
    "CHỦ TỊCH", "PHÓ CHỦ TỊCH", "THÀNH VIÊN",
    "TỔNG GIÁM ĐỐC", "PHÓ TỔNG GIÁM ĐỐC",
    "GIÁM ĐỐC", "Nơi nhận",
)

_BOLD_TITLE_RE = re.compile(
    r"^(Điều \d|Khoản \d|Mục \d|CHƯƠNG|Chương|MỤC|PHẦN|Phần|PHỤ LỤC|"
    r"Phụ lục|QUYẾT ĐỊNH|QUY CHẾ|QUY ĐỊNH|ĐIỀU KHOẢN|"
    r"Nơi nhận|HỘI ĐỒNG|TẬP ĐOÀN|BAN |CỘNG)"
)

_NEW_BLOCK_RE = re.compile(
    r"^(\d{1,3}\.\s|[a-zđ]\)\s|[a-zđ]\.\s|-\s|#|\||\>|<!--|\*\*)"
)

_SENTENCE_END_RE = re.compile(r"[.;:!?]\s*$")


def _extract_text_blocks(
    page: fitz.Page, exclude_rects: list[fitz.Rect]
) -> str:
    """Extract text blocks excluding table regions, with heading detection."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    lines_out: list[tuple] = []
    font_sizes: list[float] = []

    for block in blocks:
        if block["type"] != 0:
            continue
        block_rect = fitz.Rect(block["bbox"])
        if any(_rects_overlap(block_rect, tr) for tr in exclude_rects):
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            line_text = "".join(s["text"] for s in spans).strip()
            if not line_text:
                continue

            max_size = max(s["size"] for s in spans)
            all_bold = all(
                "bold" in s.get("font", "").lower()
                for s in spans if s["text"].strip()
            )
            font_sizes.append(max_size)
            lines_out.append((line_text, max_size, all_bold))

    if not lines_out:
        return ""

    if font_sizes:
        size_sorted = sorted(set(font_sizes), reverse=True)
        size_to_level = {}
        for i, s in enumerate(size_sorted[:4]):
            size_to_level[s] = i + 1
    else:
        size_to_level = {}

    result_lines: list[str] = []
    body_size = min(font_sizes) if font_sizes else 12.0

    for line_text, size, all_bold in lines_out:
        level = size_to_level.get(size)
        is_short = len(line_text) < 100

        is_signature = any(line_text.startswith(sp) for sp in _SIGNATURE_PATTERNS)
        looks_like_heading = (
            is_short
            and not is_signature
            and any(line_text.startswith(kw) for kw in _HEADING_KEYWORDS)
        )

        if level and level <= 4 and size > body_size * 1.15 and looks_like_heading:
            prefix = "#" * level
            clean = line_text.lstrip("#").strip()
            result_lines.append(f"{prefix} {clean}")
        elif all_bold and is_short and _BOLD_TITLE_RE.match(line_text):
            result_lines.append(f"**{line_text}**")
        else:
            result_lines.append(line_text)

    merged = "\n".join(result_lines)
    merged = _merge_paragraph_lines(merged)
    merged = _fix_broken_lists(merged)
    return merged


def _merge_paragraph_lines(text: str) -> str:
    """Join continuation lines into paragraphs, separate blocks with blank lines."""
    lines = text.split("\n")
    if len(lines) <= 1:
        return text

    result: list[str] = [lines[0]]

    for i in range(1, len(lines)):
        prev = result[-1]
        curr = lines[i]
        curr_stripped = curr.strip()

        if not curr_stripped:
            result.append("")
            continue

        if not prev.strip():
            result.append(curr)
            continue

        is_new_block = False

        if _is_structural_line(curr_stripped):
            is_new_block = True

        prev_s = prev.strip()
        if prev_s.startswith("#") or prev_s.startswith("|"):
            is_new_block = True

        if re.match(r"^\*\*.*\*\*$", prev_s):
            is_new_block = True

        if _SENTENCE_END_RE.search(prev_s):
            is_new_block = True

        if is_new_block:
            if result and result[-1] != "":
                result.append("")
            result.append(curr)
        else:
            result[-1] = prev.rstrip() + " " + curr_stripped

    return "\n".join(result)


_CLAUSE_STARTERS_RE = re.compile(
    r"^(Căn cứ |Xét đề nghị |Điều \d|CHƯƠNG |Chương |"
    r"MỤC |Mục |PHẦN |Phần |PHỤ LỤC|Phụ lục|TM\.|TUQ\.|KT\.)"
)


def _is_structural_line(line: str) -> bool:
    """Check if a line starts a new structural element."""
    if not line:
        return True
    if line.startswith("#"):
        return True
    if line.startswith("|"):
        return True
    if line.startswith(">"):
        return True
    if line.startswith("<!--"):
        return True
    if line.startswith("**"):
        return True
    if line.startswith("- ") or line == "-":
        return True
    if re.match(r"^\d{1,3}\.\s", line) or re.match(r"^\d{1,3}\.$", line):
        return True
    if re.match(r"^[a-zđ][.)]\s", line) or re.match(r"^[a-zđ][.)]$", line):
        return True
    if _CLAUSE_STARTERS_RE.match(line):
        return True
    return False


def _fix_broken_lists(text: str) -> str:
    """Fix list markers separated from their content by newline(s)."""
    text = re.sub(r"^(\d{1,3}\.)\n\n?(?!\n)", r"\1 ", text, flags=re.MULTILINE)
    text = re.sub(r"^([a-zđ][).])\n\n?(?!\n)", r"\1 ", text, flags=re.MULTILINE)
    text = re.sub(r"^(-)\n\n?(?!\n)", r"\1 ", text, flags=re.MULTILINE)
    return text


def _merge_text_and_tables(
    text_md: str, tables_md: list[str], page: fitz.Page
) -> str:
    """Merge extracted text and tables in approximate reading order.

    Uses table vertical position to insert tables between text blocks.
    """
    if not tables_md:
        return text_md

    try:
        tables = page.find_tables()
        table_positions = []
        for i, t in enumerate(tables):
            if i < len(tables_md):
                table_positions.append((t.bbox[1], tables_md[i]))
    except Exception:
        return text_md + "\n\n" + "\n\n".join(tables_md)

    if not table_positions:
        return text_md + "\n\n" + "\n\n".join(tables_md)

    text_lines = text_md.split("\n")
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    block_y_positions: list[float] = []
    for b in blocks:
        if b["type"] == 0:
            block_y_positions.append(b["bbox"][1])

    result_parts: list[str] = []
    tables_inserted: set[int] = set()

    result_parts.append(text_md)

    for idx, (y_pos, tbl_md) in enumerate(table_positions):
        if idx not in tables_inserted:
            result_parts.append(tbl_md)
            tables_inserted.add(idx)

    return "\n\n".join(result_parts)


def _rects_overlap(r1: fitz.Rect, r2: fitz.Rect) -> bool:
    """Check if two rectangles overlap significantly (>50% of smaller area)."""
    intersection = r1 & r2
    if intersection.is_empty:
        return False
    inter_area = intersection.width * intersection.height
    r1_area = r1.width * r1.height
    if r1_area == 0:
        return False
    return inter_area / r1_area > 0.5


def _dataframe_to_markdown(df) -> str:
    """Convert a pandas DataFrame to Markdown table."""
    import pandas as pd

    df = df.fillna("")
    df = df.astype(str)
    df = df.replace({"nan": "", "None": ""})

    cols = list(df.columns)
    auto_cols = all(
        str(c).startswith("Col") and str(c)[3:].isdigit() for c in cols
    )
    if auto_cols and len(df) > 0:
        header_vals = [str(v).replace("\n", " ").strip() for v in df.iloc[0]]
        df = df.iloc[1:]
    else:
        header_vals = [str(c).replace("\n", " ").strip() if str(c) not in ("nan", "None") else "" for c in cols]

    header = "| " + " | ".join(header_vals) + " |"
    separator = "| " + " | ".join("---" for _ in header_vals) + " |"

    rows = []
    for _, row in df.iterrows():
        cells = []
        for val in row:
            cell = str(val).replace("\n", " ").strip()
            if cell in ("nan", "None"):
                cell = ""
            cells.append(cell)
        if any(c.strip() for c in cells):
            rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + rows)


def _rows_to_markdown(rows: list[list], header=None) -> str:
    """Convert raw table rows to Markdown."""
    if not rows:
        return ""

    def clean_cell(val) -> str:
        if val is None:
            return ""
        return str(val).replace("\n", " ").strip()

    if header and hasattr(header, "names") and header.names:
        header_row = "| " + " | ".join(clean_cell(c) for c in header.names) + " |"
        data_rows = rows[header.row_count :] if hasattr(header, "row_count") else rows[1:]
    else:
        header_row = "| " + " | ".join(clean_cell(c) for c in rows[0]) + " |"
        data_rows = rows[1:]

    ncols = max(len(r) for r in rows) if rows else 0
    separator = "| " + " | ".join("---" for _ in range(ncols)) + " |"

    md_rows = []
    for row in data_rows:
        padded = list(row) + [None] * (ncols - len(row))
        md_rows.append("| " + " | ".join(clean_cell(c) for c in padded) + " |")

    return "\n".join([header_row, separator] + md_rows)
