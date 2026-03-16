from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime

from .analyzer import PDFAnalysis
from .converter import ChunkResult

logger = logging.getLogger(__name__)

_QUOC_HIEU_RE = re.compile(
    r"\*{0,2}CỘNG\s+H[OÒ][AÀ]\s+XÃ\s+HỘI\s+CHỦ\s+NGHĨA\s+VIỆT\s+NAM\*{0,2}"
)
_TIEU_NGU_RE = re.compile(
    r"\*{0,2}Độc\s+lập\s*[-–—]\s*Tự\s+do\s*[-–—]\s*Hạnh\s+phúc\*{0,2}"
)
_VN_DATE_RE = re.compile(
    r".{0,40}ngày\s+\**\d{1,2}\**\s+tháng\s+\**\d{1,2}\**\s+năm\s+\**\d{4}\**"
)
_DOC_NUM_RE = re.compile(r"Số\s*:\s*\S+(?:\s*/\s*\S+)*")

_HALLUCINATION_KEYWORDS = re.compile(
    r"\b(embodiment|invention|patent|lithium|electrode|electrolyte|cathode|anode"
    r"|battery cell|separator|coating|slurry|abstract|claims|prior art"
    r"|preferred embodiment|aqueous solution|substrate|polymer)\b",
    re.IGNORECASE,
)

_IMAGE_REF_PATTERN = re.compile(r"!\[.*?\]\(images/.*?\)")
_HTML_COMMENT_IMAGE = re.compile(r"<!--\s*\[Hình ảnh:.*?\]\s*-->")
_HTML_TAG = re.compile(r"</?(?:div|span|br|table|tr|td|th|p|img|center|font|b|i|u|em|strong)[^>]*>", re.IGNORECASE)
_LONG_DOTS = re.compile(r"\.{11,}")


def assemble(
    chunk_results: list[ChunkResult],
    analysis: PDFAnalysis,
) -> str:
    """Merge chunk markdowns into a single coherent document."""
    successful = sorted(
        [r for r in chunk_results if r.success],
        key=lambda r: r.chunk_id,
    )

    if not successful:
        logger.error("No successful chunks to assemble for %s", analysis.filename)
        return ""

    parts: list[str] = []

    parts.append(_build_metadata_header(analysis))

    for i, result in enumerate(successful):
        md = result.markdown.strip()
        if not md:
            continue

        if i > 0 and parts:
            md = _remove_duplicate_overlap(parts[-1], md)

        parts.append(md)

    merged = "\n\n---\n\n".join(parts) if len(parts) > 1 else parts[0] if parts else ""

    merged = _normalize_headings(merged)
    merged = _fix_broken_tables(merged)
    merged = _detect_hallucination(merged)
    merged = _remove_image_refs(merged)
    merged = _remove_html_tags(merged)
    merged = _shorten_dot_sequences(merged)
    merged = _remove_page_numbers(merged)
    merged = _format_vn_header(merged)
    merged = _clean_whitespace(merged)

    failed = [r for r in chunk_results if not r.success]
    if failed:
        merged += "\n\n---\n\n"
        merged += "<!-- CẢNH BÁO: Các phần sau chuyển đổi thất bại -->\n"
        for r in failed:
            merged += (
                f"<!-- Chunk {r.chunk_id} (trang {r.start_page + 1}-{r.end_page}): "
                f"{r.error} -->\n"
            )

    return merged


def _build_metadata_header(analysis: PDFAnalysis) -> str:
    lines = [
        f"<!-- Chuyển đổi tự động từ PDF sang Markdown -->",
        f"<!-- File gốc: {analysis.filename} -->",
        f"<!-- Số trang: {analysis.total_pages} | Kích thước: {analysis.file_size_mb} MB -->",
        f"<!-- Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->",
        "",
    ]
    return "\n".join(lines)


def _remove_duplicate_overlap(prev_text: str, current_text: str) -> str:
    """Remove overlapping content at the beginning of current chunk
    that may duplicate the end of the previous chunk."""
    prev_lines = prev_text.strip().split("\n")
    curr_lines = current_text.strip().split("\n")

    if len(prev_lines) < 3 or len(curr_lines) < 3:
        return current_text

    prev_tail = [l.strip() for l in prev_lines[-5:] if l.strip()]
    curr_head = [l.strip() for l in curr_lines[:5] if l.strip()]

    overlap_len = 0
    for i in range(min(len(prev_tail), len(curr_head))):
        if prev_tail[-(i + 1):] == curr_head[:i + 1]:
            overlap_len = i + 1

    if overlap_len > 0:
        removed = 0
        result_lines = []
        for line in curr_lines:
            if removed < overlap_len and line.strip() in [l.strip() for l in prev_tail]:
                removed += 1
                continue
            result_lines.append(line)
        return "\n".join(result_lines)

    return current_text


def _normalize_headings(text: str) -> str:
    """Ensure heading hierarchy is consistent throughout the document."""
    lines = text.split("\n")
    result = []
    for line in lines:
        match = re.match(r"^(#{1,6})([^ #\n])", line)
        if match:
            line = match.group(1) + " " + line[len(match.group(1)):]
        result.append(line)
    return "\n".join(result)


def _fix_broken_tables(text: str) -> str:
    """Fix tables that may have been broken across chunks."""
    lines = text.split("\n")
    result = []
    in_table = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                in_table = True
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if not re.match(r"^\|[\s\-:|]+\|$", next_line):
                        col_count = stripped.count("|") - 1
                        if col_count > 0:
                            result.append(line)
                            separator = "|" + "|".join(["---"] * col_count) + "|"
                            result.append(separator)
                            continue
            result.append(line)
        else:
            if in_table:
                in_table = False
            result.append(line)

    return "\n".join(result)


def _detect_hallucination(text: str) -> str:
    """Remove blocks of English text that are likely hallucinated content."""
    lines = text.split("\n")
    result = []
    english_buffer: list[str] = []

    def _is_mostly_english(line: str) -> bool:
        stripped = line.strip()
        if not stripped or stripped.startswith("|") or stripped.startswith("#"):
            return False
        if stripped.startswith("<!--") or stripped.startswith("---"):
            return False
        viet_chars = sum(1 for c in stripped if ord(c) > 127)
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        if alpha_chars == 0:
            return False
        return viet_chars / alpha_chars < 0.05 and alpha_chars > 30

    def _flush_buffer() -> None:
        block = "\n".join(english_buffer)
        if len(block) > 200 and _HALLUCINATION_KEYWORDS.search(block):
            logger.warning(
                "Removed hallucinated block (%d chars): %.80s...",
                len(block),
                block.replace("\n", " "),
            )
        else:
            result.extend(english_buffer)
        english_buffer.clear()

    for line in lines:
        if _is_mostly_english(line):
            english_buffer.append(line)
        else:
            if english_buffer:
                _flush_buffer()
            result.append(line)

    if english_buffer:
        _flush_buffer()

    return "\n".join(result)


def _remove_image_refs(text: str) -> str:
    """Remove leftover image markdown references and HTML image comments."""
    text = _IMAGE_REF_PATTERN.sub("", text)
    text = _HTML_COMMENT_IMAGE.sub("", text)
    return text


def _remove_html_tags(text: str) -> str:
    """Strip all HTML tags, keeping the text content inside them."""
    return _HTML_TAG.sub("", text)


def _shorten_dot_sequences(text: str) -> str:
    """Shorten excessively long sequences of dots to max 10."""
    return _LONG_DOTS.sub("..........", text)


_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$")


def _remove_page_numbers(text: str) -> str:
    """Remove standalone page number lines (1-3 digit numbers alone on a line)."""
    lines = text.split("\n")
    result = []
    for line in lines:
        if _PAGE_NUMBER_LINE.match(line):
            continue
        result.append(line)
    return "\n".join(result)


def _strip_md(s: str) -> str:
    """Strip bold/italic markdown markers."""
    return s.strip().replace("**", "").strip("*").strip("_").strip()


def _format_vn_header(text: str) -> str:
    """Reformat VN legal document header into a 2-column borderless Markdown table.

    Detects the standard pattern (quốc hiệu, tiêu ngữ, tên cơ quan, số hiệu,
    ngày tháng) within the first ~40 lines and reorganises them into:

        |  |  |
        |---|---|
        | Tên cơ quan | **CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM** |
        | Số: .../QĐ-... | **Độc lập - Tự do - Hạnh phúc** |
        |  | *Địa điểm, ngày ... tháng ... năm ...* |
    """
    text_nfc = unicodedata.normalize("NFC", text)
    lines = text_nfc.split("\n")

    qh_idx = None
    for i, line in enumerate(lines[:40]):
        if _QUOC_HIEU_RE.search(line):
            if line.strip().startswith("|"):
                return text
            qh_idx = i
            break

    if qh_idx is None:
        return text

    win_start = max(0, qh_idx - 6)
    win_end = min(len(lines), qh_idx + 8)

    qh_match = _QUOC_HIEU_RE.search(lines[qh_idx])
    qh_clean = _strip_md(qh_match.group(0))

    tn_idx = None
    tn_clean = ""
    for i in range(win_start, win_end):
        m = _TIEU_NGU_RE.search(lines[i])
        if m:
            tn_idx = i
            tn_clean = _strip_md(m.group(0))
            break

    date_idx = None
    date_clean = ""
    for i in range(win_start, win_end):
        m = _VN_DATE_RE.search(lines[i])
        if m:
            date_idx = i
            date_clean = _strip_md(m.group(0))
            break

    if tn_idx is not None and date_idx == tn_idx:
        tn_match = _TIEU_NGU_RE.search(lines[tn_idx])
        after_tn = lines[tn_idx][tn_match.end():]
        dm = _VN_DATE_RE.search(after_tn)
        if dm:
            date_clean = _strip_md(dm.group(0))

    left_raw: list[str] = []
    header_start = qh_idx
    gap = 0
    for i in range(qh_idx - 1, max(-1, qh_idx - 8), -1):
        stripped = lines[i].strip()
        if stripped.startswith("<!--") or stripped == "---":
            break
        if not stripped:
            gap += 1
            if gap > 2:
                break
            continue
        gap = 0
        header_start = i
        left_raw.insert(0, _strip_md(stripped))

    before_qh = lines[qh_idx][: qh_match.start()].strip()
    if before_qh:
        left_raw.append(_strip_md(before_qh))

    org_parts: list[str] = []
    doc_num = ""
    for part in left_raw:
        nm = _DOC_NUM_RE.search(part)
        if nm:
            before = part[: nm.start()].strip()
            if before:
                org_parts.append(before)
            doc_num = _strip_md(nm.group(0))
        else:
            org_parts.append(part)

    extra_remove: list[int] = []
    if not doc_num:
        exclude = {qh_idx, tn_idx, date_idx}
        for i in range(win_start, win_end):
            if i in exclude or not lines[i].strip():
                continue
            nm = _DOC_NUM_RE.search(lines[i])
            if nm:
                doc_num = _strip_md(nm.group(0))
                extra_remove.append(i)
                break

    remove: set[int] = set()
    for i in range(header_start, qh_idx + 1):
        remove.add(i)
    if tn_idx is not None:
        remove.add(tn_idx)
    if date_idx is not None:
        remove.add(date_idx)
    for idx in extra_remove:
        remove.add(idx)
    for i in range(min(remove), max(remove) + 1):
        if not lines[i].strip():
            remove.add(i)

    left_col = org_parts[:]
    if doc_num:
        left_col.append(doc_num)

    right_col: list[str] = [f"**{qh_clean}**"]
    if tn_clean:
        right_col.append(f"**{tn_clean}**")
    if date_clean:
        right_col.append(f"*{date_clean}*")

    n = max(len(left_col), len(right_col), 1)
    left_col.extend([""] * (n - len(left_col)))
    right_col.extend([""] * (n - len(right_col)))

    table = ["|  |  |", "|---|---|"]
    for lc, rc in zip(left_col, right_col):
        table.append(f"| {lc} | {rc} |")

    insert_at = min(remove)
    new_lines: list[str] = []
    for i, line in enumerate(lines):
        if i == insert_at:
            new_lines.extend(table)
        if i in remove:
            continue
        new_lines.append(line)

    logger.info(
        "Formatted VN header: %d lines replaced with %d-row table",
        len(remove),
        n,
    )
    return "\n".join(new_lines)


def _clean_whitespace(text: str) -> str:
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"
