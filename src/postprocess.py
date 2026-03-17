"""Common post-processing functions for PDF-to-Markdown conversion.

Shared between online (Gemini) and offline (marker) pipelines.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from difflib import SequenceMatcher

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

_PROMPT_LEAK_PATTERNS = [
    re.compile(r"chuyên gia chuyển đổi.*(?:markdown|Markdown)", re.IGNORECASE),
    re.compile(r"Bạn là chuyên gia", re.IGNORECASE),
    re.compile(r"văn bản pháp quy.*(?:sang|dạng).*(?:markdown|Markdown)", re.IGNORECASE),
    re.compile(r"Quy tắc bắt buộc", re.IGNORECASE),
    re.compile(r"CHỈ trả về nội dung Markdown", re.IGNORECASE),
]

_ADJACENT_DUP_RE = re.compile(r"\b(\w{2,})\s+\1\b")
_IMAGE_REF_PATTERN = re.compile(r"!\[.*?\]\(images/.*?\)")
_HTML_COMMENT_IMAGE = re.compile(r"<!--\s*\[Hình ảnh:.*?\]\s*-->")
_HTML_TAG = re.compile(
    r"</?(?:div|span|br|table|tr|td|th|p|img|center|font|b|i|u|em|strong)[^>]*>",
    re.IGNORECASE,
)
_LONG_DOTS = re.compile(r"\.{11,}")
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,3}\s*$")
_EMBEDDED_PAGE_NUM_RE = re.compile(
    r"(?<=[\u00C0-\u024F\u1E00-\u1EFFa-zA-Z,.])\s+(\d{1,3})\s+(?=[\u00C0-\u024F\u1E00-\u1EFFa-zA-Z])"
)


def remove_duplicate_overlap(prev_text: str, current_text: str) -> str:
    prev_lines = prev_text.strip().split("\n")
    curr_lines = current_text.strip().split("\n")
    if len(prev_lines) < 3 or len(curr_lines) < 3:
        return current_text

    window = 15
    prev_tail = [l.strip() for l in prev_lines[-window:] if l.strip()]
    curr_head = [l.strip() for l in curr_lines[:window] if l.strip()]
    overlap_len = 0
    for i in range(min(len(prev_tail), len(curr_head))):
        if prev_tail[-(i + 1):] == curr_head[:i + 1]:
            overlap_len = i + 1

    if overlap_len > 0:
        removed = 0
        result_lines = []
        prev_tail_set = set(prev_tail[-overlap_len:])
        for line in curr_lines:
            if removed < overlap_len and line.strip() in prev_tail_set:
                removed += 1
                continue
            result_lines.append(line)
        return "\n".join(result_lines)
    return current_text


def normalize_headings(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        match = re.match(r"^(#{1,6})([^ #\n])", line)
        if match:
            line = match.group(1) + " " + line[len(match.group(1)):]
        result.append(line)
    return "\n".join(result)


def fix_broken_tables(text: str) -> str:
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


def remove_gibberish_ocr(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    gibberish_buffer: list[str] = []
    gibberish_count = 0

    def _is_gibberish_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            return False
        if stripped.startswith("|") or stripped.startswith("#") or stripped.startswith("<!--"):
            return False
        if stripped.startswith("*") or stripped.startswith(">") or stripped.startswith("---"):
            return False
        alpha_chars = sum(1 for c in stripped if c.isalpha())
        if alpha_chars < 5:
            return False
        unusual = sum(1 for c in stripped if c in "§¿¡Ø£¥€{}[]@©®™¶†‡°·•…Œœ")
        if unusual > 2:
            return True
        words = stripped.split()
        if not words:
            return False
        short_words = sum(1 for w in words if len(w) <= 2 and w.isalpha())
        if len(words) >= 4 and short_words / len(words) > 0.6:
            return True
        consonant_clusters = len(re.findall(r"[bcdfghjklmnpqrstvwxz]{4,}", stripped, re.IGNORECASE))
        if consonant_clusters >= 2:
            return True
        return False

    def _flush_gibberish():
        nonlocal gibberish_count
        block = "\n".join(gibberish_buffer)
        total_lines = len([l for l in gibberish_buffer if l.strip()])
        gibberish_lines = sum(1 for l in gibberish_buffer if _is_gibberish_line(l))
        if total_lines > 0 and gibberish_lines / total_lines > 0.5 and gibberish_lines >= 5:
            gibberish_count += gibberish_lines
            logger.warning("Removed %d lines of gibberish OCR: %.60s...", gibberish_lines, block.replace("\n", " "))
        else:
            result.extend(gibberish_buffer)
        gibberish_buffer.clear()

    for line in lines:
        if _is_gibberish_line(line):
            gibberish_buffer.append(line)
        else:
            if gibberish_buffer:
                _flush_gibberish()
            result.append(line)

    if gibberish_buffer:
        _flush_gibberish()
    if gibberish_count:
        logger.info("Total gibberish lines removed: %d", gibberish_count)
    return "\n".join(result)


def detect_hallucination(text: str) -> str:
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
            logger.warning("Removed hallucinated block (%d chars): %.80s...", len(block), block.replace("\n", " "))
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


def remove_image_refs(text: str) -> str:
    text = _IMAGE_REF_PATTERN.sub("", text)
    text = _HTML_COMMENT_IMAGE.sub("", text)
    return text


def remove_html_tags(text: str) -> str:
    return _HTML_TAG.sub("", text)


def shorten_dot_sequences(text: str) -> str:
    return _LONG_DOTS.sub("..........", text)


def remove_prompt_leak(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if any(p.search(stripped) for p in _PROMPT_LEAK_PATTERNS):
            logger.warning("Removed prompt leak line: %.80s", stripped)
            continue
        result.append(line)
    return "\n".join(result)


def remove_adjacent_duplicates(text: str) -> str:
    def _dedup_match(m: re.Match) -> str:
        return m.group(1)

    lines = text.split("\n")
    result = []
    for line in lines:
        if line.strip().startswith("|") or line.strip().startswith("#"):
            result.append(line)
            continue
        result.append(_ADJACENT_DUP_RE.sub(_dedup_match, line))
    return "\n".join(result)


def remove_stray_code_fences(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("```", "```markdown", "```md"):
            continue
        result.append(line)
    return "\n".join(result)


def remove_page_numbers(text: str) -> str:
    lines = text.split("\n")
    result = []
    for line in lines:
        if _PAGE_NUMBER_LINE.match(line):
            continue
        result.append(line)
    return "\n".join(result)


def remove_embedded_page_numbers(text: str, total_pages: int) -> str:
    if total_pages <= 0:
        return text
    valid_range = set(range(1, total_pages + 1))
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") or stripped.startswith("#"):
            result.append(line)
            continue

        def _check_and_remove(m: re.Match) -> str:
            num = int(m.group(1))
            if num in valid_range:
                return " "
            return m.group(0)

        result.append(_EMBEDDED_PAGE_NUM_RE.sub(_check_and_remove, line))
    return "\n".join(result)


def remove_duplicate_sections(text: str) -> str:
    lines = text.split("\n")
    sections: list[tuple[int, int, str]] = []
    current_start = 0
    current_heading = ""

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("<!--"):
            if i > current_start:
                sections.append((current_start, i, current_heading))
            current_heading = stripped
            current_start = i
    sections.append((current_start, len(lines), current_heading))

    if len(sections) < 2:
        return text

    keep: list[bool] = [True] * len(sections)
    seen_headings: dict[str, int] = {}

    for idx, (start, end, heading) in enumerate(sections):
        if not heading:
            continue
        heading_norm = heading.strip().lower().replace("*", "").replace("#", "").strip()
        if not heading_norm:
            continue
        if heading_norm in seen_headings:
            prev_idx = seen_headings[heading_norm]
            prev_start, prev_end, _ = sections[prev_idx]
            prev_content = "\n".join(lines[prev_start:prev_end]).strip()
            curr_content = "\n".join(lines[start:end]).strip()
            if len(prev_content) > 50 and len(curr_content) > 50:
                ratio = SequenceMatcher(None, prev_content[:2000], curr_content[:2000]).ratio()
                if ratio > 0.8:
                    keep[idx] = False
                    logger.info("Removed duplicate section (%.0f%% similar): %.60s", ratio * 100, heading)
                    continue
        seen_headings[heading_norm] = idx

    if all(keep):
        return text

    result_lines: list[str] = []
    for idx, (start, end, _) in enumerate(sections):
        if keep[idx]:
            result_lines.extend(lines[start:end])
    return "\n".join(result_lines)


def _strip_md(s: str) -> str:
    return s.strip().replace("**", "").strip("*").strip("_").strip()


def format_vn_header(text: str) -> str:
    """Reformat VN legal document header into a 2-column borderless Markdown table."""
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

    logger.info("Formatted VN header: %d lines replaced with %d-row table", len(remove), n)
    return "\n".join(new_lines)


def clean_whitespace(text: str) -> str:
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def apply_all_postprocessing(text: str, total_pages: int = 0) -> str:
    """Apply the full post-processing pipeline to raw markdown text."""
    text = remove_prompt_leak(text)
    text = normalize_headings(text)
    text = fix_broken_tables(text)
    text = remove_gibberish_ocr(text)
    text = detect_hallucination(text)
    text = remove_image_refs(text)
    text = remove_html_tags(text)
    text = remove_stray_code_fences(text)
    text = shorten_dot_sequences(text)
    text = remove_page_numbers(text)
    if total_pages > 0:
        text = remove_embedded_page_numbers(text, total_pages)
    text = remove_adjacent_duplicates(text)
    text = remove_duplicate_sections(text)
    text = format_vn_header(text)
    text = clean_whitespace(text)
    return text
