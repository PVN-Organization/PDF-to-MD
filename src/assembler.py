from __future__ import annotations

import logging
import re
from datetime import datetime

from .analyzer import PDFAnalysis
from .converter import ChunkResult
from .postprocess import (
    remove_duplicate_overlap as _remove_duplicate_overlap,
    normalize_headings as _normalize_headings,
    fix_broken_tables as _fix_broken_tables,
    remove_gibberish_ocr as _remove_gibberish_ocr,
    detect_hallucination as _detect_hallucination,
    remove_image_refs as _remove_image_refs,
    remove_html_tags as _remove_html_tags,
    shorten_dot_sequences as _shorten_dot_sequences,
    remove_prompt_leak as _remove_prompt_leak,
    remove_adjacent_duplicates as _remove_adjacent_duplicates,
    remove_stray_code_fences as _remove_stray_code_fences,
    remove_page_numbers as _remove_page_numbers,
    remove_embedded_page_numbers as _remove_embedded_page_numbers,
    remove_duplicate_sections as _remove_duplicate_sections,
    format_vn_header as _format_vn_header,
    clean_whitespace as _clean_whitespace,
)

logger = logging.getLogger(__name__)


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

    running_headers = _detect_running_headers(successful)

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

    merged = _remove_prompt_leak(merged)
    if running_headers:
        merged = _remove_running_headers(merged, running_headers)
    merged = _normalize_headings(merged)
    merged = _fix_broken_tables(merged)
    merged = _remove_gibberish_ocr(merged)
    merged = _detect_hallucination(merged)
    merged = _remove_image_refs(merged)
    merged = _remove_html_tags(merged)
    merged = _remove_stray_code_fences(merged)
    merged = _shorten_dot_sequences(merged)
    merged = _remove_page_numbers(merged)
    merged = _remove_embedded_page_numbers(merged, analysis.total_pages)
    merged = _remove_adjacent_duplicates(merged)
    merged = _remove_duplicate_sections(merged)
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


def _detect_running_headers(chunks: list[ChunkResult]) -> set[str]:
    """Detect repeated lines across chunks that are likely running headers/footers."""
    if len(chunks) < 2:
        return set()

    from collections import Counter
    line_counts: Counter[str] = Counter()

    for result in chunks:
        lines = result.markdown.strip().split("\n")
        seen_in_chunk: set[str] = set()
        check_lines = lines[:5] + lines[-5:]
        for line in check_lines:
            normalized = line.strip()
            if not normalized or len(normalized) < 10:
                continue
            if normalized.startswith("|") or normalized.startswith("#") or normalized.startswith("<!--"):
                continue
            if normalized not in seen_in_chunk:
                seen_in_chunk.add(normalized)
                line_counts[normalized] += 1

    threshold = max(2, len(chunks) // 2)
    headers = {
        line for line, count in line_counts.items()
        if count >= threshold
    }
    if headers:
        logger.info("Detected %d running header pattern(s)", len(headers))
    return headers


def _remove_running_headers(text: str, headers: set[str]) -> str:
    if not headers:
        return text
    lines = text.split("\n")
    result = []
    removed = 0
    for line in lines:
        if line.strip() in headers:
            removed += 1
            continue
        result.append(line)
    if removed:
        logger.info("Removed %d running header lines", removed)
    return "\n".join(result)
