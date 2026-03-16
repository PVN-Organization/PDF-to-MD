from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fitz

from .analyzer import PDFAnalysis
from .config import Config

logger = logging.getLogger(__name__)

TOKENS_PER_IMAGE_PAGE = 512
MAX_CONTEXT_TOKENS = 900_000  # keep some headroom from the 1M limit


@dataclass
class ChunkPlan:
    chunk_id: int
    start_page: int   # 0-indexed inclusive
    end_page: int      # exclusive
    strategy: str      # "direct_pdf" | "page_images"
    estimated_tokens: int
    page_types: list[str]
    table_pages: list[int] = field(default_factory=list)


def _find_section_breaks(pdf_path: str, total_pages: int) -> set[int]:
    """Find page indices that start a new section (Chương, Mục, Phần)
    by detecting large/bold text at the top of a page."""
    breaks: set[int] = set()
    doc = fitz.open(pdf_path)
    section_keywords = ("CHƯƠNG", "Chương", "MỤC", "Mục", "PHẦN", "Phần", "PHỤ LỤC", "Phụ lục")

    for i in range(total_pages):
        page = doc.load_page(i)
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks[:5]:  # only look at top blocks
            if block["type"] != 0:
                continue
            for line in block.get("lines", [])[:3]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if any(line_text.startswith(kw) for kw in section_keywords):
                    breaks.add(i)
                    break

    doc.close()
    return breaks


def create_plan(analysis: PDFAnalysis, config: Config) -> list[ChunkPlan]:
    total = analysis.total_pages
    chunk_size = config.default_chunk_pages

    if total <= chunk_size:
        strategy = _pick_strategy(analysis.pages, 0, total)
        tokens = _estimate_chunk_tokens(analysis, 0, total, strategy)
        return [
            ChunkPlan(
                chunk_id=0,
                start_page=0,
                end_page=total,
                strategy=strategy,
                estimated_tokens=tokens,
                page_types=[p.page_type for p in analysis.pages],
                table_pages=[p.page_num for p in analysis.pages if p.has_tables],
            )
        ]

    section_breaks = _find_section_breaks(str(analysis.filepath), total)

    chunks: list[ChunkPlan] = []
    start = 0
    chunk_id = 0

    while start < total:
        ideal_end = min(start + chunk_size, total)

        # Try to snap to a section break near the ideal end
        best_end = ideal_end
        for candidate in range(ideal_end, max(start + 3, ideal_end - 4), -1):
            if candidate in section_breaks:
                best_end = candidate
                break

        page_slice = analysis.pages[start:best_end]
        strategy = _pick_strategy(page_slice, start, best_end)
        tokens = _estimate_chunk_tokens(analysis, start, best_end, strategy)

        # Split further if tokens exceed context limit
        if tokens > MAX_CONTEXT_TOKENS and (best_end - start) > 2:
            mid = start + (best_end - start) // 2
            for candidate in range(mid, start + 2, -1):
                if candidate in section_breaks:
                    mid = candidate
                    break

            for sub_start, sub_end in [(start, mid), (mid, best_end)]:
                sub_slice = analysis.pages[sub_start:sub_end]
                sub_strategy = _pick_strategy(sub_slice, sub_start, sub_end)
                sub_tokens = _estimate_chunk_tokens(analysis, sub_start, sub_end, sub_strategy)
                chunks.append(
                    ChunkPlan(
                        chunk_id=chunk_id,
                        start_page=sub_start,
                        end_page=sub_end,
                        strategy=sub_strategy,
                        estimated_tokens=sub_tokens,
                        page_types=[p.page_type for p in sub_slice],
                        table_pages=[p.page_num for p in sub_slice if p.has_tables],
                    )
                )
                chunk_id += 1
        else:
            chunks.append(
                ChunkPlan(
                    chunk_id=chunk_id,
                    start_page=start,
                    end_page=best_end,
                    strategy=strategy,
                    estimated_tokens=tokens,
                    page_types=[p.page_type for p in page_slice],
                    table_pages=[p.page_num for p in page_slice if p.has_tables],
                )
            )
            chunk_id += 1

        start = best_end

    logger.info(
        "Created %d chunks for %s (%d pages)",
        len(chunks),
        analysis.filename,
        total,
    )
    return chunks


def _pick_strategy(pages: list, start: int, end: int) -> str:
    """Choose conversion strategy based on page types in the range."""
    scanned = sum(1 for p in pages if p.page_type == "scanned")
    total = len(pages)
    if total == 0:
        return "page_images"
    if scanned / total > 0.5:
        return "page_images"
    return "page_images"  # always use images for best quality with Vietnamese docs


def _estimate_chunk_tokens(
    analysis: PDFAnalysis, start: int, end: int, strategy: str
) -> int:
    pages = analysis.pages[start:end]
    if strategy == "page_images":
        return len(pages) * TOKENS_PER_IMAGE_PAGE
    text_tokens = sum(p.text_length // 250 for p in pages)
    layout_tokens = len(pages) * 258
    return text_tokens + layout_tokens
