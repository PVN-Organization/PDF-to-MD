from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageInfo:
    page_num: int
    width: float
    height: float
    has_text: bool
    text_length: int
    text_density: float
    has_images: bool
    image_count: int
    has_tables: bool
    page_type: str  # "text" | "scanned" | "mixed"


@dataclass
class PDFAnalysis:
    filepath: Path
    filename: str
    total_pages: int
    file_size_mb: float
    pages: list[PageInfo] = field(default_factory=list)
    estimated_tokens: int = 0

    @property
    def scanned_ratio(self) -> float:
        if not self.pages:
            return 0.0
        scanned = sum(1 for p in self.pages if p.page_type == "scanned")
        return scanned / len(self.pages)

    @property
    def has_tables(self) -> bool:
        return any(p.has_tables for p in self.pages)

    @property
    def has_images(self) -> bool:
        return any(p.has_images for p in self.pages)


def _detect_tables_heuristic(page: fitz.Page) -> bool:
    """Detect tables by looking for grid-like line drawings."""
    drawings = page.get_drawings()
    h_lines = 0
    v_lines = 0
    for d in drawings:
        for item in d["items"]:
            if item[0] == "l":  # line
                p1, p2 = item[1], item[2]
                dx = abs(p2.x - p1.x)
                dy = abs(p2.y - p1.y)
                if dy < 2 and dx > 30:
                    h_lines += 1
                elif dx < 2 and dy > 30:
                    v_lines += 1
    return h_lines >= 3 and v_lines >= 2


def _classify_page(text_length: int, text_density: float, has_images: bool) -> str:
    if text_density < 0.02 and text_length < 50:
        return "scanned"
    if has_images and text_density < 0.3:
        return "mixed"
    return "text"


def analyze_page(page: fitz.Page, page_num: int) -> PageInfo:
    text = page.get_text("text")
    text_length = len(text.strip())
    rect = page.rect
    page_area = rect.width * rect.height
    text_density = text_length / page_area if page_area > 0 else 0.0

    images = page.get_images(full=True)
    has_images = len(images) > 0

    has_tables = _detect_tables_heuristic(page)

    page_type = _classify_page(text_length, text_density, has_images)

    return PageInfo(
        page_num=page_num,
        width=rect.width,
        height=rect.height,
        has_text=text_length > 10,
        text_length=text_length,
        text_density=round(text_density, 4),
        has_images=has_images,
        image_count=len(images),
        has_tables=has_tables,
        page_type=page_type,
    )


def analyze_pdf(pdf_path: str | Path) -> PDFAnalysis:
    pdf_path = Path(pdf_path)
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)

    doc = fitz.open(str(pdf_path))
    pages: list[PageInfo] = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        info = analyze_page(page, i)
        pages.append(info)

    total_text = sum(p.text_length for p in pages)
    # ~4 tokens per 1KB text + ~258 tokens per page for layout
    estimated_tokens = (total_text // 250) + (doc.page_count * 258)

    analysis = PDFAnalysis(
        filepath=pdf_path,
        filename=pdf_path.name,
        total_pages=doc.page_count,
        file_size_mb=round(file_size_mb, 2),
        pages=pages,
        estimated_tokens=estimated_tokens,
    )

    doc.close()
    logger.info(
        "Analyzed %s: %d pages, %.1f MB, %.0f%% scanned, ~%d tokens",
        pdf_path.name,
        analysis.total_pages,
        analysis.file_size_mb,
        analysis.scanned_ratio * 100,
        analysis.estimated_tokens,
    )
    return analysis
