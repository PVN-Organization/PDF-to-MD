from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from .config import Config
from .planner import ChunkPlan

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedImage:
    page_num: int
    image_index: int
    xref: int
    width: int
    height: int
    saved_path: Path


@dataclass
class RenderedPage:
    page_num: int
    image_path: Path


@dataclass
class RenderResult:
    pages: list[RenderedPage] = field(default_factory=list)
    embedded_images: list[EmbeddedImage] = field(default_factory=list)


def render_pages(
    pdf_path: str | Path,
    chunks: list[ChunkPlan],
    config: Config,
    output_subdir: str = "",
) -> RenderResult:
    """Render PDF pages to images for Gemini to read."""
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    stem = pdf_path.stem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)

    page_dir = config.temp_dir / safe_stem / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)

    pages_needed: set[int] = set()
    for chunk in chunks:
        for p in range(chunk.start_page, chunk.end_page):
            pages_needed.add(p)

    result = RenderResult()

    for page_num in sorted(pages_needed):
        page = doc.load_page(page_num)

        pix = page.get_pixmap(dpi=config.render_dpi)
        img_path = page_dir / f"page_{page_num:04d}.png"
        pix.save(str(img_path))

        result.pages.append(RenderedPage(page_num=page_num, image_path=img_path))

    doc.close()
    logger.info(
        "Rendered %d pages for %s",
        len(result.pages),
        pdf_path.name,
    )
    return result
