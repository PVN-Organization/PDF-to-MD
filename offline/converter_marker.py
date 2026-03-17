from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

_VN_CHARS = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
                "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ")


@dataclass
class MarkerResult:
    filename: str
    markdown: str
    images: dict
    elapsed_seconds: float
    page_count: int
    strategy_used: str = "marker"


def _ensure_device(device: str) -> None:
    """Set TORCH_DEVICE env var before marker imports torch."""
    os.environ.setdefault("TORCH_DEVICE", device)


def _assess_vietnamese_quality(text: str) -> float:
    """Return ratio of Vietnamese diacritical characters to total alpha chars."""
    alpha = sum(1 for c in text if c.isalpha())
    if alpha < 50:
        return 0.0
    vn = sum(1 for c in text if c in _VN_CHARS)
    return vn / alpha


def _is_page_scanned(page: fitz.Page) -> bool:
    """Heuristic: a page is 'scanned' if it has very little extractable text."""
    text = page.get_text("text").strip()
    return len(text) < 50


def _pdftext_extract(pdf_path: Path) -> tuple[str, int, list[bool]]:
    """Extract text directly from PDF using PyMuPDF (no OCR).
    Returns (full_text, page_count, list of is_scanned per page).
    """
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    pages_text: list[str] = []
    scanned_flags: list[bool] = []

    for page in doc:
        text = page.get_text("text")
        is_scanned = len(text.strip()) < 50
        scanned_flags.append(is_scanned)
        pages_text.append(text)

    doc.close()
    full_text = "\n\n".join(pages_text)
    return full_text, page_count, scanned_flags


def _pdftext_to_markdown(pdf_path: Path) -> tuple[str, int, list[bool]]:
    """Extract text and apply basic markdown formatting from PDF structure."""
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    md_pages: list[str] = []
    scanned_flags: list[bool] = []

    for page_num in range(page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")

        if len(text.strip()) < 50:
            scanned_flags.append(True)
            md_pages.append("")
            continue

        scanned_flags.append(False)

        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        lines_md: list[str] = []

        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue

                max_size = max(s["size"] for s in spans)
                is_bold = all("bold" in s.get("font", "").lower() or
                              (s.get("flags", 0) & 2**4) for s in spans)

                if max_size >= 16 and is_bold:
                    lines_md.append(f"\n# {line_text}\n")
                elif max_size >= 14 and is_bold:
                    lines_md.append(f"\n## {line_text}\n")
                elif max_size >= 12 and is_bold:
                    lines_md.append(f"\n### {line_text}\n")
                elif is_bold:
                    lines_md.append(f"**{line_text}**")
                else:
                    lines_md.append(line_text)

        md_pages.append("\n".join(lines_md))

    doc.close()
    full_md = "\n\n---\n\n".join(p for p in md_pages if p.strip())
    return full_md, page_count, scanned_flags


def convert_single(
    pdf_path: Path,
    device: str = "mps",
    languages: tuple[str, ...] = ("vi", "en"),
    force_ocr: bool = False,
) -> MarkerResult:
    """Convert a single PDF to markdown using marker-pdf."""
    _ensure_device(device)

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.config.parser import ConfigParser

    config = {
        "output_format": "markdown",
        "languages": ",".join(languages),
        "disable_image_extraction": True,
    }
    if force_ocr:
        config["force_ocr"] = True

    config_parser = ConfigParser(config)

    logger.info("Loading marker models (device=%s)...", device)
    start = time.time()

    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )

    model_time = time.time() - start
    logger.info("Models loaded in %.1fs", model_time)

    logger.info("Converting %s...", pdf_path.name)
    start = time.time()

    rendered = converter(str(pdf_path))
    text, _, images = text_from_rendered(rendered)

    elapsed = time.time() - start

    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    doc.close()

    logger.info(
        "Converted %s: %d pages, %d chars in %.1fs",
        pdf_path.name, page_count, len(text), elapsed,
    )

    return MarkerResult(
        filename=pdf_path.name,
        markdown=text,
        images=images or {},
        elapsed_seconds=elapsed,
        page_count=page_count,
        strategy_used="marker",
    )


def convert_single_smart(
    pdf_path: Path,
    device: str = "mps",
    languages: tuple[str, ...] = ("vi", "en"),
    force_ocr: bool = False,
    strategy: str = "auto",
) -> MarkerResult:
    """Smart convert: choose best strategy per-file for Vietnamese quality.

    Strategies:
    - "auto": try pdftext first, use marker OCR only for scanned pages
    - "marker": always use marker-pdf (original behavior)
    - "pdftext": always use direct text extraction (no OCR)
    - "hybrid": extract text pages with pdftext, OCR scanned pages with marker
    """
    pdf_path = Path(pdf_path)
    start_all = time.time()

    if strategy == "marker":
        return convert_single(pdf_path, device, languages, force_ocr)

    logger.info("Smart convert (%s): extracting text from %s...", strategy, pdf_path.name)

    pdftext_md, page_count, scanned_flags = _pdftext_to_markdown(pdf_path)
    scanned_count = sum(scanned_flags)
    text_count = page_count - scanned_count

    logger.info(
        "  pdftext: %d pages text-extractable, %d scanned, %d chars",
        text_count, scanned_count, len(pdftext_md),
    )

    pdftext_vn_ratio = _assess_vietnamese_quality(pdftext_md)
    logger.info("  pdftext Vietnamese quality: %.1f%%", pdftext_vn_ratio * 100)

    if strategy == "pdftext":
        elapsed = time.time() - start_all
        logger.info("Using pdftext result (strategy=pdftext): %d chars", len(pdftext_md))
        return MarkerResult(
            filename=pdf_path.name,
            markdown=pdftext_md,
            images={},
            elapsed_seconds=elapsed,
            page_count=page_count,
            strategy_used="pdftext",
        )

    # "auto" or "hybrid": decide based on quality
    if pdftext_vn_ratio >= 0.05 and scanned_count == 0:
        # Good Vietnamese text, no scanned pages => use pdftext directly
        elapsed = time.time() - start_all
        logger.info(
            "Auto: using pdftext (%.1f%% VN, 0 scanned) — %d chars in %.1fs",
            pdftext_vn_ratio * 100, len(pdftext_md), elapsed,
        )
        return MarkerResult(
            filename=pdf_path.name,
            markdown=pdftext_md,
            images={},
            elapsed_seconds=elapsed,
            page_count=page_count,
            strategy_used="pdftext",
        )

    if scanned_count > 0 and strategy in ("auto", "hybrid"):
        logger.info(
            "Hybrid: %d scanned pages detected, running marker OCR for full document...",
            scanned_count,
        )
        marker_result = convert_single(pdf_path, device, languages, force_ocr)
        marker_vn_ratio = _assess_vietnamese_quality(marker_result.markdown)
        logger.info("  marker Vietnamese quality: %.1f%%", marker_vn_ratio * 100)

        if pdftext_vn_ratio > marker_vn_ratio * 1.5 and pdftext_vn_ratio >= 0.03:
            logger.info(
                "  pdftext has better VN quality (%.1f%% vs %.1f%%), using pdftext",
                pdftext_vn_ratio * 100, marker_vn_ratio * 100,
            )
            elapsed = time.time() - start_all
            return MarkerResult(
                filename=pdf_path.name,
                markdown=pdftext_md,
                images={},
                elapsed_seconds=elapsed,
                page_count=page_count,
                strategy_used="pdftext+fallback",
            )

        elapsed = time.time() - start_all
        marker_result.elapsed_seconds = elapsed
        marker_result.strategy_used = "marker(hybrid)"
        return marker_result

    # Fallback for "auto" when pdftext VN quality is low and no scanned pages:
    # this means the PDF itself has poor text (e.g. embedded fonts without unicode mapping)
    # Try marker anyway
    logger.info(
        "Auto: pdftext VN quality low (%.1f%%), trying marker OCR...",
        pdftext_vn_ratio * 100,
    )
    marker_result = convert_single(pdf_path, device, languages, force_ocr)
    marker_vn_ratio = _assess_vietnamese_quality(marker_result.markdown)
    logger.info("  marker Vietnamese quality: %.1f%%", marker_vn_ratio * 100)

    if pdftext_vn_ratio > marker_vn_ratio and pdftext_vn_ratio >= 0.03:
        logger.info("  pdftext still better, using pdftext result")
        elapsed = time.time() - start_all
        return MarkerResult(
            filename=pdf_path.name,
            markdown=pdftext_md,
            images={},
            elapsed_seconds=elapsed,
            page_count=page_count,
            strategy_used="pdftext(auto-fallback)",
        )

    elapsed = time.time() - start_all
    marker_result.elapsed_seconds = elapsed
    chosen = "marker" if marker_vn_ratio >= pdftext_vn_ratio else "pdftext"
    if chosen == "pdftext":
        return MarkerResult(
            filename=pdf_path.name,
            markdown=pdftext_md,
            images={},
            elapsed_seconds=elapsed,
            page_count=page_count,
            strategy_used="pdftext(auto-best)",
        )

    marker_result.strategy_used = "marker(auto-best)"
    return marker_result


def convert_batch(
    pdf_dir: Path,
    device: str = "mps",
    languages: tuple[str, ...] = ("vi", "en"),
    force_ocr: bool = False,
    strategy: str = "auto",
) -> list[MarkerResult]:
    """Convert all PDFs in a directory using smart strategy."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return []

    results: list[MarkerResult] = []
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info("[%d/%d] Converting %s (strategy=%s)...", i, len(pdf_files), pdf_file.name, strategy)
        try:
            result = convert_single_smart(
                pdf_file, device=device, languages=languages,
                force_ocr=force_ocr, strategy=strategy,
            )
            results.append(result)
            logger.info(
                "  Done: %d pages, %d chars, strategy=%s, %.1fs",
                result.page_count, len(result.markdown),
                result.strategy_used, result.elapsed_seconds,
            )
        except Exception as e:
            logger.error("  Failed to convert %s: %s", pdf_file.name, e)

    return results
