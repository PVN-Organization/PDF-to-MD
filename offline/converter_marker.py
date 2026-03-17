from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MarkerResult:
    filename: str
    markdown: str
    images: dict
    elapsed_seconds: float
    page_count: int


def _ensure_device(device: str) -> None:
    """Set TORCH_DEVICE env var before marker imports torch."""
    os.environ.setdefault("TORCH_DEVICE", device)


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

    import fitz
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
    )


def convert_batch(
    pdf_dir: Path,
    device: str = "mps",
    languages: tuple[str, ...] = ("vi", "en"),
    force_ocr: bool = False,
) -> list[MarkerResult]:
    """Convert all PDFs in a directory."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return []

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
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
    )

    results: list[MarkerResult] = []
    for i, pdf_file in enumerate(pdf_files, 1):
        logger.info("[%d/%d] Converting %s...", i, len(pdf_files), pdf_file.name)
        start = time.time()
        try:
            rendered = converter(str(pdf_file))
            text, _, images = text_from_rendered(rendered)
            elapsed = time.time() - start

            import fitz
            doc = fitz.open(str(pdf_file))
            page_count = doc.page_count
            doc.close()

            results.append(MarkerResult(
                filename=pdf_file.name,
                markdown=text,
                images=images or {},
                elapsed_seconds=elapsed,
                page_count=page_count,
            ))
            logger.info("  Done: %d pages, %d chars in %.1fs", page_count, len(text), elapsed)
        except Exception as e:
            logger.error("  Failed to convert %s: %s", pdf_file.name, e)

    return results
