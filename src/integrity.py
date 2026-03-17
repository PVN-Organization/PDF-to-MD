"""Integrity checks: verify page coverage and article (Điều) coverage.

Provides functions to detect missing pages and missing articles in the
converted markdown, and to convert those missing pages via Gemini to
fill the gaps.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from google import genai
from google.genai import types

from .config import Config
from .converter import ChunkResult, _call_gemini_sync, _strip_markdown_fences
from .prompts import SYSTEM_PROMPT, build_chunk_prompt
from .renderer import RenderResult

logger = logging.getLogger(__name__)


@dataclass
class IntegrityReport:
    total_pages: int
    covered_pages: int
    missing_pages: list[int] = field(default_factory=list)
    pdf_articles: list[int] = field(default_factory=list)
    md_articles: list[int] = field(default_factory=list)
    missing_articles: list[int] = field(default_factory=list)
    article_page_map: dict[int, list[int]] = field(default_factory=dict)
    pages_to_convert: list[int] = field(default_factory=list)
    converted_count: int = 0


def check_page_coverage(
    chunk_results: list[ChunkResult],
    total_pages: int,
) -> set[int]:
    """Return the set of page numbers (0-based) NOT covered by any successful chunk."""
    covered: set[int] = set()
    for cr in chunk_results:
        if cr.success:
            for p in range(cr.start_page, cr.end_page):
                covered.add(p)
    all_pages = set(range(total_pages))
    return all_pages - covered


_DIEU_PDF_RE = re.compile(
    r"(?:"
    r"Điều"                    # correct Unicode
    r"|Đi[eêềếểễệ]u"         # partial diacritics
    r"|Di[€~]u"               # common PDF encoding artifacts
    r"|[fF]\)i[€~eêều]u?"    # f) prefix = Đ in some fonts
    r"|Dieu"                   # no diacritics at all
    r")"
    r"\s+(\d+)",
    re.IGNORECASE,
)


def extract_articles_from_pdf(pdf_path: str | Path) -> dict[int, list[int]]:
    """Scan every PDF page for 'Điều N' patterns.

    Handles common Vietnamese PDF encoding artifacts where PyMuPDF
    returns garbled text (Di€u, Di~u, f)i~u, Dieu, etc.).
    Returns mapping {article_number: [page_num_0based, ...]}.
    """
    doc = fitz.open(str(pdf_path))
    articles: dict[int, list[int]] = {}
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        for m in _DIEU_PDF_RE.finditer(text):
            num = int(m.group(1))
            articles.setdefault(num, [])
            if page_num not in articles[num]:
                articles[num].append(page_num)
    doc.close()
    return articles


def extract_articles_from_markdown(markdown: str) -> set[int]:
    """Return set of article numbers found in the markdown text."""
    return {int(m.group(1)) for m in re.finditer(r"Điều\s+(\d+)", markdown)}


def check_article_coverage(
    pdf_path: str | Path,
    markdown: str,
) -> tuple[set[int], dict[int, list[int]]]:
    """Compare articles in PDF vs markdown.

    Filters out unreasonable article numbers (false positives from
    garbled PDF text) by capping at 2x the highest markdown article.
    Returns (missing_article_numbers, article_page_map).
    """
    article_page_map = extract_articles_from_pdf(pdf_path)
    md_articles = extract_articles_from_markdown(markdown)

    max_md = max(md_articles) if md_articles else 0
    max_reasonable = max(max_md * 2, 100)
    pdf_articles = {a for a in article_page_map if a <= max_reasonable}

    spurious = set(article_page_map) - pdf_articles
    if spurious:
        logger.debug("Integrity — Bỏ qua %d article number bất thường: %s",
                      len(spurious), sorted(spurious))

    missing = pdf_articles - md_articles
    return missing, {k: v for k, v in article_page_map.items() if k in pdf_articles}


def convert_missing_pages(
    pdf_path: str | Path,
    markdown: str,
    missing_pages: set[int],
    config: Config,
    render_result: RenderResult,
    doc_title: str = "",
    cache_dir: Path | None = None,
) -> str:
    """Convert specific missing pages via Gemini and append to the markdown.

    Uses the main conversion model (config.gemini_model).
    Returns the updated markdown with appended content.
    """
    if not missing_pages:
        return markdown

    pdf_path = Path(pdf_path)
    client = genai.Client(api_key=config.gemini_api_key)
    page_image_map = {rp.page_num: rp.image_path for rp in render_result.pages}
    doc = fitz.open(str(pdf_path))

    appended_parts: list[str] = []

    for page_num in sorted(missing_pages):
        logger.info("Integrity: converting missing page %d...", page_num + 1)

        if cache_dir:
            cache_file = cache_dir / f"integrity_page_{page_num:04d}.md"
            if cache_file.exists():
                cached = cache_file.read_text(encoding="utf-8")
                if cached.strip():
                    logger.info("  Using cached integrity result for page %d", page_num + 1)
                    appended_parts.append(cached)
                    continue

        contents: list = []

        img_path = page_image_map.get(page_num)
        if img_path and img_path.exists():
            img_data = img_path.read_bytes()
        else:
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=config.render_dpi)
            img_data = pix.tobytes("png")

        contents.append(
            types.Part.from_bytes(data=img_data, mime_type="image/png")
        )

        prompt_text = build_chunk_prompt(
            chunk_id=0,
            total_chunks=1,
            start_page=page_num,
            end_page=page_num + 1,
            doc_title=doc_title or pdf_path.stem,
        )
        contents.append(prompt_text)

        try:
            result_md = _call_gemini_sync(
                client, config.gemini_model, contents,
                system_instruction=SYSTEM_PROMPT,
            )
            result_md = _strip_markdown_fences(result_md)
            appended_parts.append(result_md)
            logger.info("  Page %d converted: %d chars", page_num + 1, len(result_md))

            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = cache_dir / f"integrity_page_{page_num:04d}.md"
                cache_file.write_text(result_md, encoding="utf-8")

        except Exception as e:
            logger.error("  Failed to convert page %d: %s", page_num + 1, e)

    doc.close()

    if appended_parts:
        separator = "\n\n---\n\n"
        appended_md = separator.join(appended_parts)
        markdown = markdown.rstrip() + separator + appended_md

    return markdown


def run_integrity_check(
    pdf_path: str | Path,
    markdown: str,
    chunk_results: list[ChunkResult],
    config: Config,
    render_result: RenderResult,
    total_pages: int,
    doc_title: str = "",
    cache_dir: Path | None = None,
    auto_fix: bool = True,
) -> tuple[str, IntegrityReport]:
    """Run both page coverage and article coverage checks.

    If auto_fix is True, missing pages are converted via Gemini and appended.
    Returns (updated_markdown, report).
    """
    report = IntegrityReport(total_pages=total_pages, covered_pages=0)

    # --- 1. Page coverage (highest priority) ---
    missing_page_set = check_page_coverage(chunk_results, total_pages)
    covered = total_pages - len(missing_page_set)
    report.covered_pages = covered
    report.missing_pages = sorted(missing_page_set)

    logger.info("Integrity — Page coverage: %d/%d trang covered", covered, total_pages)
    if missing_page_set:
        logger.warning(
            "Integrity — Thiếu %d trang: %s",
            len(missing_page_set),
            ", ".join(str(p + 1) for p in sorted(missing_page_set)),
        )

    # --- 2. Article coverage ---
    missing_articles, article_page_map = check_article_coverage(pdf_path, markdown)
    pdf_article_nums = sorted(article_page_map.keys())
    md_article_nums = sorted(extract_articles_from_markdown(markdown))

    report.pdf_articles = pdf_article_nums
    report.md_articles = md_article_nums
    report.missing_articles = sorted(missing_articles)
    report.article_page_map = {k: v for k, v in article_page_map.items() if k in missing_articles}

    if pdf_article_nums:
        logger.info(
            "Integrity — Article coverage: %d/%d Điều trong markdown (PDF có %s)",
            len(md_article_nums), len(pdf_article_nums),
            ", ".join(f"Đ.{a}" for a in pdf_article_nums),
        )
    else:
        logger.info(
            "Integrity — Article check: PDF scanned (không trích xuất được text), "
            "tìm thấy %d Điều trong markdown",
            len(md_article_nums),
        )
    if missing_articles:
        logger.warning(
            "Integrity — Thiếu %d Điều: %s",
            len(missing_articles),
            ", ".join(f"Điều {a}" for a in sorted(missing_articles)),
        )
        for a in sorted(missing_articles):
            pages = article_page_map.get(a, [])
            logger.info("  Điều %d nằm ở trang: %s", a, ", ".join(str(p + 1) for p in pages))

    # --- 3. Determine pages to convert ---
    pages_to_convert: set[int] = set()

    if missing_page_set:
        pages_to_convert.update(missing_page_set)

    if missing_articles:
        for a in missing_articles:
            for p in article_page_map.get(a, []):
                if p not in pages_to_convert:
                    already_covered = p not in missing_page_set
                    if already_covered:
                        pages_to_convert.add(p)
                        logger.info(
                            "  Adding page %d for missing Điều %d (page was covered but article missing)",
                            p + 1, a,
                        )

    report.pages_to_convert = sorted(pages_to_convert)

    # --- 4. Convert missing pages ---
    if pages_to_convert and auto_fix:
        logger.info(
            "Integrity — Converting %d trang thiếu: %s",
            len(pages_to_convert),
            ", ".join(str(p + 1) for p in sorted(pages_to_convert)),
        )
        markdown = convert_missing_pages(
            pdf_path, markdown, pages_to_convert, config,
            render_result, doc_title, cache_dir,
        )
        report.converted_count = len(pages_to_convert)
        logger.info("Integrity — Đã convert và append %d trang", report.converted_count)

        # Re-check articles after append
        still_missing, _ = check_article_coverage(pdf_path, markdown)
        if still_missing:
            logger.warning(
                "Integrity — Sau khi append, vẫn thiếu %d Điều: %s",
                len(still_missing),
                ", ".join(f"Điều {a}" for a in sorted(still_missing)),
            )
        else:
            logger.info("Integrity — Sau khi append: tất cả Điều đã có trong markdown")
    elif not pages_to_convert:
        logger.info("Integrity — Không có trang nào cần bổ sung")

    return markdown, report
