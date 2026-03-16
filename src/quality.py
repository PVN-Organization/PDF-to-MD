from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from google import genai
from google.genai import types

from .analyzer import PDFAnalysis
from .config import Config
from .prompts import QUALITY_REVIEW_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    filename: str
    total_pages: int
    text_completeness: float
    structure_score: float
    table_score: float
    vietnamese_score: float
    ai_review: dict | None = None
    overall_score: float = 0.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "total_pages": self.total_pages,
            "text_completeness": round(self.text_completeness, 3),
            "structure_score": round(self.structure_score, 2),
            "table_score": round(self.table_score, 2),
            "vietnamese_score": round(self.vietnamese_score, 2),
            "ai_review": self.ai_review,
            "overall_score": round(self.overall_score, 2),
            "issues": self.issues,
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def check_quality(
    pdf_path: str | Path,
    markdown: str,
    analysis: PDFAnalysis,
    config: Config,
    run_ai_review: bool = True,
) -> QualityReport:
    """Run all quality checks and return a comprehensive report."""
    report = QualityReport(
        filename=analysis.filename,
        total_pages=analysis.total_pages,
        text_completeness=0.0,
        structure_score=0.0,
        table_score=0.0,
        vietnamese_score=0.0,
    )

    report.text_completeness = _check_text_completeness(pdf_path, markdown, report)
    report.structure_score = _check_structure(markdown, report)
    report.table_score = _check_tables(pdf_path, markdown, analysis, report)
    report.vietnamese_score = _check_vietnamese(markdown, report)

    if run_ai_review and config.gemini_api_key:
        report.ai_review = _run_ai_review(pdf_path, markdown, config, report)

    # Weighted overall score
    weights = {
        "text": 0.30,
        "structure": 0.20,
        "table": 0.15,
        "vietnamese": 0.15,
        "ai": 0.20,
    }
    ai_score = report.ai_review.get("overall", 7) if report.ai_review else 7
    report.overall_score = (
        report.text_completeness * 10 * weights["text"]
        + report.structure_score * weights["structure"]
        + report.table_score * weights["table"]
        + report.vietnamese_score * weights["vietnamese"]
        + ai_score * weights["ai"]
    )

    logger.info(
        "Quality report for %s: overall=%.1f, completeness=%.1f%%, structure=%.1f, "
        "tables=%.1f, vietnamese=%.1f",
        analysis.filename,
        report.overall_score,
        report.text_completeness * 100,
        report.structure_score,
        report.table_score,
        report.vietnamese_score,
    )
    return report


def _check_text_completeness(
    pdf_path: str | Path, markdown: str, report: QualityReport
) -> float:
    """Compare extracted text volume from PDF vs markdown output."""
    doc = fitz.open(str(pdf_path))
    pdf_text = ""
    for page in doc:
        pdf_text += page.get_text("text")
    doc.close()

    pdf_chars = len(pdf_text.strip())
    md_chars = len(re.sub(r"[#*|\-`>\[\]()!]", "", markdown).strip())

    if pdf_chars == 0:
        # Likely fully scanned — can't compare text
        return 0.8

    ratio = min(md_chars / pdf_chars, 1.5)

    if ratio < 0.5:
        report.issues.append(
            f"Nội dung markdown ({md_chars} ký tự) ít hơn nhiều so với PDF "
            f"({pdf_chars} ký tự) — có thể thiếu nội dung"
        )
    elif ratio > 1.3:
        report.issues.append(
            "Markdown dài hơn nhiều so với text PDF gốc — có thể có nội dung thừa"
        )

    return min(ratio, 1.0)


def _check_structure(markdown: str, report: QualityReport) -> float:
    """Check heading hierarchy and structural elements."""
    score = 10.0

    headings = re.findall(r"^(#{1,6})\s+(.+)$", markdown, re.MULTILINE)
    if not headings:
        report.issues.append("Không tìm thấy heading nào trong markdown")
        return 3.0

    # Check for "Điều" headings
    dieu_count = len(re.findall(r"^#{1,6}\s+.*Điều\s+\d+", markdown, re.MULTILINE))
    if dieu_count == 0:
        # Not all docs have "Điều" — only flag if the PDF text mentions it
        dieu_in_md = markdown.lower().count("điều")
        if dieu_in_md > 3:
            report.issues.append(
                f'Tìm thấy {dieu_in_md} lần từ "Điều" nhưng không có heading Điều nào'
            )
            score -= 2

    # Check heading levels don't skip
    levels = [len(h[0]) for h in headings]
    for i in range(1, len(levels)):
        if levels[i] > levels[i - 1] + 1:
            score -= 0.5
            if score < 5:
                break

    # Check for ordered lists
    list_items = re.findall(r"^\d+\.\s+", markdown, re.MULTILINE)
    if len(list_items) < 3 and len(markdown) > 2000:
        report.issues.append("Ít danh sách có thứ tự — có thể cần kiểm tra khoản/điểm")
        score -= 1

    return max(score, 1.0)


def _check_tables(
    pdf_path: str | Path,
    markdown: str,
    analysis: PDFAnalysis,
    report: QualityReport,
) -> float:
    """Check table integrity."""
    pdf_tables = sum(1 for p in analysis.pages if p.has_tables)
    md_tables = len(re.findall(r"^\|.+\|$\n^\|[-:\s|]+\|$", markdown, re.MULTILINE))

    if pdf_tables == 0 and md_tables == 0:
        return 10.0  # No tables expected or found

    if pdf_tables > 0 and md_tables == 0:
        report.issues.append(
            f"PDF có ~{pdf_tables} trang chứa bảng nhưng markdown không có bảng nào"
        )
        return 3.0

    # Check table formatting
    table_blocks = re.findall(
        r"(\|.+\|\n\|[-:\s|]+\|\n(?:\|.+\|\n)*)", markdown, re.MULTILINE
    )
    malformed = 0
    for table in table_blocks:
        rows = table.strip().split("\n")
        col_counts = [row.count("|") for row in rows]
        if len(set(col_counts)) > 1:
            malformed += 1

    if malformed > 0:
        report.issues.append(f"{malformed} bảng có số cột không đồng nhất giữa các dòng")

    score = 10.0
    if malformed > 0:
        score -= min(malformed * 2, 5)

    return max(score, 1.0)


def _check_vietnamese(markdown: str, report: QualityReport) -> float:
    """Check Vietnamese text quality — diacritics, encoding issues."""
    score = 10.0

    # Check for common encoding artifacts
    bad_patterns = [
        (r"Ã¡|Ã |Ã¢|Ã£|Ã©|Ã¨|Ãª|Ã³|Ã²|Ã´|Ã¹|Ãº|Ã½|ƒ", "ký tự mojibake encoding bị lỗi"),
        (r"\\u[0-9a-fA-F]{4}", "Unicode escape sequences chưa decode"),
        (r"\?{3,}", "Chuỗi ??? — có thể ký tự không nhận được"),
    ]
    for pattern, desc in bad_patterns:
        matches = re.findall(pattern, markdown)
        if matches:
            report.issues.append(f"Phát hiện {len(matches)} lần {desc}")
            score -= 2

    # Check Vietnamese diacritics are present
    vn_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
    found_vn = sum(1 for c in markdown.lower() if c in vn_chars)
    total_alpha = sum(1 for c in markdown if c.isalpha())

    if total_alpha > 100:
        vn_ratio = found_vn / total_alpha
        if vn_ratio < 0.05:
            report.issues.append(
                f"Tỷ lệ ký tự tiếng Việt thấp ({vn_ratio:.1%}) — "
                "có thể mất dấu thanh"
            )
            score -= 3

    return max(score, 1.0)


def _run_ai_review(
    pdf_path: str | Path,
    markdown: str,
    config: Config,
    report: QualityReport,
) -> dict | None:
    """Use Gemini to review a sample of pages."""
    try:
        client = genai.Client(api_key=config.gemini_api_key)
        doc = fitz.open(str(pdf_path))

        total_pages = doc.page_count
        sample_count = min(config.quality_sample_pages, total_pages)
        sample_pages = sorted(random.sample(range(total_pages), sample_count))

        # Render sample pages as inline image parts
        image_parts = []
        for p in sample_pages:
            page = doc.load_page(p)
            pix = page.get_pixmap(dpi=150)
            image_parts.append(
                types.Part.from_bytes(data=pix.tobytes("png"), mime_type="image/png")
            )

        doc.close()

        # Get corresponding markdown section (approximate by splitting)
        md_lines = markdown.split("\n")
        lines_per_page = max(len(md_lines) // max(total_pages, 1), 10)
        md_sample = ""
        for p in sample_pages:
            start = p * lines_per_page
            end = min(start + lines_per_page, len(md_lines))
            md_sample += "\n".join(md_lines[start:end]) + "\n\n"

        prompt = QUALITY_REVIEW_PROMPT.replace("{markdown_content}", md_sample[:5000])

        contents = image_parts + [prompt]
        response = client.models.generate_content(
            model=config.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        response_text = response.text
        if response_text is None:
            logger.warning("AI review: Gemini returned None response")
            return None
        response_text = response_text.strip()

        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            review = json.loads(json_match.group())
            return review

        logger.warning("Could not parse AI review response as JSON")
        return None

    except Exception as e:
        logger.warning("AI review failed: %s", e)
        report.issues.append(f"AI review thất bại: {e}")
        return None
