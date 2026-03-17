"""Offline quality check — reuses heuristic checks from src/quality.py
and optionally uses Ollama for AI-based review.
"""
from __future__ import annotations

import base64
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

QUALITY_REVIEW_PROMPT_OLLAMA = """\
Bạn là chuyên gia kiểm tra chất lượng chuyển đổi văn bản pháp quy.

So sánh hình ảnh trang gốc với nội dung Markdown đã chuyển đổi bên dưới.

Đánh giá theo thang điểm 1-10 cho từng tiêu chí:
1. completeness: Có thiếu sót nội dung nào không?
2. accuracy: Nội dung tiếng Việt có đúng không? (dấu, chính tả)
3. structure: Heading, list, table có đúng cấp bậc không?
4. tables: Các bảng có đầy đủ dòng/cột không?

Trả về JSON (KHÔNG giải thích):
{"completeness": <1-10>, "accuracy": <1-10>, "structure": <1-10>, "tables": <1-10>, "overall": <1-10>, "issues": ["mô tả vấn đề 1"]}

Markdown cần kiểm tra:
"""


@dataclass
class QualityReportOffline:
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


def _check_text_completeness(pdf_path: str | Path, markdown: str, report: QualityReportOffline) -> float:
    doc = fitz.open(str(pdf_path))
    pdf_text = ""
    for page in doc:
        pdf_text += page.get_text("text")
    doc.close()

    pdf_chars = len(pdf_text.strip())
    md_chars = len(re.sub(r"[#*|\-`>\[\]()!]", "", markdown).strip())

    if pdf_chars == 0:
        return 0.8

    ratio = min(md_chars / pdf_chars, 1.5)

    if ratio < 0.5:
        report.issues.append(
            f"Nội dung markdown ({md_chars} ký tự) ít hơn nhiều so với PDF "
            f"({pdf_chars} ký tự) — có thể thiếu nội dung"
        )
    elif ratio > 1.3:
        report.issues.append("Markdown dài hơn nhiều so với text PDF gốc — có thể có nội dung thừa")

    return min(ratio, 1.0)


def _check_structure(markdown: str, report: QualityReportOffline) -> float:
    score = 10.0
    headings = re.findall(r"^(#{1,6})\s+(.+)$", markdown, re.MULTILINE)
    if not headings:
        report.issues.append("Không tìm thấy heading nào trong markdown")
        return 3.0

    dieu_count = len(re.findall(r"^#{1,6}\s+.*Điều\s+\d+", markdown, re.MULTILINE))
    if dieu_count == 0:
        dieu_in_md = markdown.lower().count("điều")
        if dieu_in_md > 3:
            report.issues.append(
                f'Tìm thấy {dieu_in_md} lần từ "Điều" nhưng không có heading Điều nào'
            )
            score -= 2

    levels = [len(h[0]) for h in headings]
    for i in range(1, len(levels)):
        if levels[i] > levels[i - 1] + 1:
            score -= 0.5
            if score < 5:
                break

    list_items = re.findall(r"^\d+\.\s+", markdown, re.MULTILINE)
    if len(list_items) < 3 and len(markdown) > 2000:
        report.issues.append("Ít danh sách có thứ tự — có thể cần kiểm tra khoản/điểm")
        score -= 1

    return max(score, 1.0)


def _check_tables(pdf_path: str | Path, markdown: str, report: QualityReportOffline) -> float:
    md_tables = len(re.findall(r"^\|.+\|$\n^\|[-:\s|]+\|$", markdown, re.MULTILINE))

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


def _check_vietnamese(markdown: str, report: QualityReportOffline) -> float:
    score = 10.0
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

    vn_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
    found_vn = sum(1 for c in markdown.lower() if c in vn_chars)
    total_alpha = sum(1 for c in markdown if c.isalpha())

    if total_alpha > 100:
        vn_ratio = found_vn / total_alpha
        if vn_ratio < 0.05:
            report.issues.append(
                f"Tỷ lệ ký tự tiếng Việt thấp ({vn_ratio:.1%}) — có thể mất dấu thanh"
            )
            score -= 3

    return max(score, 1.0)


def _run_ollama_review(
    pdf_path: str | Path,
    markdown: str,
    model: str,
    base_url: str,
    report: QualityReportOffline,
) -> dict | None:
    try:
        import ollama

        client = ollama.Client(host=base_url)
        doc = fitz.open(str(pdf_path))

        total_pages = doc.page_count
        sample_count = min(3, total_pages)
        sample_pages = sorted(random.sample(range(total_pages), sample_count))

        images_b64: list[str] = []
        for p in sample_pages:
            page = doc.load_page(p)
            pix = page.get_pixmap(dpi=150)
            images_b64.append(base64.b64encode(pix.tobytes("png")).decode())
        doc.close()

        md_lines = markdown.split("\n")
        lines_per_page = max(len(md_lines) // max(total_pages, 1), 10)
        md_sample = ""
        for p in sample_pages:
            start = p * lines_per_page
            end = min(start + lines_per_page, len(md_lines))
            md_sample += "\n".join(md_lines[start:end]) + "\n\n"

        prompt = QUALITY_REVIEW_PROMPT_OLLAMA + md_sample[:3000]

        logger.info("Quality check: gọi Ollama (%s) với %d trang mẫu...", model, sample_count)

        response = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": images_b64[:1],
            }],
            options={"temperature": 0.1},
        )
        text = response.get("message", {}).get("content", "")
        if not text:
            return None

        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            review = json.loads(json_match.group())
            logger.info("Quality AI review: overall=%s", review.get("overall", "?"))
            return review

        logger.warning("Không parse được JSON từ Ollama review response")
        return None

    except ImportError:
        logger.warning("ollama package chưa được cài, bỏ qua AI review")
        return None
    except Exception as e:
        logger.warning("Ollama AI review thất bại: %s", e)
        report.issues.append(f"Ollama AI review thất bại: {e}")
        return None


def check_quality_offline(
    pdf_path: str | Path,
    markdown: str,
    total_pages: int,
    use_ollama: bool = False,
    ollama_model: str = "qwen3-vl:8b",
    ollama_base_url: str = "http://localhost:11434",
) -> QualityReportOffline:
    """Run quality checks (heuristic + optional Ollama AI review)."""
    pdf_path = Path(pdf_path)
    report = QualityReportOffline(
        filename=pdf_path.name,
        total_pages=total_pages,
        text_completeness=0.0,
        structure_score=0.0,
        table_score=0.0,
        vietnamese_score=0.0,
    )

    logger.info("Quality check: bắt đầu kiểm tra %s (%d trang)", pdf_path.name, total_pages)

    report.text_completeness = _check_text_completeness(pdf_path, markdown, report)
    logger.info("  Text completeness: %.1f%%", report.text_completeness * 100)

    report.structure_score = _check_structure(markdown, report)
    logger.info("  Structure score: %.1f/10", report.structure_score)

    report.table_score = _check_tables(pdf_path, markdown, report)
    logger.info("  Table score: %.1f/10", report.table_score)

    report.vietnamese_score = _check_vietnamese(markdown, report)
    logger.info("  Vietnamese score: %.1f/10", report.vietnamese_score)

    if use_ollama:
        report.ai_review = _run_ollama_review(
            pdf_path, markdown, ollama_model, ollama_base_url, report
        )

    weights = {"text": 0.30, "structure": 0.20, "table": 0.15, "vietnamese": 0.15, "ai": 0.20}
    ai_score = report.ai_review.get("overall", 7) if report.ai_review else 7
    report.overall_score = (
        report.text_completeness * 10 * weights["text"]
        + report.structure_score * weights["structure"]
        + report.table_score * weights["table"]
        + report.vietnamese_score * weights["vietnamese"]
        + ai_score * weights["ai"]
    )

    logger.info(
        "Quality check hoàn thành: %s — overall=%.1f/10",
        pdf_path.name, report.overall_score,
    )
    for issue in report.issues:
        logger.info("  Issue: %s", issue)

    return report
