"""Auto-fix module: automatically correct issues found by spot-check.

Uses Gemini to compare the original PDF page image against the markdown
snippet containing the error, then replaces the faulty section with
the corrected version.

Batches all issues from the same SpotCheckResult into a single Gemini call
to reduce API usage.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from google import genai
from google.genai import types

from .config import Config
from .prompts import AUTO_FIX_PROMPT
from .spot_check import SpotCheckReport, SpotCheckResult

logger = logging.getLogger(__name__)


@dataclass
class FixAttempt:
    page_num: int
    line_start: int
    line_end: int
    issue_type: str
    issue_description: str
    original_snippet: str
    fixed_snippet: str
    success: bool
    log_message: str

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "issue_type": self.issue_type,
            "issue_description": self.issue_description,
            "original_snippet_preview": self.original_snippet[:200],
            "fixed_snippet_preview": self.fixed_snippet[:200] if self.fixed_snippet else "",
            "success": self.success,
            "log_message": self.log_message,
        }


@dataclass
class AutoFixReport:
    filename: str
    total_issues: int
    fixed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    fixes: list[FixAttempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "total_issues": self.total_issues,
            "fixed_count": self.fixed_count,
            "failed_count": self.failed_count,
            "skipped_count": self.skipped_count,
            "fixes": [f.to_dict() for f in self.fixes],
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_log(self, path: Path) -> None:
        lines = [
            f"=== Auto-Fix Report: {self.filename} ===",
            f"Total issues: {self.total_issues}",
            f"Fixed: {self.fixed_count} | Failed: {self.failed_count} | Skipped: {self.skipped_count}",
            "",
        ]
        for i, fix in enumerate(self.fixes, 1):
            status = "FIXED" if fix.success else "FAILED"
            lines.append(f"[{status}] #{i} Trang {fix.page_num + 1}, "
                         f"dòng {fix.line_start + 1}-{fix.line_end} | "
                         f"{fix.issue_type}: {fix.issue_description}")
            lines.append(f"  Log: {fix.log_message}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def summary(self) -> str:
        parts = [f"Auto-fix: {self.total_issues} lỗi"]
        if self.fixed_count:
            parts.append(f"{self.fixed_count} đã sửa")
        if self.failed_count:
            parts.append(f"{self.failed_count} thất bại")
        if self.skipped_count:
            parts.append(f"{self.skipped_count} bỏ qua")
        return ", ".join(parts)


def _build_batched_issue_text(issues: list) -> tuple[str, str]:
    """Combine multiple issues into a single issue_type + description string."""
    if len(issues) == 1:
        return issues[0].type, issues[0].description
    types = list(dict.fromkeys(i.type for i in issues))
    combined_type = " + ".join(types)
    descriptions = [f"- [{i.severity}] {i.type}: {i.description}" for i in issues]
    combined_desc = "\n".join(descriptions)
    return combined_type, combined_desc


def _call_auto_fix(
    client: genai.Client,
    model_name: str,
    page_image: bytes,
    snippet: str,
    line_start: int,
    line_end: int,
    issue_type: str,
    issue_description: str,
) -> str | None:
    prompt = (
        AUTO_FIX_PROMPT
        .replace("{issue_type}", issue_type)
        .replace("{issue_description}", issue_description)
        .replace("{line_start}", str(line_start + 1))
        .replace("{line_end}", str(line_end))
        .replace("{markdown_snippet}", snippet[:4000])
    )

    contents = [
        types.Part.from_bytes(data=page_image, mime_type="image/png"),
        prompt,
    ]

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        text = response.text
        if not text or not text.strip():
            return None
        text = text.strip()
        if text.startswith("```markdown"):
            text = text[len("```markdown"):].strip()
        elif text.startswith("```md"):
            text = text[len("```md"):].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text
    except Exception as e:
        logger.warning("Auto-fix Gemini call failed: %s", e)
        return None


def _replace_snippet_in_markdown(
    md_lines: list[str],
    line_start: int,
    line_end: int,
    fixed_snippet: str,
) -> list[str]:
    fixed_lines = fixed_snippet.split("\n")
    return md_lines[:line_start] + fixed_lines + md_lines[line_end:]


def auto_fix(
    pdf_path: str | Path,
    markdown: str,
    spot_report: SpotCheckReport,
    config: Config,
    skip_pages: set[int] | None = None,
) -> tuple[str, AutoFixReport]:
    """Attempt to auto-fix issues found by spot-check.

    Batches all issues from the same SpotCheckResult into one Gemini call.
    Pages in skip_pages are excluded.
    Returns the (possibly modified) markdown and an AutoFixReport.
    """
    pdf_path = Path(pdf_path)
    skip = skip_pages or set()

    fixable_results = [
        r for r in spot_report.results
        if r.severity in ("critical", "warning") and r.issues and r.page_num not in skip
    ]

    report = AutoFixReport(
        filename=pdf_path.name,
        total_issues=sum(len(r.issues) for r in fixable_results),
    )

    skipped_by_page = [
        r for r in spot_report.results
        if r.severity in ("critical", "warning") and r.issues and r.page_num in skip
    ]
    report.skipped_count = sum(len(r.issues) for r in skipped_by_page)

    if not fixable_results:
        logger.info("Auto-fix: không có lỗi cần sửa")
        return markdown, report

    if not config.gemini_api_key:
        logger.warning("Auto-fix: thiếu Gemini API key, bỏ qua")
        report.skipped_count += report.total_issues
        return markdown, report

    try:
        client = genai.Client(api_key=config.gemini_api_key)
    except Exception as e:
        logger.warning("Auto-fix: không khởi tạo được Gemini client: %s", e)
        report.skipped_count += report.total_issues
        return markdown, report

    doc = fitz.open(str(pdf_path))
    md_lines = markdown.split("\n")

    fixable_results.sort(key=lambda r: r.md_line_start, reverse=True)

    for result in fixable_results:
        page_num = result.page_num
        line_start = result.md_line_start
        line_end = result.md_line_end

        if page_num < 0 or page_num >= doc.page_count:
            logger.warning(
                "Auto-fix: trang %d ngoài phạm vi (tổng %d trang), bỏ qua",
                page_num + 1, doc.page_count,
            )
            report.skipped_count += len(result.issues)
            continue

        if line_start >= line_end or line_start < 0 or line_end > len(md_lines):
            logger.warning(
                "Auto-fix: trang %d, line range lỗi (%d-%d, tổng %d dòng), bỏ qua",
                page_num + 1, line_start, line_end, len(md_lines),
            )
            report.skipped_count += len(result.issues)
            continue

        original_snippet = "\n".join(md_lines[line_start:line_end])

        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=config.spot_check_dpi)
        page_image = pix.tobytes("png")

        combined_type, combined_desc = _build_batched_issue_text(result.issues)
        n_issues = len(result.issues)

        logger.info(
            "Auto-fix: Trang %d, dòng %d-%d | %d lỗi: %s",
            page_num + 1, line_start + 1, line_end, n_issues, combined_type,
        )

        fixed_text = _call_auto_fix(
            client,
            config.gemini_verify_model,
            page_image,
            original_snippet,
            line_start,
            line_end,
            combined_type,
            combined_desc,
        )

        MAX_SIZE_RATIO = 3.0
        if fixed_text and len(fixed_text) > len(original_snippet) * MAX_SIZE_RATIO:
            logger.warning(
                "  -> TỪ CHỐI: trang %d — fix quá lớn (%d ký tự vs gốc %d, ratio %.1fx), "
                "có thể hallucinate",
                page_num + 1, len(fixed_text), len(original_snippet),
                len(fixed_text) / max(len(original_snippet), 1),
            )
            for issue in result.issues:
                fix = FixAttempt(
                    page_num=page_num,
                    line_start=line_start,
                    line_end=line_end,
                    issue_type=issue.type,
                    issue_description=issue.description,
                    original_snippet=result.md_snippet,
                    fixed_snippet="",
                    success=False,
                    log_message=f"Fix quá lớn ({len(fixed_text)} vs {len(original_snippet)} ký tự)",
                )
                report.fixes.append(fix)
                report.failed_count += 1
            continue

        if fixed_text and fixed_text.strip() != original_snippet.strip():
            md_lines = _replace_snippet_in_markdown(
                md_lines, line_start, line_end, fixed_text,
            )

            for issue in result.issues:
                fix = FixAttempt(
                    page_num=page_num,
                    line_start=line_start,
                    line_end=line_end,
                    issue_type=issue.type,
                    issue_description=issue.description,
                    original_snippet=result.md_snippet,
                    fixed_snippet=fixed_text[:500],
                    success=True,
                    log_message=f"Đã sửa (batched {n_issues} issues, {len(fixed_text)} ký tự)",
                )
                report.fixes.append(fix)
                report.fixed_count += 1

            logger.info(
                "  -> ĐÃ SỬA: %d lỗi trang %d (%d ký tự)",
                n_issues, page_num + 1, len(fixed_text),
            )
        else:
            reason = "Gemini trả về kết quả giống hệt" if fixed_text else "Gemini không trả về kết quả"
            for issue in result.issues:
                fix = FixAttempt(
                    page_num=page_num,
                    line_start=line_start,
                    line_end=line_end,
                    issue_type=issue.type,
                    issue_description=issue.description,
                    original_snippet=result.md_snippet,
                    fixed_snippet="",
                    success=False,
                    log_message=reason,
                )
                report.fixes.append(fix)
                report.failed_count += 1

            logger.warning(
                "  -> THẤT BẠI: trang %d - %s", page_num + 1, reason,
            )

    doc.close()

    logger.info(
        "Auto-fix hoàn thành: %d lỗi, %d đã sửa, %d thất bại, %d bỏ qua",
        report.total_issues,
        report.fixed_count,
        report.failed_count,
        report.skipped_count,
    )

    return "\n".join(md_lines), report
