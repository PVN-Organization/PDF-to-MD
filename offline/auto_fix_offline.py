"""Offline auto-fix — uses Ollama to correct issues found by spot-check."""
from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from .spot_check_offline import SpotCheckReportOffline

logger = logging.getLogger(__name__)

AUTO_FIX_PROMPT_OLLAMA = """\
Sửa lỗi trong đoạn Markdown bên dưới dựa trên hình ảnh trang PDF gốc.

Loại lỗi: {issue_type}
Mô tả: {issue_description}

Đoạn Markdown cần sửa (dòng {line_start}-{line_end}):
{markdown_snippet}

Yêu cầu:
- Đối chiếu KỸ với hình ảnh trang gốc
- Sửa CHÍNH XÁC lỗi được mô tả
- Giữ nguyên nội dung đúng, CHỈ sửa phần bị lỗi
- Giữ nguyên định dạng Markdown
- Trả về CHỈ đoạn Markdown đã sửa, KHÔNG giải thích
"""


@dataclass
class FixAttemptOffline:
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
class AutoFixReportOffline:
    filename: str
    total_issues: int
    fixed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    fixes: list[FixAttemptOffline] = field(default_factory=list)

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
            f"=== Auto-Fix Log (Offline): {self.filename} ===",
            f"Total: {self.total_issues} | Fixed: {self.fixed_count} | "
            f"Failed: {self.failed_count} | Skipped: {self.skipped_count}",
            "",
        ]
        for i, fix in enumerate(self.fixes, 1):
            status = "FIXED" if fix.success else "FAILED"
            lines.append(
                f"[{status}] #{i} Trang {fix.page_num + 1}, "
                f"dòng {fix.line_start + 1}-{fix.line_end} | "
                f"{fix.issue_type}: {fix.issue_description}"
            )
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


def auto_fix_offline(
    pdf_path: str | Path,
    markdown: str,
    spot_report: SpotCheckReportOffline,
    check_dpi: int = 150,
    ollama_model: str = "qwen3-vl:8b",
    ollama_base_url: str = "http://localhost:11434",
) -> tuple[str, AutoFixReportOffline]:
    """Auto-fix issues using Ollama. Returns (fixed_markdown, report)."""
    pdf_path = Path(pdf_path)

    fixable = [
        r for r in spot_report.results
        if r.severity in ("critical", "warning") and r.issues
    ]

    report = AutoFixReportOffline(
        filename=pdf_path.name,
        total_issues=sum(len(r.issues) for r in fixable),
    )

    if not fixable:
        logger.info("Auto-fix: không có lỗi cần sửa")
        return markdown, report

    try:
        import ollama
        client = ollama.Client(host=ollama_base_url)
        client.list()
    except ImportError:
        logger.warning("ollama package chưa được cài, bỏ qua auto-fix")
        report.skipped_count = report.total_issues
        return markdown, report
    except Exception as e:
        logger.warning("Ollama không khả dụng: %s", e)
        report.skipped_count = report.total_issues
        return markdown, report

    doc = fitz.open(str(pdf_path))
    md_lines = markdown.split("\n")

    fixable.sort(key=lambda r: r.md_line_start, reverse=True)

    for result in fixable:
        page_num = result.page_num
        line_start = result.md_line_start
        line_end = result.md_line_end

        if page_num < 0 or page_num >= doc.page_count:
            logger.warning("Auto-fix: trang %d ngoài phạm vi, bỏ qua", page_num + 1)
            report.skipped_count += len(result.issues)
            continue

        original_snippet = "\n".join(md_lines[line_start:line_end])

        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=check_dpi)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()

        for issue in result.issues:
            logger.info(
                "Auto-fix: [%s] Trang %d, dòng %d-%d | %s: %s",
                issue.severity.upper(), page_num + 1,
                line_start + 1, line_end,
                issue.type, issue.description,
            )

            prompt = (
                AUTO_FIX_PROMPT_OLLAMA
                .replace("{issue_type}", issue.type)
                .replace("{issue_description}", issue.description)
                .replace("{line_start}", str(line_start + 1))
                .replace("{line_end}", str(line_end))
                .replace("{markdown_snippet}", original_snippet[:3000])
            )

            try:
                response = client.chat(
                    model=ollama_model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }],
                    options={"temperature": 0.1},
                )
                fixed_text = response.get("message", {}).get("content", "")

                if fixed_text:
                    fixed_text = fixed_text.strip()
                    if fixed_text.startswith("```"):
                        lines_f = fixed_text.split("\n")
                        if len(lines_f) > 2:
                            fixed_text = "\n".join(lines_f[1:-1] if lines_f[-1].strip() == "```" else lines_f[1:])
                    if fixed_text.endswith("```"):
                        fixed_text = fixed_text[:-3].strip()
            except Exception as e:
                logger.warning("Ollama auto-fix call failed: %s", e)
                fixed_text = ""

            if fixed_text and fixed_text.strip() != original_snippet.strip():
                fixed_lines = fixed_text.split("\n")
                md_lines = md_lines[:line_start] + fixed_lines + md_lines[line_end:]
                original_snippet = fixed_text
                line_end = line_start + len(fixed_lines)

                fix = FixAttemptOffline(
                    page_num=page_num,
                    line_start=line_start,
                    line_end=line_end,
                    issue_type=issue.type,
                    issue_description=issue.description,
                    original_snippet=result.md_snippet,
                    fixed_snippet=fixed_text[:500],
                    success=True,
                    log_message=f"Đã sửa thành công ({len(fixed_text)} ký tự)",
                )
                report.fixes.append(fix)
                report.fixed_count += 1
                logger.info("  -> ĐÃ SỬA: %s trang %d", issue.type, page_num + 1)
            else:
                reason = "Ollama trả về kết quả giống hệt" if fixed_text else "Ollama không trả về kết quả"
                fix = FixAttemptOffline(
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
                logger.warning("  -> THẤT BẠI: %s trang %d — %s", issue.type, page_num + 1, reason)

    doc.close()

    logger.info(
        "Auto-fix hoàn thành: %d lỗi, %d đã sửa, %d thất bại, %d bỏ qua",
        report.total_issues, report.fixed_count,
        report.failed_count, report.skipped_count,
    )

    return "\n".join(md_lines), report
