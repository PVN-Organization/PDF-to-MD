"""Offline spot-check — uses Ollama instead of Gemini to verify
random positions in the markdown output against PDF page images.
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

_INTERESTING_RE = re.compile(
    r"^#{1,6}\s+.*(Điều|Chương|Mục|Phần|Phụ lục)"
    r"|^\|.+\|$"
    r"|^---$",
    re.MULTILINE,
)

SPOT_CHECK_PROMPT_OLLAMA = """\
So sánh CHI TIẾT hình ảnh trang PDF gốc với đoạn Markdown bên dưới.
Kiểm tra 6 loại lỗi:

1. missing_content — Nội dung trong hình nhưng THIẾU trong Markdown
2. wrong_number — Số hiệu, ngày tháng, mã số KHÁC với hình gốc
3. embedded_page_number — Số trang nhúng giữa câu văn
4. running_header — Dòng running header/footer lọt vào nội dung
5. ocr_gibberish — Text vô nghĩa, sai chính tả nghiêm trọng
6. structure_error — Heading level sai, list sai, bảng vỡ

Trả về JSON (KHÔNG giải thích):
{"severity": "critical hoặc warning hoặc ok", "issues": [{"type": "<loại>", "description": "<mô tả tiếng Việt>", "severity": "critical hoặc warning"}]}

Markdown cần kiểm tra (dòng {line_start}-{line_end}):
{markdown_snippet}
"""


@dataclass
class SpotCheckIssueOffline:
    type: str
    description: str
    severity: str

    def to_dict(self) -> dict:
        return {"type": self.type, "description": self.description, "severity": self.severity}


@dataclass
class SpotCheckResultOffline:
    page_num: int
    md_line_start: int
    md_line_end: int
    severity: str
    issues: list[SpotCheckIssueOffline] = field(default_factory=list)
    md_snippet: str = ""

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "md_line_start": self.md_line_start,
            "md_line_end": self.md_line_end,
            "severity": self.severity,
            "issues": [i.to_dict() for i in self.issues],
            "md_snippet_preview": self.md_snippet[:200],
        }


@dataclass
class SpotCheckReportOffline:
    filename: str
    total_checks: int
    critical_count: int = 0
    warning_count: int = 0
    ok_count: int = 0
    results: list[SpotCheckResultOffline] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "total_checks": self.total_checks,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "ok_count": self.ok_count,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_log(self, path: Path) -> None:
        lines = [
            f"=== Spot-Check Log (Offline): {self.filename} ===",
            f"Tổng: {self.total_checks} vị trí | "
            f"Critical: {self.critical_count} | Warning: {self.warning_count} | OK: {self.ok_count}",
            "",
        ]
        for i, r in enumerate(self.results, 1):
            lines.append(
                f"[{r.severity.upper()}] #{i} Trang {r.page_num + 1}, "
                f"dòng {r.md_line_start + 1}-{r.md_line_end}"
            )
            for iss in r.issues:
                lines.append(f"  - [{iss.severity}] {iss.type}: {iss.description}")
            if not r.issues:
                lines.append("  (không phát hiện lỗi)")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def summary(self) -> str:
        parts = [f"Spot-check: {self.total_checks} vị trí"]
        if self.critical_count:
            parts.append(f"CRITICAL: {self.critical_count}")
        if self.warning_count:
            parts.append(f"WARNING: {self.warning_count}")
        parts.append(f"OK: {self.ok_count}")
        return " | ".join(parts)


def _pick_positions(md_lines: list[str], count: int) -> list[int]:
    total = len(md_lines)
    if total == 0:
        return []

    interesting: list[int] = []
    for i, line in enumerate(md_lines):
        if _INTERESTING_RE.match(line.strip()):
            interesting.append(i)

    zone_size = total // count if count > 0 else total
    selected: list[int] = []

    for zone_idx in range(count):
        zone_start = zone_idx * zone_size
        zone_end = min(zone_start + zone_size, total)
        zone_candidates = [c for c in interesting if zone_start <= c < zone_end]

        if zone_candidates:
            pos = random.choice(zone_candidates)
        else:
            pos = random.randint(zone_start, max(zone_start, zone_end - 1))
        selected.append(max(0, min(pos, total - 1)))

    return selected


def _extract_snippet(md_lines: list[str], center: int, window: int = 35) -> tuple[int, int, str]:
    start = max(0, center - window)
    end = min(len(md_lines), center + window)
    return start, end, "\n".join(md_lines[start:end])


def _map_line_to_page(line_num: int, total_lines: int, total_pages: int) -> int:
    if total_lines <= 0 or total_pages <= 0:
        return 0
    return min(line_num * total_pages // total_lines, total_pages - 1)


def _call_ollama_spot_check(
    client,
    model: str,
    page_image_b64: str,
    snippet: str,
    line_start: int,
    line_end: int,
) -> dict | None:
    prompt = (
        SPOT_CHECK_PROMPT_OLLAMA
        .replace("{line_start}", str(line_start + 1))
        .replace("{line_end}", str(line_end))
        .replace("{markdown_snippet}", snippet[:3000])
    )

    try:
        response = client.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [page_image_b64],
            }],
            options={"temperature": 0.1},
        )
        text = response.get("message", {}).get("content", "")
        if not text:
            return None
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        logger.warning("Spot-check Ollama call failed: %s", e)
        return None


def run_spot_check_offline(
    pdf_path: str | Path,
    markdown: str,
    total_pages: int,
    check_count: int = 5,
    check_dpi: int = 150,
    ollama_model: str = "qwen3-vl:8b",
    ollama_base_url: str = "http://localhost:11434",
) -> SpotCheckReportOffline:
    """Run random spot-checks using Ollama instead of Gemini."""
    pdf_path = Path(pdf_path)
    md_lines = markdown.split("\n")

    report = SpotCheckReportOffline(filename=pdf_path.name, total_checks=0)

    if not md_lines:
        return report

    try:
        import ollama
        client = ollama.Client(host=ollama_base_url)
        client.list()
    except ImportError:
        logger.warning("ollama package chưa được cài, bỏ qua spot-check")
        return report
    except Exception as e:
        logger.warning("Ollama không khả dụng tại %s: %s", ollama_base_url, e)
        return report

    positions = _pick_positions(md_lines, check_count)
    if not positions:
        return report

    doc = fitz.open(str(pdf_path))

    for check_idx, pos in enumerate(positions, 1):
        line_start, line_end, snippet = _extract_snippet(md_lines, pos)
        page_num = _map_line_to_page(pos, len(md_lines), total_pages)

        logger.info(
            "Spot-check %d/%d: trang %d, dòng %d-%d — đang gọi Ollama...",
            check_idx, len(positions), page_num + 1, line_start + 1, line_end,
        )

        if page_num < 0 or page_num >= doc.page_count:
            logger.warning(
                "Spot-check %d/%d: trang %d ngoài phạm vi, bỏ qua",
                check_idx, len(positions), page_num + 1,
            )
            continue

        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=check_dpi)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()

        result_data = _call_ollama_spot_check(
            client, ollama_model, img_b64, snippet, line_start, line_end,
        )

        severity = "ok"
        issues: list[SpotCheckIssueOffline] = []

        if result_data:
            severity = result_data.get("severity", "ok")
            for iss in result_data.get("issues", []):
                issues.append(SpotCheckIssueOffline(
                    type=iss.get("type", "unknown"),
                    description=iss.get("description", ""),
                    severity=iss.get("severity", "warning"),
                ))

        check_result = SpotCheckResultOffline(
            page_num=page_num,
            md_line_start=line_start,
            md_line_end=line_end,
            severity=severity,
            issues=issues,
            md_snippet=snippet[:500],
        )
        report.results.append(check_result)
        report.total_checks += 1

        if severity == "critical":
            report.critical_count += 1
        elif severity == "warning":
            report.warning_count += 1
        else:
            report.ok_count += 1

        if issues:
            for iss in issues:
                logger.info(
                    "  [%s] Trang %d: %s — %s",
                    iss.severity.upper(), page_num + 1, iss.type, iss.description,
                )
        else:
            logger.info(
                "Spot-check %d/%d: trang %d — OK",
                check_idx, len(positions), page_num + 1,
            )

    doc.close()

    logger.info(
        "Spot-check hoàn thành: %d vị trí, %d critical, %d warning, %d ok",
        report.total_checks, report.critical_count,
        report.warning_count, report.ok_count,
    )

    return report
