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

from .config import Config
from .converter import ChunkResult
from .prompts import SPOT_CHECK_PROMPT

logger = logging.getLogger(__name__)

_INTERESTING_RE = re.compile(
    r"^#{1,6}\s+.*(Điều|Chương|Mục|Phần|Phụ lục)"
    r"|^\|.+\|$"
    r"|^---$",
    re.MULTILINE,
)


@dataclass
class SpotCheckIssue:
    type: str
    description: str
    severity: str

    def to_dict(self) -> dict:
        return {"type": self.type, "description": self.description, "severity": self.severity}


@dataclass
class SpotCheckResult:
    page_num: int
    md_line_start: int
    md_line_end: int
    severity: str
    issues: list[SpotCheckIssue] = field(default_factory=list)
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
class SpotCheckReport:
    filename: str
    total_checks: int
    critical_count: int = 0
    warning_count: int = 0
    ok_count: int = 0
    results: list[SpotCheckResult] = field(default_factory=list)

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
            f"=== Spot-Check Log: {self.filename} ===",
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
        lines = [f"Spot-check: {self.total_checks} vị trí"]
        if self.critical_count:
            lines.append(f"  CRITICAL: {self.critical_count}")
        if self.warning_count:
            lines.append(f"  WARNING: {self.warning_count}")
        lines.append(f"  OK: {self.ok_count}")
        return "\n".join(lines)

    @property
    def failed_pages(self) -> set[int]:
        """Return set of page_nums that had critical or warning issues."""
        return {
            r.page_num for r in self.results
            if r.severity in ("critical", "warning") and r.issues
        }


def _build_page_map(chunk_results: list[ChunkResult], md_lines: list[str]) -> list[tuple[int, int, int]]:
    """Build a mapping from markdown line ranges to PDF page numbers.

    Returns list of (line_start, line_end, page_num) sorted by line_start.
    """
    mapping: list[tuple[int, int, int]] = []
    current_line = 0

    for cr in sorted(chunk_results, key=lambda r: r.chunk_id):
        if not cr.success or not cr.markdown.strip():
            continue
        chunk_lines = cr.markdown.strip().split("\n")
        chunk_line_count = len(chunk_lines)
        num_pages = cr.end_page - cr.start_page
        if num_pages <= 0:
            continue
        lines_per_page = max(chunk_line_count // num_pages, 1)
        for p_offset in range(num_pages):
            page_num = cr.start_page + p_offset
            line_start = current_line + p_offset * lines_per_page
            line_end = current_line + (p_offset + 1) * lines_per_page
            if p_offset == num_pages - 1:
                line_end = current_line + chunk_line_count
            mapping.append((line_start, min(line_end, len(md_lines)), page_num))
        current_line += chunk_line_count + 3

    return mapping


def _find_page_for_line(page_map: list[tuple[int, int, int]], target_line: int) -> int:
    for line_start, line_end, page_num in page_map:
        if line_start <= target_line < line_end:
            return page_num
    if page_map:
        return page_map[-1][2]
    return 0


def _find_lines_for_page(page_map: list[tuple[int, int, int]], target_page: int) -> tuple[int, int] | None:
    """Find the markdown line range for a given PDF page number."""
    for line_start, line_end, page_num in page_map:
        if page_num == target_page:
            return line_start, line_end
    return None


def _pick_check_positions(
    md_lines: list[str],
    chunk_results: list[ChunkResult],
    count: int,
) -> list[int]:
    total = len(md_lines)
    if total == 0:
        return []

    interesting: list[int] = []
    for i, line in enumerate(md_lines):
        if _INTERESTING_RE.match(line.strip()):
            interesting.append(i)

    chunk_boundaries: list[int] = []
    offset = 0
    for cr in sorted(chunk_results, key=lambda r: r.chunk_id):
        if not cr.success or not cr.markdown.strip():
            continue
        chunk_len = len(cr.markdown.strip().split("\n"))
        chunk_boundaries.append(min(offset + chunk_len, total - 1))
        offset += chunk_len + 3

    candidates = interesting + chunk_boundaries

    zone_size = total // count if count > 0 else total
    selected: list[int] = []

    for zone_idx in range(count):
        zone_start = zone_idx * zone_size
        zone_end = min(zone_start + zone_size, total)
        zone_candidates = [c for c in candidates if zone_start <= c < zone_end]

        if zone_candidates:
            pos = random.choice(zone_candidates)
        else:
            pos = random.randint(zone_start, max(zone_start, zone_end - 1))

        selected.append(max(0, min(pos, total - 1)))

    return selected


def _extract_snippet(md_lines: list[str], center: int, window: int = 35) -> tuple[int, int, str]:
    start = max(0, center - window)
    end = min(len(md_lines), center + window)
    snippet = "\n".join(md_lines[start:end])
    return start, end, snippet


def _call_spot_check(
    client: genai.Client,
    model_name: str,
    page_image: bytes,
    snippet: str,
    line_start: int,
    line_end: int,
) -> dict | None:
    prompt = SPOT_CHECK_PROMPT.replace("{line_start}", str(line_start + 1))
    prompt = prompt.replace("{line_end}", str(line_end))
    prompt = prompt.replace("{markdown_snippet}", snippet[:4000])

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
        if not text:
            return None
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        logger.warning("Spot-check Gemini call failed: %s", e)
        return None


def _do_check(
    client: genai.Client,
    model_name: str,
    doc: fitz.Document,
    md_lines: list[str],
    page_num: int,
    line_start: int,
    line_end: int,
    check_idx: int,
    total: int,
    dpi: int,
) -> SpotCheckResult | None:
    """Run a single spot-check against one page, return SpotCheckResult."""
    snippet = "\n".join(md_lines[line_start:line_end])

    logger.info(
        "Spot-check %d/%d: trang %d, dòng %d-%d — đang gọi Gemini...",
        check_idx, total, page_num + 1, line_start + 1, line_end,
    )

    if page_num < 0 or page_num >= doc.page_count:
        logger.warning("Spot-check: trang %d ngoài phạm vi, bỏ qua", page_num + 1)
        return None

    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=dpi)
    page_image = pix.tobytes("png")

    result_data = _call_spot_check(
        client, model_name, page_image, snippet, line_start, line_end,
    )

    severity = "ok"
    issues: list[SpotCheckIssue] = []

    if result_data:
        severity = result_data.get("severity", "ok")
        for iss in result_data.get("issues", []):
            issues.append(SpotCheckIssue(
                type=iss.get("type", "unknown"),
                description=iss.get("description", ""),
                severity=iss.get("severity", "warning"),
            ))

    check_result = SpotCheckResult(
        page_num=page_num,
        md_line_start=line_start,
        md_line_end=line_end,
        severity=severity,
        issues=issues,
        md_snippet=snippet[:500],
    )

    if issues:
        for iss in issues:
            logger.info(
                "  [%s] Trang %d: %s — %s",
                iss.severity.upper(), page_num + 1, iss.type, iss.description,
            )
    else:
        logger.info("Spot-check %d/%d: trang %d — OK", check_idx, total, page_num + 1)

    return check_result


def run_spot_check(
    pdf_path: str | Path,
    markdown: str,
    chunk_results: list[ChunkResult],
    config: Config,
) -> SpotCheckReport:
    """Run random spot-checks comparing markdown output against PDF page images."""
    pdf_path = Path(pdf_path)
    md_lines = markdown.split("\n")
    count = config.spot_check_count

    report = SpotCheckReport(filename=pdf_path.name, total_checks=0)

    if not md_lines or not config.gemini_api_key:
        return report

    positions = _pick_check_positions(md_lines, chunk_results, count)
    if not positions:
        return report

    page_map = _build_page_map(chunk_results, md_lines)

    try:
        client = genai.Client(api_key=config.gemini_api_key)
    except Exception as e:
        logger.warning("Could not init Gemini client for spot-check: %s", e)
        return report

    doc = fitz.open(str(pdf_path))

    for check_idx, pos in enumerate(positions, 1):
        line_start, line_end, snippet = _extract_snippet(md_lines, pos)
        page_num = _find_page_for_line(page_map, pos)

        result = _do_check(
            client, config.gemini_verify_model, doc, md_lines,
            page_num, line_start, line_end,
            check_idx, len(positions), config.spot_check_dpi,
        )
        if result is None:
            continue

        report.results.append(result)
        report.total_checks += 1
        if result.severity == "critical":
            report.critical_count += 1
        elif result.severity == "warning":
            report.warning_count += 1
        else:
            report.ok_count += 1

    doc.close()

    logger.info(
        "Spot-check hoàn thành: %d vị trí, %d critical, %d warning, %d ok",
        report.total_checks, report.critical_count,
        report.warning_count, report.ok_count,
    )

    return report


def _build_even_page_map(total_lines: int, total_pages: int) -> list[tuple[int, int, int]]:
    """Build page_map by dividing markdown lines evenly across pages.

    Used for recheck when original chunk_results page_map is stale
    (after auto-fix modified the markdown).
    """
    if total_pages <= 0 or total_lines <= 0:
        return []
    lines_per_page = max(total_lines // total_pages, 1)
    mapping = []
    for p in range(total_pages):
        start = p * lines_per_page
        end = (p + 1) * lines_per_page if p < total_pages - 1 else total_lines
        mapping.append((start, min(end, total_lines), p))
    return mapping


def recheck_pages(
    pdf_path: str | Path,
    markdown: str,
    chunk_results: list[ChunkResult],
    config: Config,
    page_nums: set[int],
    skip_pages: set[int] | None = None,
) -> SpotCheckReport:
    """Re-check specific PDF pages (instead of random positions).

    Uses an even page-map based on current markdown length (not original
    chunk_results) so line ranges stay valid after auto-fix rounds.
    Pages in skip_pages are excluded (already deemed unfixable).
    """
    pdf_path = Path(pdf_path)
    md_lines = markdown.split("\n")
    pages_to_check = sorted(page_nums - (skip_pages or set()))

    report = SpotCheckReport(filename=pdf_path.name, total_checks=0)

    if not pages_to_check or not config.gemini_api_key:
        return report

    try:
        client = genai.Client(api_key=config.gemini_api_key)
    except Exception as e:
        logger.warning("Could not init Gemini client for recheck: %s", e)
        return report

    doc = fitz.open(str(pdf_path))

    even_map = _build_even_page_map(len(md_lines), doc.page_count)

    for check_idx, pn in enumerate(pages_to_check, 1):
        line_range = _find_lines_for_page(even_map, pn)
        if line_range is None:
            logger.warning("Recheck: không tìm thấy dòng cho trang %d, bỏ qua", pn + 1)
            continue
        line_start, line_end = line_range

        if line_start >= line_end or line_start < 0:
            logger.warning("Recheck: trang %d, line range lỗi (%d-%d), bỏ qua",
                           pn + 1, line_start, line_end)
            continue

        result = _do_check(
            client, config.gemini_verify_model, doc, md_lines,
            pn, line_start, line_end,
            check_idx, len(pages_to_check), config.spot_check_dpi,
        )
        if result is None:
            continue

        report.results.append(result)
        report.total_checks += 1
        if result.severity == "critical":
            report.critical_count += 1
        elif result.severity == "warning":
            report.warning_count += 1
        else:
            report.ok_count += 1

    doc.close()

    logger.info(
        "Recheck hoàn thành: %d trang, %d critical, %d warning, %d ok",
        report.total_checks, report.critical_count,
        report.warning_count, report.ok_count,
    )

    return report
