from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .analyzer import PDFAnalysis, analyze_pdf
from .assembler import assemble
from .auto_fix import AutoFixReport, auto_fix
from .config import Config
from .converter import convert_chunks
from .extractor import ChunkExtraction, extract_chunk
from .planner import ChunkPlan, create_plan
from .quality import QualityReport, check_quality
from .renderer import RenderResult, render_pages
from .spot_check import SpotCheckReport, run_spot_check

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class PipelineResult:
    filename: str
    markdown: str
    analysis: PDFAnalysis
    chunks: list[ChunkPlan]
    quality_report: QualityReport | None
    spot_check_report: SpotCheckReport | None
    auto_fix_report: AutoFixReport | None
    output_path: Path
    elapsed_seconds: float


async def process_pdf(
    pdf_path: str | Path,
    config: Config,
    run_quality_check: bool = True,
    run_spot_check_flag: bool = True,
    run_auto_fix_flag: bool = True,
) -> PipelineResult:
    pdf_path = Path(pdf_path)
    start_time = time.time()

    stem = pdf_path.stem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)
    output_subdir = safe_stem
    output_dir = config.output_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = config.temp_dir / safe_stem / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Phase 1: Analyze
        task = progress.add_task(f"[1/7] Phân tích {pdf_path.name}...", total=None)
        analysis = analyze_pdf(pdf_path)
        progress.update(task, completed=True, description=f"[1/7] Phân tích: {analysis.total_pages} trang")

        # Phase 2: Plan chunks
        task = progress.add_task("[2/7] Lên kế hoạch chia chunk...", total=None)
        chunks = create_plan(analysis, config)
        table_count = sum(len(c.table_pages) for c in chunks)
        progress.update(
            task, completed=True,
            description=f"[2/7] Kế hoạch: {len(chunks)} chunks, {table_count} trang có bảng",
        )

        # Phase 3: Render pages
        task = progress.add_task("[3/7] Render trang thành ảnh...", total=None)
        render_result = render_pages(pdf_path, chunks, config, output_subdir)
        progress.update(
            task, completed=True,
            description=f"[3/7] Render: {len(render_result.pages)} trang",
        )

        # Phase 4: Extract table hints + OCR fallback (FREE)
        task = progress.add_task(
            "[4/7] Trích xuất table hints (PyMuPDF)...", total=None
        )
        image_paths = {rp.page_num: rp.image_path for rp in render_result.pages}
        extractions: list[ChunkExtraction] = []
        for chunk in chunks:
            chunk_page_infos = [
                p for p in analysis.pages
                if chunk.start_page <= p.page_num < chunk.end_page
            ]
            extraction = extract_chunk(
                pdf_path=pdf_path,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                page_infos=chunk_page_infos,
                image_paths=image_paths,
                chunk_id=chunk.chunk_id,
                tesseract_lang=config.tesseract_lang,
                use_tesseract=config.use_tesseract,
            )
            extractions.append(extraction)

        table_chunk_count = sum(
            1 for e in extractions
            if any(pe.has_tables for pe in e.page_extractions)
        )
        progress.update(
            task, completed=True,
            description=(
                f"[4/7] Table hints: {table_chunk_count}/{len(extractions)} chunks có bảng"
            ),
        )

        # Phase 5: Convert with Gemini (gemini-2.5-flash)
        task = progress.add_task(
            f"[5/7] Gemini convert {len(chunks)} chunks ({config.gemini_model})...",
            total=None,
        )
        chunk_results = await convert_chunks(
            chunks, extractions, render_result, config,
            doc_title=pdf_path.stem,
            cache_dir=cache_dir,
        )
        success_count = sum(1 for r in chunk_results if r.success)
        fallback_count = sum(1 for r in chunk_results if r.skipped_gemini)
        progress.update(
            task, completed=True,
            description=(
                f"[5/7] Convert: {success_count}/{len(chunks)} OK"
                + (f", {fallback_count} OCR fallback" if fallback_count else "")
            ),
        )

        # Phase 6: Assemble
        task = progress.add_task("[6/7] Ghép nối markdown...", total=None)
        final_markdown = assemble(chunk_results, analysis)
        output_path = output_dir / f"output_{stem}.md"
        output_path.write_text(final_markdown, encoding="utf-8")
        progress.update(
            task, completed=True,
            description=f"[6/7] Ghép nối: {len(final_markdown)} ký tự",
        )

        # Phase 7: Quality check
        quality_report = None
        if run_quality_check:
            task = progress.add_task("[7/8] Kiểm tra chất lượng...", total=None)
            quality_report = check_quality(
                pdf_path, final_markdown, analysis, config
            )
            report_path = output_dir / "quality_report.json"
            quality_report.save(report_path)
            progress.update(
                task, completed=True,
                description=f"[7/8] Chất lượng: {quality_report.overall_score:.1f}/10",
            )

        # Phase 8: Spot check
        spot_check_report = None
        if run_spot_check_flag:
            task = progress.add_task(
                f"[8/9] Spot-check {config.spot_check_count} vị trí...", total=None
            )
            spot_check_report = run_spot_check(
                pdf_path, final_markdown, chunk_results, config
            )
            sc_path = output_dir / "spot_check_report.json"
            spot_check_report.save(sc_path)
            sc_log_path = output_dir / "spot_check.log"
            spot_check_report.save_log(sc_log_path)
            desc = f"[8/9] Spot-check: {spot_check_report.ok_count} OK"
            if spot_check_report.critical_count:
                desc += f", {spot_check_report.critical_count} critical"
            if spot_check_report.warning_count:
                desc += f", {spot_check_report.warning_count} warning"
            progress.update(task, completed=True, description=desc)

        # Phase 9: Auto-fix
        auto_fix_report = None
        if (
            run_auto_fix_flag
            and spot_check_report
            and (spot_check_report.critical_count + spot_check_report.warning_count) > 0
        ):
            total_fixable = spot_check_report.critical_count + spot_check_report.warning_count
            task = progress.add_task(
                f"[9/9] Auto-fix {total_fixable} lỗi...", total=None
            )
            final_markdown, auto_fix_report = auto_fix(
                pdf_path, final_markdown, spot_check_report, config
            )
            af_path = output_dir / "auto_fix_report.json"
            auto_fix_report.save(af_path)
            af_log_path = output_dir / "auto_fix.log"
            auto_fix_report.save_log(af_log_path)

            output_path.write_text(final_markdown, encoding="utf-8")

            desc = f"[9/9] Auto-fix: {auto_fix_report.fixed_count} sửa"
            if auto_fix_report.failed_count:
                desc += f", {auto_fix_report.failed_count} thất bại"
            progress.update(task, completed=True, description=desc)

    elapsed = time.time() - start_time

    return PipelineResult(
        filename=pdf_path.name,
        markdown=final_markdown,
        analysis=analysis,
        chunks=chunks,
        quality_report=quality_report,
        spot_check_report=spot_check_report,
        auto_fix_report=auto_fix_report,
        output_path=output_path,
        elapsed_seconds=elapsed,
    )


async def process_batch(
    pdf_dir: str | Path,
    config: Config,
    run_quality_check: bool = True,
    run_spot_check_flag: bool = True,
    run_auto_fix_flag: bool = True,
) -> list[PipelineResult]:
    """Process all PDFs in a directory."""
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[red]Không tìm thấy file PDF nào trong {pdf_dir}[/red]")
        return []

    console.print(f"\n[bold]Tìm thấy {len(pdf_files)} file PDF:[/bold]")
    for i, f in enumerate(pdf_files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        console.print(f"  {i}. {f.name} ({size_mb:.1f} MB)")
    console.print()

    results: list[PipelineResult] = []
    for i, pdf_file in enumerate(pdf_files, 1):
        console.rule(f"[bold cyan]File {i}/{len(pdf_files)}: {pdf_file.name}")
        try:
            result = await process_pdf(
                pdf_file, config, run_quality_check,
                run_spot_check_flag, run_auto_fix_flag,
            )
            results.append(result)
            console.print(
                f"[green]Hoàn thành trong {result.elapsed_seconds:.1f}s → "
                f"{result.output_path}[/green]\n"
            )
        except Exception as e:
            console.print(f"[red]LỖI xử lý {pdf_file.name}: {e}[/red]\n")
            logger.exception("Failed to process %s", pdf_file.name)

    _print_summary(results)
    return results


def _print_summary(results: list[PipelineResult]) -> None:
    if not results:
        return

    console.rule("[bold green]TÓM TẮT KẾT QUẢ")
    from rich.table import Table

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan", max_width=50)
    table.add_column("Trang", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Ký tự MD", justify="right")
    table.add_column("Chất lượng", justify="right")
    table.add_column("Spot-check", justify="right")
    table.add_column("Fixed", justify="right")
    table.add_column("Thời gian", justify="right")

    for r in results:
        quality_str = (
            f"{r.quality_report.overall_score:.1f}/10"
            if r.quality_report
            else "N/A"
        )
        sc = r.spot_check_report
        if sc and sc.total_checks > 0:
            parts = []
            if sc.critical_count:
                parts.append(f"{sc.critical_count}C")
            if sc.warning_count:
                parts.append(f"{sc.warning_count}W")
            parts.append(f"{sc.ok_count}OK")
            spot_str = "/".join(parts)
        else:
            spot_str = "N/A"
        af = r.auto_fix_report
        if af and af.total_issues > 0:
            fix_str = f"{af.fixed_count}/{af.total_issues}"
        else:
            fix_str = "N/A"
        table.add_row(
            r.filename[:48],
            str(r.analysis.total_pages),
            str(len(r.chunks)),
            f"{len(r.markdown):,}",
            quality_str,
            spot_str,
            fix_str,
            f"{r.elapsed_seconds:.1f}s",
        )

    console.print(table)
    console.print()
