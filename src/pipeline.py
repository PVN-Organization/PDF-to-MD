from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from .analyzer import PDFAnalysis, analyze_pdf
from .assembler import assemble
from .auto_fix import AutoFixReport, auto_fix
from .config import Config
from .converter import convert_chunks
from .extractor import ChunkExtraction, extract_chunk
from .integrity import IntegrityReport, run_integrity_check
from .planner import ChunkPlan, create_plan
from .quality import QualityReport, check_quality
from .renderer import RenderResult, render_pages
from .spot_check import SpotCheckReport, run_spot_check, recheck_pages

logger = logging.getLogger(__name__)
console = Console()


def _add_file_logger(output_dir: Path) -> logging.FileHandler:
    log_path = output_dir / "pipeline.log"
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    return fh


@dataclass
class PipelineResult:
    filename: str
    markdown: str
    analysis: PDFAnalysis
    chunks: list[ChunkPlan]
    quality_report: QualityReport | None
    spot_check_report: SpotCheckReport | None
    auto_fix_report: AutoFixReport | None
    final_spot_report: SpotCheckReport | None
    fix_rounds: int
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

    file_handler = _add_file_logger(output_dir)

    logger.info("=" * 60)
    logger.info("Pipeline online: %s", pdf_path.name)
    logger.info("Model convert: %s | Model verify: %s",
                config.gemini_model, config.gemini_verify_model)
    logger.info("Integrity: %s | Spot-check: %s | Auto-fix: %s | Max rounds: %d | Min score: %.1f",
                config.integrity_check, run_spot_check_flag, run_auto_fix_flag,
                config.max_fix_rounds, config.min_acceptable_score)
    logger.info("=" * 60)

    spot_check_report = None
    auto_fix_report = None
    final_spot_report = None
    fix_rounds = 0
    quality_report = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Phase 1: Analyze
        task = progress.add_task(f"[1/12] Phân tích {pdf_path.name}...", total=None)
        logger.info("Phase 1: Phân tích PDF")
        analysis = analyze_pdf(pdf_path)
        progress.update(task, completed=True, description=f"[1/12] Phân tích: {analysis.total_pages} trang")
        logger.info("Phase 1 done: %d trang, %.1f MB, %.0f%% scanned",
                     analysis.total_pages, analysis.file_size_mb, analysis.scanned_ratio * 100)

        # Phase 2: Plan chunks
        task = progress.add_task("[2/12] Lên kế hoạch chia chunk...", total=None)
        logger.info("Phase 2: Lên kế hoạch chunk")
        chunks = create_plan(analysis, config)
        table_count = sum(len(c.table_pages) for c in chunks)
        progress.update(
            task, completed=True,
            description=f"[2/12] Kế hoạch: {len(chunks)} chunks, {table_count} trang có bảng",
        )
        logger.info("Phase 2 done: %d chunks, %d trang có bảng", len(chunks), table_count)

        # Phase 3: Render pages
        task = progress.add_task("[3/12] Render trang thành ảnh...", total=None)
        logger.info("Phase 3: Render trang thành ảnh (DPI=%d)", config.render_dpi)
        render_result = render_pages(pdf_path, chunks, config, output_subdir)
        progress.update(
            task, completed=True,
            description=f"[3/12] Render: {len(render_result.pages)} trang",
        )
        logger.info("Phase 3 done: %d trang rendered", len(render_result.pages))

        # Phase 4: Extract table hints + OCR fallback
        task = progress.add_task("[4/12] Trích xuất table hints (PyMuPDF)...", total=None)
        logger.info("Phase 4: Trích xuất table hints + OCR fallback")
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
            description=f"[4/12] Table hints: {table_chunk_count}/{len(extractions)} chunks có bảng",
        )
        logger.info("Phase 4 done: %d/%d chunks có bảng", table_chunk_count, len(extractions))

        # Phase 5: Convert with Gemini
        task = progress.add_task(
            f"[5/12] Gemini convert {len(chunks)} chunks ({config.gemini_model})...",
            total=None,
        )
        logger.info("Phase 5: Gemini convert (%s)", config.gemini_model)
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
                f"[5/12] Convert: {success_count}/{len(chunks)} OK"
                + (f", {fallback_count} OCR fallback" if fallback_count else "")
            ),
        )
        logger.info("Phase 5 done: %d/%d OK, %d OCR fallback",
                     success_count, len(chunks), fallback_count)

        # Phase 6: Assemble
        task = progress.add_task("[6/12] Ghép nối markdown...", total=None)
        logger.info("Phase 6: Ghép nối markdown")
        final_markdown = assemble(chunk_results, analysis)
        output_path = output_dir / f"output_{stem}.md"
        output_path.write_text(final_markdown, encoding="utf-8")
        progress.update(
            task, completed=True,
            description=f"[6/12] Ghép nối: {len(final_markdown):,} ký tự",
        )
        logger.info("Phase 6 done: %d ký tự, saved to %s", len(final_markdown), output_path)

        # Phase 7: Integrity Check (page + article coverage)
        integrity_report = None
        if config.integrity_check:
            task = progress.add_task("[7/12] Kiểm tra tính toàn vẹn (trang + Điều)...", total=None)
            logger.info("Phase 7: Integrity Check (page + article coverage)")
            final_markdown, integrity_report = run_integrity_check(
                pdf_path, final_markdown, chunk_results, config,
                render_result, analysis.total_pages,
                doc_title=pdf_path.stem, cache_dir=cache_dir,
                auto_fix=True,
            )
            if integrity_report.converted_count > 0:
                output_path.write_text(final_markdown, encoding="utf-8")

            desc = f"[7/12] Integrity: {integrity_report.covered_pages}/{integrity_report.total_pages} trang"
            if integrity_report.missing_articles:
                desc += f", thiếu {len(integrity_report.missing_articles)} Điều"
            else:
                desc += ", đủ Điều"
            if integrity_report.converted_count > 0:
                desc += f" → +{integrity_report.converted_count} trang bổ sung"
            progress.update(task, completed=True, description=desc)
            logger.info(
                "Phase 7 done: %d/%d trang, thiếu %d Điều, convert bổ sung %d trang",
                integrity_report.covered_pages, integrity_report.total_pages,
                len(integrity_report.missing_articles), integrity_report.converted_count,
            )
        else:
            task = progress.add_task("[7/12] Integrity check — bỏ qua", total=None)
            progress.update(task, completed=True, description="[7/12] Integrity check — bỏ qua")
            logger.info("Phase 7: Integrity check — bỏ qua (--no-integrity)")

        # Phase 8: Spot-check (random verification)
        if run_spot_check_flag:
            task = progress.add_task(
                f"[8/12] Spot-check {config.spot_check_count} vị trí...", total=None
            )
            logger.info("Phase 8: Spot-check (%d vị trí ngẫu nhiên)", config.spot_check_count)
            spot_check_report = run_spot_check(
                pdf_path, final_markdown, chunk_results, config
            )
            sc_path = output_dir / "spot_check_report.json"
            spot_check_report.save(sc_path)
            sc_log_path = output_dir / "spot_check.log"
            spot_check_report.save_log(sc_log_path)

            desc = f"[8/12] Spot-check: {spot_check_report.ok_count} OK"
            if spot_check_report.critical_count:
                desc += f", {spot_check_report.critical_count} critical"
            if spot_check_report.warning_count:
                desc += f", {spot_check_report.warning_count} warning"
            progress.update(task, completed=True, description=desc)
            logger.info("Phase 8 done: %d OK, %d critical, %d warning",
                         spot_check_report.ok_count, spot_check_report.critical_count,
                         spot_check_report.warning_count)

        # Phase 9: Auto-fix (first pass)
        if (
            run_auto_fix_flag
            and spot_check_report
            and (spot_check_report.critical_count + spot_check_report.warning_count) > 0
        ):
            total_fixable = spot_check_report.critical_count + spot_check_report.warning_count
            task = progress.add_task(
                f"[9/12] Auto-fix {total_fixable} lỗi...", total=None
            )
            logger.info("Phase 9: Auto-fix (%d lỗi phát hiện)", total_fixable)
            final_markdown, auto_fix_report = auto_fix(
                pdf_path, final_markdown, spot_check_report, config
            )
            af_path = output_dir / "auto_fix_report.json"
            auto_fix_report.save(af_path)
            af_log_path = output_dir / "auto_fix.log"
            auto_fix_report.save_log(af_log_path)

            output_path.write_text(final_markdown, encoding="utf-8")

            desc = f"[9/12] Auto-fix: {auto_fix_report.fixed_count} sửa"
            if auto_fix_report.failed_count:
                desc += f", {auto_fix_report.failed_count} thất bại"
            progress.update(task, completed=True, description=desc)
            logger.info("Phase 9 done: %d sửa, %d thất bại, %d bỏ qua",
                         auto_fix_report.fixed_count, auto_fix_report.failed_count,
                         auto_fix_report.skipped_count)

        # Phase 10: Smart final check loop
        # Uses recheck_pages() on previously-failed pages, per-page retry tracking,
        # and early exit when quality score is acceptable.
        if (
            run_spot_check_flag
            and run_auto_fix_flag
            and spot_check_report
            and spot_check_report.critical_count > 0
            and config.max_fix_rounds > 0
        ):
            max_rounds = config.max_fix_rounds
            task = progress.add_task(
                f"[10/12] Smart fix loop (tối đa {max_rounds} vòng)...", total=None
            )
            logger.info("Phase 10: Smart fix loop (tối đa %d vòng, min score %.1f)",
                         max_rounds, config.min_acceptable_score)

            pages_to_check = spot_check_report.failed_pages
            fix_attempts: dict[int, int] = {}
            unfixable_pages: set[int] = set()

            for round_num in range(1, max_rounds + 1):
                active_pages = pages_to_check - unfixable_pages
                if not active_pages:
                    logger.info("  Vòng %d: không còn trang cần check -> kết thúc", round_num)
                    fix_rounds = round_num
                    break

                logger.info("  Vòng %d/%d: recheck %d trang (skip %d unfixable)...",
                             round_num, max_rounds, len(active_pages), len(unfixable_pages))

                re_check = recheck_pages(
                    pdf_path, final_markdown, chunk_results, config,
                    page_nums=pages_to_check,
                    skip_pages=unfixable_pages,
                )
                re_check.save(output_dir / f"final_check_round_{round_num}.json")
                re_check.save_log(output_dir / f"final_check_round_{round_num}.log")

                logger.info("  Vòng %d: %d critical, %d warning, %d OK",
                             round_num, re_check.critical_count,
                             re_check.warning_count, re_check.ok_count)

                final_spot_report = re_check
                fix_rounds = round_num

                if re_check.critical_count == 0:
                    logger.info("  Vòng %d: không còn CRITICAL -> kết thúc", round_num)
                    break

                still_bad_pages = re_check.failed_pages
                for pn in still_bad_pages:
                    fix_attempts[pn] = fix_attempts.get(pn, 0) + 1
                    if fix_attempts[pn] >= 2:
                        unfixable_pages.add(pn)
                        logger.warning("  Trang %d: đã thử %d lần -> đánh dấu unfixable",
                                        pn + 1, fix_attempts[pn])

                fixable_in_round = still_bad_pages - unfixable_pages
                if not fixable_in_round:
                    logger.info("  Vòng %d: tất cả trang lỗi đều unfixable -> kết thúc", round_num)
                    break

                logger.info("  Vòng %d: fix %d trang (skip %d unfixable)...",
                             round_num, len(fixable_in_round), len(unfixable_pages))

                final_markdown, round_fix = auto_fix(
                    pdf_path, final_markdown, re_check, config,
                    skip_pages=unfixable_pages,
                )
                round_fix.save(output_dir / f"auto_fix_round_{round_num}.json")
                round_fix.save_log(output_dir / f"auto_fix_round_{round_num}.log")

                output_path.write_text(final_markdown, encoding="utf-8")

                logger.info("  Vòng %d fix: %d sửa, %d thất bại, %d bỏ qua",
                             round_num, round_fix.fixed_count,
                             round_fix.failed_count, round_fix.skipped_count)

                pages_to_check = still_bad_pages

                # Early exit: quick quality check
                if run_quality_check and config.min_acceptable_score > 0:
                    quick_qr = check_quality(pdf_path, final_markdown, analysis, config)
                    logger.info("  Vòng %d quality: %.1f/10 (ngưỡng: %.1f)",
                                 round_num, quick_qr.overall_score,
                                 config.min_acceptable_score)
                    if quick_qr.overall_score >= config.min_acceptable_score:
                        logger.info("  Đạt ngưỡng %.1f -> kết thúc sớm",
                                     config.min_acceptable_score)
                        break
            else:
                logger.warning("  Đã hết %d vòng", max_rounds)
                if unfixable_pages:
                    logger.warning("  Trang unfixable: %s",
                                    ", ".join(str(p + 1) for p in sorted(unfixable_pages)))

            desc = f"[10/12] Smart fix: {fix_rounds} vòng"
            if unfixable_pages:
                desc += f", {len(unfixable_pages)} unfixable"
            if final_spot_report:
                if final_spot_report.critical_count == 0:
                    desc += " — hết CRITICAL"
                else:
                    desc += f" — còn {final_spot_report.critical_count} CRITICAL"
            progress.update(task, completed=True, description=desc)
            logger.info("Phase 10 done: %d vòng, %d unfixable pages",
                         fix_rounds, len(unfixable_pages))

        # Phase 11: Final Integrity Check (sau smart fix, trước quality)
        final_integrity_report = None
        if config.integrity_check:
            task = progress.add_task("[11/12] Final integrity check...", total=None)
            logger.info("Phase 11: Final Integrity Check (sau smart fix)")
            final_markdown, final_integrity_report = run_integrity_check(
                pdf_path, final_markdown, chunk_results, config,
                render_result, analysis.total_pages,
                doc_title=pdf_path.stem, cache_dir=cache_dir,
                auto_fix=True,
            )
            if final_integrity_report.converted_count > 0:
                output_path.write_text(final_markdown, encoding="utf-8")

            fi = final_integrity_report
            desc = f"[11/12] Final integrity: {fi.covered_pages}/{fi.total_pages} trang"
            if fi.missing_articles:
                desc += f", thiếu {len(fi.missing_articles)} Điều"
            else:
                desc += ", đủ Điều"
            if fi.converted_count > 0:
                desc += f" → +{fi.converted_count} trang"
            progress.update(task, completed=True, description=desc)
            logger.info(
                "Phase 11 done: %d/%d trang, thiếu %d Điều, convert bổ sung %d trang",
                fi.covered_pages, fi.total_pages,
                len(fi.missing_articles), fi.converted_count,
            )
        else:
            task = progress.add_task("[11/12] Final integrity — bỏ qua", total=None)
            progress.update(task, completed=True, description="[11/12] Final integrity — bỏ qua")
            logger.info("Phase 11: Final integrity — bỏ qua (--no-integrity)")

        # Phase 12: Quality check (CUOI CUNG — chấm điểm trên bản đã fix xong)
        if run_quality_check:
            task = progress.add_task("[12/12] Chấm điểm chất lượng (final)...", total=None)
            logger.info("Phase 12: Quality check (trên bản cuối cùng)")
            quality_report = check_quality(
                pdf_path, final_markdown, analysis, config
            )
            report_path = output_dir / "quality_report.json"
            quality_report.save(report_path)
            progress.update(
                task, completed=True,
                description=f"[12/12] Chất lượng: {quality_report.overall_score:.1f}/10",
            )
            logger.info("Phase 12 done: overall=%.1f/10", quality_report.overall_score)

    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("Pipeline hoàn thành: %s — %.1fs", pdf_path.name, elapsed)
    logger.info("  Markdown: %d ký tự", len(final_markdown))
    if quality_report:
        logger.info("  Chất lượng: %.1f/10", quality_report.overall_score)
    if spot_check_report:
        logger.info("  Spot-check: %d C / %d W / %d OK",
                     spot_check_report.critical_count, spot_check_report.warning_count,
                     spot_check_report.ok_count)
    if auto_fix_report:
        logger.info("  Auto-fix: %d/%d sửa", auto_fix_report.fixed_count, auto_fix_report.total_issues)
    if fix_rounds > 0:
        logger.info("  Final check: %d vòng", fix_rounds)
        if final_spot_report:
            logger.info("  Final: %d C / %d W / %d OK",
                         final_spot_report.critical_count, final_spot_report.warning_count,
                         final_spot_report.ok_count)
    logger.info("=" * 60)

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()

    return PipelineResult(
        filename=pdf_path.name,
        markdown=final_markdown,
        analysis=analysis,
        chunks=chunks,
        quality_report=quality_report,
        spot_check_report=spot_check_report,
        auto_fix_report=auto_fix_report,
        final_spot_report=final_spot_report,
        fix_rounds=fix_rounds,
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
    table.add_column("Rounds", justify="right")
    table.add_column("Final", justify="right")
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
        fix_str = f"{af.fixed_count}/{af.total_issues}" if af and af.total_issues > 0 else "N/A"

        rounds_str = str(r.fix_rounds) if r.fix_rounds > 0 else "-"

        fs = r.final_spot_report
        if fs and fs.total_checks > 0:
            fparts = []
            if fs.critical_count:
                fparts.append(f"{fs.critical_count}C")
            if fs.warning_count:
                fparts.append(f"{fs.warning_count}W")
            fparts.append(f"{fs.ok_count}OK")
            final_str = "/".join(fparts)
        else:
            final_str = "-"

        table.add_row(
            r.filename[:48],
            str(r.analysis.total_pages),
            str(len(r.chunks)),
            f"{len(r.markdown):,}",
            quality_str,
            spot_str,
            fix_str,
            rounds_str,
            final_str,
            f"{r.elapsed_seconds:.1f}s",
        )

    console.print(table)
    console.print()
