#!/usr/bin/env python3
"""Offline PDF to Markdown Pipeline — no internet required.

Uses marker-pdf for conversion, with optional Ollama local LLM for polishing,
quality check, spot-check, and auto-fix. Full pipeline logging.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from offline.config_offline import OfflineConfig, VALID_STRATEGIES
from offline.converter_marker import (
    MarkerResult, convert_single_smart, _assess_vietnamese_quality,
)
from src.postprocess import apply_all_postprocessing

console = Console()
logger = logging.getLogger("offline")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def _add_file_logger(output_dir: Path) -> logging.FileHandler:
    """Add a file handler that logs everything to pipeline.log in the output dir."""
    log_path = output_dir / "pipeline.log"
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)
    return fh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline offline: PDF sang Markdown (marker-pdf + Ollama)",
    )
    parser.add_argument("--input", "-i", type=str, help="Đường dẫn đến 1 file PDF cụ thể")
    parser.add_argument("--input-dir", type=str, default="inputs", help="Thư mục chứa các file PDF")
    parser.add_argument("--output-dir", type=str, default="outputs_offline", help="Thư mục lưu kết quả")
    parser.add_argument(
        "--device", type=str, default="mps", choices=["mps", "cpu", "cuda"],
        help="Thiết bị chạy marker (mặc định: mps)",
    )
    parser.add_argument(
        "--strategy", type=str, default="auto", choices=VALID_STRATEGIES,
        help="Chiến lược trích xuất: auto|marker|pdftext|hybrid (mặc định: auto)",
    )
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR trên tất cả trang")
    parser.add_argument("--use-ollama", action="store_true", help="Sử dụng Ollama local LLM")
    parser.add_argument("--ollama-model", type=str, default="qwen3-vl:8b", help="Model Ollama")
    parser.add_argument("--no-postprocess", action="store_true", help="Bỏ qua post-processing")
    parser.add_argument("--no-quality-check", action="store_true", help="Bỏ qua quality check")
    parser.add_argument("--no-spot-check", action="store_true", help="Bỏ qua spot-check")
    parser.add_argument("--no-auto-fix", action="store_true", help="Bỏ qua auto-fix")
    parser.add_argument("--verbose", "-v", action="store_true", help="Hiển thị log chi tiết")
    return parser.parse_args()


def _count_phases(config: OfflineConfig, do_postprocess: bool, do_quality: bool,
                  do_spot: bool, do_fix: bool) -> int:
    """Count total phases for progress display."""
    n = 2  # convert + save
    if do_postprocess:
        n += 1
    if config.use_ollama:
        n += 1  # ollama polish / diacritics repair
    if do_quality:
        n += 1
    if do_spot and config.use_ollama:
        n += 1
    if do_fix and config.use_ollama:
        n += 1
    return n


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    config: OfflineConfig,
    do_postprocess: bool = True,
    do_quality: bool = True,
    do_spot: bool = True,
    do_fix: bool = True,
) -> dict:
    """Process a single PDF through the full offline pipeline with Rich progress."""
    stem = pdf_path.stem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)
    sub_dir = output_dir / safe_stem
    sub_dir.mkdir(parents=True, exist_ok=True)

    file_handler = _add_file_logger(sub_dir)

    total_phases = _count_phases(config, do_postprocess, do_quality, do_spot, do_fix)
    phase = 0
    info: dict = {"filename": pdf_path.name}

    logger.info("=" * 60)
    logger.info("Pipeline offline: %s", pdf_path.name)
    logger.info("Strategy: %s | Device: %s | Ollama: %s",
                config.extraction_strategy, config.marker_device, config.use_ollama)
    logger.info("=" * 60)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Phase 1: Convert
        phase += 1
        task = progress.add_task(
            f"[{phase}/{total_phases}] Convert ({config.extraction_strategy})...", total=None
        )
        logger.info("Phase %d: Convert (strategy=%s)", phase, config.extraction_strategy)

        result = convert_single_smart(
            pdf_path,
            device=config.marker_device,
            languages=config.marker_languages,
            force_ocr=False,
            strategy=config.extraction_strategy,
        )
        md = result.markdown
        info["page_count"] = result.page_count
        info["convert_time"] = result.elapsed_seconds
        info["strategy_used"] = result.strategy_used

        vn_ratio = _assess_vietnamese_quality(md)
        info["vn_ratio_raw"] = vn_ratio

        progress.update(
            task, completed=True,
            description=(
                f"[{phase}/{total_phases}] Convert: {result.page_count} trang, "
                f"{len(md):,} chars, VN={vn_ratio:.1%}, strategy={result.strategy_used}"
            ),
        )
        logger.info(
            "Phase %d done: %d pages, %d chars, VN=%.1f%%, strategy=%s, %.1fs",
            phase, result.page_count, len(md), vn_ratio * 100,
            result.strategy_used, result.elapsed_seconds,
        )

        # Phase: Post-processing
        if do_postprocess:
            phase += 1
            task = progress.add_task(f"[{phase}/{total_phases}] Post-processing...", total=None)
            logger.info("Phase %d: Post-processing", phase)

            md_before = len(md)
            md = apply_all_postprocessing(md, total_pages=result.page_count)
            md_after = len(md)

            progress.update(
                task, completed=True,
                description=f"[{phase}/{total_phases}] Post-processing: {md_before:,} -> {md_after:,} chars",
            )
            logger.info("Phase %d done: %d -> %d chars", phase, md_before, md_after)

        # Phase: Ollama polish / diacritics repair
        if config.use_ollama:
            phase += 1
            vn_ratio_now = _assess_vietnamese_quality(md)

            if vn_ratio_now < 0.05:
                task = progress.add_task(
                    f"[{phase}/{total_phases}] Ollama diacritics repair (VN={vn_ratio_now:.1%})...",
                    total=None,
                )
                logger.info("Phase %d: Diacritics repair (VN=%.1f%%)", phase, vn_ratio_now * 100)
                try:
                    from offline.polisher_ollama import repair_vietnamese_diacritics
                    md = repair_vietnamese_diacritics(
                        markdown=md, pdf_path=pdf_path,
                        model=config.ollama_model, base_url=config.ollama_base_url,
                        render_dpi=config.render_dpi,
                    )
                    new_ratio = _assess_vietnamese_quality(md)
                    progress.update(
                        task, completed=True,
                        description=(
                            f"[{phase}/{total_phases}] Diacritics repair: "
                            f"VN {vn_ratio_now:.1%} -> {new_ratio:.1%}"
                        ),
                    )
                    logger.info("Phase %d done: VN %.1f%% -> %.1f%%", phase, vn_ratio_now * 100, new_ratio * 100)
                except Exception as e:
                    progress.update(task, completed=True, description=f"[{phase}/{total_phases}] Diacritics repair: failed")
                    logger.warning("Phase %d failed: %s", phase, e)
            else:
                task = progress.add_task(
                    f"[{phase}/{total_phases}] Ollama polish ({config.ollama_model})...",
                    total=None,
                )
                logger.info("Phase %d: Ollama polish", phase)
                try:
                    from offline.polisher_ollama import polish_with_ollama
                    md = polish_with_ollama(
                        markdown=md, pdf_path=pdf_path,
                        model=config.ollama_model, base_url=config.ollama_base_url,
                        render_dpi=config.render_dpi,
                    )
                    progress.update(
                        task, completed=True,
                        description=f"[{phase}/{total_phases}] Ollama polish: {len(md):,} chars",
                    )
                    logger.info("Phase %d done: %d chars", phase, len(md))
                except Exception as e:
                    progress.update(task, completed=True, description=f"[{phase}/{total_phases}] Ollama polish: failed")
                    logger.warning("Phase %d failed: %s", phase, e)

        # Phase: Save output
        phase += 1
        task = progress.add_task(f"[{phase}/{total_phases}] Lưu output...", total=None)

        header = (
            f"<!-- Chuyển đổi offline từ PDF sang Markdown (marker-pdf) -->\n"
            f"<!-- File gốc: {result.filename} -->\n"
            f"<!-- Số trang: {result.page_count} -->\n"
            f"<!-- Strategy: {result.strategy_used} -->\n"
            f"<!-- Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n\n"
        )
        md = header + md
        output_path = sub_dir / f"output_{stem}.md"
        output_path.write_text(md, encoding="utf-8")
        info["output_path"] = str(output_path)
        info["md_chars"] = len(md)

        vn_final = _assess_vietnamese_quality(md)
        info["vn_ratio_final"] = vn_final

        progress.update(
            task, completed=True,
            description=f"[{phase}/{total_phases}] Saved: {len(md):,} chars, VN={vn_final:.1%}",
        )
        logger.info("Saved output: %s (%d chars, VN=%.1f%%)", output_path, len(md), vn_final * 100)

        # Phase: Quality check
        quality_score = None
        if do_quality:
            phase += 1
            task = progress.add_task(f"[{phase}/{total_phases}] Quality check...", total=None)
            logger.info("Phase %d: Quality check", phase)

            from offline.quality_offline import check_quality_offline
            qr = check_quality_offline(
                pdf_path=pdf_path, markdown=md, total_pages=result.page_count,
                use_ollama=config.use_ollama, ollama_model=config.ollama_model,
                ollama_base_url=config.ollama_base_url,
            )
            qr.save(sub_dir / "quality_report.json")
            quality_score = qr.overall_score
            info["quality_score"] = quality_score

            progress.update(
                task, completed=True,
                description=f"[{phase}/{total_phases}] Chất lượng: {quality_score:.1f}/10",
            )
            logger.info("Phase %d done: overall=%.1f/10", phase, quality_score)
            for issue in qr.issues:
                logger.info("  Issue: %s", issue)

        # Phase: Spot-check
        spot_report = None
        if do_spot and config.use_ollama:
            phase += 1
            task = progress.add_task(f"[{phase}/{total_phases}] Spot-check 5 vị trí...", total=None)
            logger.info("Phase %d: Spot-check", phase)

            from offline.spot_check_offline import run_spot_check_offline
            spot_report = run_spot_check_offline(
                pdf_path=pdf_path, markdown=md, total_pages=result.page_count,
                check_count=5, check_dpi=config.render_dpi,
                ollama_model=config.ollama_model, ollama_base_url=config.ollama_base_url,
            )
            spot_report.save(sub_dir / "spot_check_report.json")
            spot_report.save_log(sub_dir / "spot_check.log")
            info["spot_critical"] = spot_report.critical_count
            info["spot_warning"] = spot_report.warning_count
            info["spot_ok"] = spot_report.ok_count

            desc = f"[{phase}/{total_phases}] Spot-check: {spot_report.ok_count} OK"
            if spot_report.critical_count:
                desc += f", {spot_report.critical_count} critical"
            if spot_report.warning_count:
                desc += f", {spot_report.warning_count} warning"
            progress.update(task, completed=True, description=desc)
            logger.info("Phase %d done: %s", phase, spot_report.summary())

        # Phase: Auto-fix
        fix_report = None
        if (
            do_fix and config.use_ollama
            and spot_report
            and (spot_report.critical_count + spot_report.warning_count) > 0
        ):
            phase += 1
            total_fixable = spot_report.critical_count + spot_report.warning_count
            task = progress.add_task(f"[{phase}/{total_phases}] Auto-fix {total_fixable} lỗi...", total=None)
            logger.info("Phase %d: Auto-fix %d issues", phase, total_fixable)

            from offline.auto_fix_offline import auto_fix_offline
            md, fix_report = auto_fix_offline(
                pdf_path=pdf_path, markdown=md, spot_report=spot_report,
                check_dpi=config.render_dpi,
                ollama_model=config.ollama_model, ollama_base_url=config.ollama_base_url,
            )
            fix_report.save(sub_dir / "auto_fix_report.json")
            fix_report.save_log(sub_dir / "auto_fix.log")
            info["fixed_count"] = fix_report.fixed_count
            info["fix_failed"] = fix_report.failed_count

            output_path.write_text(md, encoding="utf-8")

            desc = f"[{phase}/{total_phases}] Auto-fix: {fix_report.fixed_count} sửa"
            if fix_report.failed_count:
                desc += f", {fix_report.failed_count} thất bại"
            progress.update(task, completed=True, description=desc)
            logger.info("Phase %d done: %s", phase, fix_report.summary())

    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    return info


def _print_summary_table(all_info: list[dict]) -> None:
    """Print a Rich summary table at the end."""
    if not all_info:
        return

    console.rule("[bold green]TÓM TẮT KẾT QUẢ OFFLINE")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan", max_width=50)
    table.add_column("Trang", justify="right")
    table.add_column("Ký tự", justify="right")
    table.add_column("VN%", justify="right")
    table.add_column("Strategy", justify="center")
    table.add_column("Chất lượng", justify="right")
    table.add_column("Spot-check", justify="right")
    table.add_column("Fixed", justify="right")
    table.add_column("Thời gian", justify="right")

    for info in all_info:
        qs = info.get("quality_score")
        quality_str = f"{qs:.1f}/10" if qs is not None else "N/A"

        vn = info.get("vn_ratio_final", 0)
        vn_str = f"{vn:.1%}"

        sc_c = info.get("spot_critical", 0)
        sc_w = info.get("spot_warning", 0)
        sc_o = info.get("spot_ok", 0)
        if sc_c + sc_w + sc_o > 0:
            parts = []
            if sc_c:
                parts.append(f"{sc_c}C")
            if sc_w:
                parts.append(f"{sc_w}W")
            parts.append(f"{sc_o}OK")
            spot_str = "/".join(parts)
        else:
            spot_str = "N/A"

        fc = info.get("fixed_count")
        ff = info.get("fix_failed")
        fix_str = f"{fc}/{fc + (ff or 0)}" if fc is not None else "N/A"

        table.add_row(
            info.get("filename", "?")[:48],
            str(info.get("page_count", "?")),
            f"{info.get('md_chars', 0):,}",
            vn_str,
            info.get("strategy_used", "?"),
            quality_str,
            spot_str,
            fix_str,
            f"{info.get('convert_time', 0):.1f}s",
        )

    console.print(table)
    console.print()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    config = OfflineConfig(
        marker_device=args.device,
        extraction_strategy=args.strategy,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        output_dir=Path(args.output_dir),
        input_dir=Path(args.input_dir),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    do_pp = not args.no_postprocess
    do_q = not args.no_quality_check
    do_s = not args.no_spot_check
    do_f = not args.no_auto_fix

    console.print("[bold]Pipeline Offline: marker-pdf[/bold]")
    console.print(
        f"[dim]Device: {config.marker_device} | Strategy: {config.extraction_strategy} | "
        f"Ollama: {config.use_ollama} | Quality: {do_q} | "
        f"Spot-check: {do_s} | Auto-fix: {do_f}[/dim]\n"
    )

    start_all = time.time()

    if args.input:
        pdf_path = Path(args.input)
        if not pdf_path.exists():
            console.print(f"[red]File không tồn tại: {pdf_path}[/red]")
            sys.exit(1)

        info = process_single_pdf(
            pdf_path, config.output_dir, config,
            do_postprocess=do_pp, do_quality=do_q, do_spot=do_s, do_fix=do_f,
        )
        total_time = time.time() - start_all

        _print_summary_table([info])
        console.print(f"[bold green]Hoàn thành! Output: {info.get('output_path')}[/bold green]")
        console.print(f"[dim]Tổng thời gian: {total_time:.1f}s[/dim]")

    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            console.print(f"[red]Thư mục không tồn tại: {input_dir}[/red]")
            sys.exit(1)

        pdf_files = sorted(input_dir.glob("*.pdf"))
        if not pdf_files:
            console.print("[red]Không tìm thấy file PDF nào.[/red]")
            sys.exit(1)

        console.print(f"[bold]Tìm thấy {len(pdf_files)} file PDF:[/bold]")
        for i, f in enumerate(pdf_files, 1):
            size_mb = f.stat().st_size / (1024 * 1024)
            console.print(f"  {i}. {f.name} ({size_mb:.1f} MB)")
        console.print()

        all_info: list[dict] = []
        for i, pdf_file in enumerate(pdf_files, 1):
            console.rule(f"[bold cyan]File {i}/{len(pdf_files)}: {pdf_file.name}")
            try:
                info = process_single_pdf(
                    pdf_file, config.output_dir, config,
                    do_postprocess=do_pp, do_quality=do_q, do_spot=do_s, do_fix=do_f,
                )
                all_info.append(info)
                console.print(f"[green]Saved: {info.get('output_path')}[/green]\n")
            except Exception as e:
                console.print(f"[red]LỖI xử lý {pdf_file.name}: {e}[/red]\n")
                logger.exception("Failed to process %s", pdf_file.name)

        total_time = time.time() - start_all
        _print_summary_table(all_info)
        console.print(f"[bold green]Hoàn thành {len(all_info)} file trong {total_time:.1f}s[/bold green]")


if __name__ == "__main__":
    main()
