#!/usr/bin/env python3
"""Offline PDF to Markdown Pipeline — no internet required.

Uses marker-pdf for conversion, with optional Ollama local LLM for polishing,
quality check, spot-check, and auto-fix.
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
from rich.table import Table

from offline.config_offline import OfflineConfig
from offline.converter_marker import convert_single, convert_batch, MarkerResult
from src.postprocess import apply_all_postprocessing

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline offline: PDF sang Markdown (marker-pdf + Ollama)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Đường dẫn đến 1 file PDF cụ thể",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="inputs",
        help="Thư mục chứa các file PDF (mặc định: inputs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_offline",
        help="Thư mục lưu kết quả (mặc định: outputs_offline/)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Thiết bị chạy marker (mặc định: mps cho Apple Silicon)",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR trên tất cả trang (chậm hơn nhưng chính xác hơn với PDF scanned)",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Sử dụng Ollama local LLM để polish kết quả",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen3-vl:8b",
        help="Model Ollama để polish (mặc định: qwen3-vl:8b)",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Bỏ qua bước post-processing",
    )
    parser.add_argument(
        "--no-quality-check",
        action="store_true",
        help="Bỏ qua bước kiểm tra chất lượng",
    )
    parser.add_argument(
        "--no-spot-check",
        action="store_true",
        help="Bỏ qua bước spot-check ngẫu nhiên",
    )
    parser.add_argument(
        "--no-auto-fix",
        action="store_true",
        help="Bỏ qua bước tự động sửa lỗi",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị log chi tiết",
    )
    return parser.parse_args()


def _process_single(
    result: MarkerResult,
    pdf_path: Path,
    output_dir: Path,
    config: OfflineConfig,
    do_postprocess: bool,
    do_quality: bool,
    do_spot: bool,
    do_fix: bool,
) -> dict:
    """Process a single conversion result through post-processing, quality, spot-check, auto-fix."""
    stem = Path(result.filename).stem
    safe_stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)
    sub_dir = output_dir / safe_stem
    sub_dir.mkdir(parents=True, exist_ok=True)

    md = result.markdown
    info: dict = {
        "filename": result.filename,
        "page_count": result.page_count,
        "convert_time": result.elapsed_seconds,
    }

    # Step 1: Post-processing
    if do_postprocess:
        console.print(f"  [dim]Post-processing...[/dim]")
        md = apply_all_postprocessing(md, total_pages=result.page_count)

    # Step 2: Ollama polish (optional)
    if config.use_ollama:
        try:
            from offline.polisher_ollama import polish_with_ollama
            console.print(f"  [dim]Ollama polish ({config.ollama_model})...[/dim]")
            md = polish_with_ollama(
                markdown=md,
                pdf_path=pdf_path,
                model=config.ollama_model,
                base_url=config.ollama_base_url,
            )
        except ImportError:
            console.print("  [yellow]ollama package not installed, skipping polish[/yellow]")
        except Exception as e:
            console.print(f"  [yellow]Ollama polish failed: {e}[/yellow]")

    # Step 3: Save initial output
    header = (
        f"<!-- Chuyển đổi offline từ PDF sang Markdown (marker-pdf) -->\n"
        f"<!-- File gốc: {result.filename} -->\n"
        f"<!-- Số trang: {result.page_count} -->\n"
        f"<!-- Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -->\n\n"
    )
    md = header + md

    output_path = sub_dir / f"output_{stem}.md"
    output_path.write_text(md, encoding="utf-8")
    info["output_path"] = str(output_path)
    info["md_chars"] = len(md)

    # Step 4: Quality check
    quality_score = None
    if do_quality:
        console.print(f"  [dim]Quality check...[/dim]")
        from offline.quality_offline import check_quality_offline
        qr = check_quality_offline(
            pdf_path=pdf_path,
            markdown=md,
            total_pages=result.page_count,
            use_ollama=config.use_ollama,
            ollama_model=config.ollama_model,
            ollama_base_url=config.ollama_base_url,
        )
        qr.save(sub_dir / "quality_report.json")
        quality_score = qr.overall_score
        info["quality_score"] = quality_score
        console.print(f"  [dim]Chất lượng: {quality_score:.1f}/10[/dim]")

    # Step 5: Spot-check
    spot_report = None
    if do_spot and config.use_ollama:
        console.print(f"  [dim]Spot-check 5 vị trí...[/dim]")
        from offline.spot_check_offline import run_spot_check_offline
        spot_report = run_spot_check_offline(
            pdf_path=pdf_path,
            markdown=md,
            total_pages=result.page_count,
            check_count=5,
            check_dpi=150,
            ollama_model=config.ollama_model,
            ollama_base_url=config.ollama_base_url,
        )
        spot_report.save(sub_dir / "spot_check_report.json")
        spot_report.save_log(sub_dir / "spot_check.log")
        info["spot_critical"] = spot_report.critical_count
        info["spot_warning"] = spot_report.warning_count
        info["spot_ok"] = spot_report.ok_count
        console.print(f"  [dim]{spot_report.summary()}[/dim]")

    # Step 6: Auto-fix
    fix_report = None
    if (
        do_fix
        and config.use_ollama
        and spot_report
        and (spot_report.critical_count + spot_report.warning_count) > 0
    ):
        total_fixable = spot_report.critical_count + spot_report.warning_count
        console.print(f"  [dim]Auto-fix {total_fixable} lỗi...[/dim]")
        from offline.auto_fix_offline import auto_fix_offline
        md, fix_report = auto_fix_offline(
            pdf_path=pdf_path,
            markdown=md,
            spot_report=spot_report,
            check_dpi=150,
            ollama_model=config.ollama_model,
            ollama_base_url=config.ollama_base_url,
        )
        fix_report.save(sub_dir / "auto_fix_report.json")
        fix_report.save_log(sub_dir / "auto_fix.log")
        info["fixed_count"] = fix_report.fixed_count
        info["fix_failed"] = fix_report.failed_count

        output_path.write_text(md, encoding="utf-8")
        console.print(f"  [dim]{fix_report.summary()}[/dim]")

    return info


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    config = OfflineConfig(
        marker_device=args.device,
        use_ollama=args.use_ollama,
        ollama_model=args.ollama_model,
        output_dir=Path(args.output_dir),
        input_dir=Path(args.input_dir),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    do_postprocess = not args.no_postprocess
    do_quality = not args.no_quality_check
    do_spot = not args.no_spot_check
    do_fix = not args.no_auto_fix

    console.print("[bold]Pipeline Offline: marker-pdf[/bold]")
    console.print(f"[dim]Device: {config.marker_device} | Ollama: {config.use_ollama} | "
                  f"Quality: {do_quality} | Spot-check: {do_spot} | Auto-fix: {do_fix}[/dim]\n")

    start_all = time.time()

    if args.input:
        pdf_path = Path(args.input)
        if not pdf_path.exists():
            console.print(f"[red]File không tồn tại: {pdf_path}[/red]")
            sys.exit(1)

        result = convert_single(
            pdf_path,
            device=config.marker_device,
            languages=config.marker_languages,
            force_ocr=args.force_ocr,
        )
        info = _process_single(
            result, pdf_path, config.output_dir, config,
            do_postprocess, do_quality, do_spot, do_fix,
        )
        total_time = time.time() - start_all
        console.print(f"\n[bold green]Hoàn thành! Output: {info.get('output_path')}[/bold green]")
        console.print(f"[dim]{result.page_count} trang, {info.get('md_chars', 0):,} chars, {total_time:.1f}s[/dim]")

    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            console.print(f"[red]Thư mục không tồn tại: {input_dir}[/red]")
            sys.exit(1)

        results = convert_batch(
            input_dir,
            device=config.marker_device,
            languages=config.marker_languages,
            force_ocr=args.force_ocr,
        )

        if not results:
            console.print("[red]Không tìm thấy file PDF nào.[/red]")
            sys.exit(1)

        all_info = []
        pdf_files = sorted(input_dir.glob("*.pdf"))
        for r, pdf_file in zip(results, pdf_files):
            console.print(f"\n[cyan]Processing: {r.filename}[/cyan]")
            info = _process_single(
                r, pdf_file, config.output_dir, config,
                do_postprocess, do_quality, do_spot, do_fix,
            )
            all_info.append(info)
            console.print(f"[green]Saved: {info.get('output_path')}[/green]")

        total_time = time.time() - start_all

        table = Table(title="Kết quả offline", show_header=True, header_style="bold magenta")
        table.add_column("File", style="cyan", max_width=50)
        table.add_column("Trang", justify="right")
        table.add_column("Ký tự", justify="right")
        table.add_column("Chất lượng", justify="right")
        table.add_column("Spot-check", justify="right")
        table.add_column("Fixed", justify="right")
        table.add_column("Thời gian", justify="right")

        for i, info in enumerate(all_info):
            r = results[i]
            qs = info.get("quality_score")
            quality_str = f"{qs:.1f}/10" if qs is not None else "N/A"

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
            if fc is not None:
                fix_str = f"{fc}/{fc + (ff or 0)}"
            else:
                fix_str = "N/A"

            table.add_row(
                r.filename[:48],
                str(r.page_count),
                f"{info.get('md_chars', 0):,}",
                quality_str,
                spot_str,
                fix_str,
                f"{r.elapsed_seconds:.1f}s",
            )

        console.print(table)
        console.print(f"\n[bold green]Hoàn thành {len(results)} file trong {total_time:.1f}s[/bold green]")


if __name__ == "__main__":
    main()
