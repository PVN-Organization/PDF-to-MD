#!/usr/bin/env python3
"""PDF to Markdown Pipeline — CLI Entry Point."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.config import Config

console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chuyển đổi PDF văn bản pháp quy tiếng Việt sang Markdown bằng Gemini AI",
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
        default="outputs",
        help="Thư mục lưu kết quả (mặc định: outputs/)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Chỉ phân tích PDF, không chuyển đổi",
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
        help="Bỏ qua bước tự động sửa lỗi sau spot-check",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10,
        help="Số trang mỗi chunk (mặc định: 10)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI khi render trang (mặc định: 300)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="Tên model Gemini cho convert gốc (mặc định: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--verify-model",
        type=str,
        default="gemini-2.5-flash-lite",
        help="Tên model Gemini cho verify/polish (mặc định: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--skip-confidence",
        type=float,
        default=0.9,
        help="Ngưỡng confidence OCR để skip Gemini (mặc định: 0.9)",
    )
    parser.add_argument(
        "--no-tesseract",
        action="store_true",
        help="Không dùng Tesseract OCR cho trang scanned",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Số request đồng thời tối đa (mặc định: 1)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=12,
        help="Giây chờ giữa các chunk (mặc định: 12, tăng nếu bị rate limit)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Xóa cache cũ trước khi chạy (buộc Gemini convert lại tất cả)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chạy thử không gọi Gemini (analyze + plan + render only)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị log chi tiết",
    )
    return parser.parse_args()


async def run_analyze_only(config: Config, pdf_files: list[Path]) -> None:
    from src.analyzer import analyze_pdf
    from rich.table import Table

    table = Table(title="Kết quả phân tích PDF", show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan", max_width=50)
    table.add_column("Trang", justify="right")
    table.add_column("MB", justify="right")
    table.add_column("Scanned %", justify="right")
    table.add_column("Có bảng", justify="center")
    table.add_column("Có ảnh", justify="center")
    table.add_column("~Tokens", justify="right")

    for pdf_file in pdf_files:
        analysis = analyze_pdf(pdf_file)
        table.add_row(
            analysis.filename[:48],
            str(analysis.total_pages),
            f"{analysis.file_size_mb:.1f}",
            f"{analysis.scanned_ratio * 100:.0f}%",
            "✓" if analysis.has_tables else "✗",
            "✓" if analysis.has_images else "✗",
            f"{analysis.estimated_tokens:,}",
        )

    console.print(table)


async def run_dry_run(config: Config, pdf_files: list[Path]) -> None:
    """Test pipeline without calling Gemini: analyze, plan, render."""
    from src.analyzer import analyze_pdf
    from src.planner import create_plan
    from src.renderer import render_pages
    from rich.table import Table

    console.print("[bold yellow]CHẾ ĐỘ DRY-RUN: Phân tích + Lên kế hoạch + Render (không gọi Gemini)[/bold yellow]\n")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_files:
        console.rule(f"[cyan]{pdf_file.name}")

        analysis = analyze_pdf(pdf_file)
        console.print(f"  Trang: {analysis.total_pages} | Scanned: {analysis.scanned_ratio*100:.0f}% | Tokens: ~{analysis.estimated_tokens:,}")

        chunks = create_plan(analysis, config)
        console.print(f"  Chunks: {len(chunks)}")
        for c in chunks:
            console.print(f"    Chunk {c.chunk_id}: trang {c.start_page+1}-{c.end_page} ({c.strategy}, ~{c.estimated_tokens:,} tokens)")

        stem = pdf_file.stem
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in stem)
        render_result = render_pages(pdf_file, chunks, config, safe_stem)
        console.print(f"  Rendered: {len(render_result.pages)} trang")
        console.print()

    console.print("[bold green]Dry-run hoàn thành! Pipeline sẵn sàng — chỉ cần thêm GEMINI_API_KEY vào .env[/bold green]")


async def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    config = Config(
        gemini_model=args.model,
        gemini_verify_model=args.verify_model,
        default_chunk_pages=args.chunk_size,
        max_concurrent_requests=args.concurrency,
        inter_chunk_delay=args.delay,
        render_dpi=args.dpi,
        output_dir=Path(args.output_dir),
        skip_gemini_confidence=args.skip_confidence,
        use_tesseract=not args.no_tesseract,
    )

    # Determine PDF files to process
    if args.input:
        pdf_path = Path(args.input)
        if not pdf_path.exists():
            console.print(f"[red]File không tồn tại: {pdf_path}[/red]")
            sys.exit(1)
        pdf_files = [pdf_path]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            console.print(f"[red]Thư mục không tồn tại: {input_dir}[/red]")
            sys.exit(1)
        pdf_files = sorted(input_dir.glob("*.pdf"))
        if not pdf_files:
            console.print(f"[red]Không tìm thấy file PDF trong {input_dir}[/red]")
            sys.exit(1)

    if args.analyze_only:
        await run_analyze_only(config, pdf_files)
        return

    if args.dry_run:
        await run_dry_run(config, pdf_files)
        return

    # Validate API key
    config.validate()

    console.print(f"[dim]Model: {config.gemini_model}[/dim]")

    if args.clear_cache:
        import shutil
        cache_root = config.temp_dir
        if cache_root.exists():
            shutil.rmtree(cache_root)
            console.print("[yellow]Cache đã được xóa.[/yellow]")
        else:
            console.print("[dim]Không có cache để xóa.[/dim]")

    from src.pipeline import process_batch, process_pdf

    do_quality = not args.no_quality_check
    do_spot = not args.no_spot_check
    do_fix = not args.no_auto_fix

    if args.input:
        result = await process_pdf(
            pdf_files[0], config,
            run_quality_check=do_quality,
            run_spot_check_flag=do_spot,
            run_auto_fix_flag=do_fix,
        )
        console.print(
            f"\n[bold green]Hoàn thành! Output: {result.output_path}[/bold green]"
        )
    else:
        results = await process_batch(
            Path(args.input_dir), config,
            run_quality_check=do_quality,
            run_spot_check_flag=do_spot,
            run_auto_fix_flag=do_fix,
        )
        console.print(
            f"\n[bold green]Hoàn thành {len(results)} file![/bold green]"
        )


if __name__ == "__main__":
    asyncio.run(main())
