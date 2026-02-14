#!/usr/bin/env python3
"""
OCR CLI — Rich terminal UI for the OCR pipeline.

Usage:
    python ocr_cli.py document.pdf --lang eng --backend auto
    python ocr_cli.py ./docs/ --recursive --lang eng+hin --workers 4
    python ocr_cli.py image.png --lang mar --backend ollama --model qwen2.5-vl
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ocr_engine import (
    BatchResult,
    OCRConfig,
    collect_inputs,
    detect_best_model,
    ocr_batch,
    write_output,
)

console = Console()


def setup_logging(quiet: bool = False) -> None:
    """Configure Rich-styled logging."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments (all bugs from ocr_doc.py fixed)."""
    parser = argparse.ArgumentParser(
        description="OCR Pipeline — Extract text from PDFs and images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python ocr_cli.py document.pdf
  python ocr_cli.py document.pdf --lang hin --backend ollama
  python ocr_cli.py ./scans/ --recursive --lang eng+hin+mar --workers 4
  python ocr_cli.py image.png --backend pytesseract --lang eng
""",
    )
    parser.add_argument("inputs", nargs="+", help="Input files or directories")
    parser.add_argument(
        "--backend",
        choices=["auto", "ollama", "pytesseract"],
        default="auto",
        help="OCR backend (default: auto)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="Languages: eng, hin, mar, eng+hin, eng+hin+mar (default: eng)",
    )
    parser.add_argument("--model", type=str, default=None, help="Ollama model name (auto-detected if omitted)")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for Ollama (default: 0.1)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per page (default: 3)")
    parser.add_argument("--max-dim", type=int, default=2048, help="Max image dimension (default: 2048)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per page in seconds (default: 120)")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers (default: 3)")
    parser.add_argument("--output-dir", type=str, default="ocr_output", help="Output directory (default: ocr_output)")
    parser.add_argument("--format", choices=["json", "txt", "md"], default="json", help="Output format (default: json)")
    parser.add_argument("--indent", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--save-debug-img", action="store_true", help="Save preprocessed images")
    parser.add_argument("--recursive", action="store_true", help="Recursively process directories")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Ollama base URL")
    return parser.parse_args()


def show_config_panel(args: argparse.Namespace, file_count: int, model: str | None) -> None:
    """Display configuration as a Rich panel."""
    lines = [
        f"[bold]Backend:[/bold]    {args.backend}",
        f"[bold]Languages:[/bold]  {args.lang}",
        f"[bold]Model:[/bold]      {model or '(auto-detect)'}",
        f"[bold]Files:[/bold]      {file_count}",
        f"[bold]Workers:[/bold]    {args.workers}",
        f"[bold]Max dim:[/bold]    {args.max_dim}",
        f"[bold]Output:[/bold]     {args.output_dir} ({args.format})",
    ]
    panel = Panel("\n".join(lines), title="OCR Pipeline", border_style="blue")
    console.print(panel)


def show_results_table(batch: BatchResult) -> None:
    """Display results summary as a Rich table."""
    table = Table(title="Results Summary", show_lines=True)
    table.add_column("File", style="cyan", max_width=40)
    table.add_column("Status", justify="center")
    table.add_column("Pages", justify="right")
    table.add_column("Backend", style="dim")
    table.add_column("Confidence", justify="right")
    table.add_column("Time", justify="right")

    for r in batch.results:
        name = Path(r.input_path).name
        status_style = {"success": "[green]OK[/green]", "partial": "[yellow]Partial[/yellow]", "error": "[red]Error[/red]"}
        status = status_style.get(r.status, r.status)
        page_count = str(len(r.pages))

        avg_conf = 0.0
        if r.pages:
            confs = [p.confidence for p in r.pages if p.confidence > 0]
            avg_conf = sum(confs) / len(confs) if confs else 0.0

        table.add_row(
            name,
            status,
            page_count,
            r.backend,
            f"{avg_conf:.0f}%",
            f"{r.processing_time:.1f}s",
        )

    console.print(table)

    # Summary line
    total_pages = sum(len(r.pages) for r in batch.results)
    ok_pages = sum(sum(1 for p in r.pages if p.status == "success") for r in batch.results)
    console.print(
        f"\n[bold]Total:[/bold] {batch.successful_files}/{batch.total_files} files, "
        f"{ok_pages}/{total_pages} pages, "
        f"{batch.total_time:.1f}s elapsed"
    )


async def run_cli(args: argparse.Namespace) -> None:
    """Main async CLI runner."""
    languages = args.lang.split("+")
    config = OCRConfig(
        backend=args.backend,
        model=args.model,
        languages=languages,
        temperature=args.temp,
        max_retries=args.max_retries,
        max_dim=args.max_dim,
        timeout=args.timeout,
        workers=args.workers,
        ollama_base_url=args.ollama_url,
        save_debug_img=args.save_debug_img,
    )

    # Collect files
    files = collect_inputs(args.inputs, recursive=args.recursive)
    if not files:
        console.print("[red]No valid input files found.[/red]")
        sys.exit(1)

    # Auto-detect model
    if config.backend in ("auto", "ollama") and config.model is None:
        config.model = detect_best_model(config.languages, config.ollama_base_url)

    # Show config
    if not args.quiet:
        show_config_panel(args, len(files), config.model)

    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        disable=args.quiet,
    ) as progress:
        task_id = progress.add_task("Processing files...", total=len(files))

        def on_file_progress(file_path: str, current: int, total: int) -> None:
            progress.update(task_id, completed=current, description=f"[cyan]{Path(file_path).name}[/cyan]")

        batch = await ocr_batch(
            input_paths=args.inputs,
            config=config,
            recursive=args.recursive,
            progress_callback=on_file_progress,
        )

    # Write output
    output_dir = Path(args.output_dir)
    write_output(batch.results, output_dir, fmt=args.format, indent=args.indent)

    # Show results table
    if not args.quiet:
        console.print()
        show_results_table(batch)
        console.print(f"\n[green]Output written to:[/green] {output_dir}/")


def main() -> None:
    """Entry point with graceful Ctrl+C handling."""
    args = parse_args()
    setup_logging(quiet=args.quiet)

    # Graceful Ctrl+C
    def handle_sigint(sig, frame):
        console.print("\n[yellow]Interrupted — partial results may be available.[/yellow]")
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        asyncio.run(run_cli(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
