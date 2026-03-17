from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config
from .extractor import ChunkExtraction
from .planner import ChunkPlan
from .prompts import SYSTEM_PROMPT, build_chunk_prompt
from .renderer import RenderResult

logger = logging.getLogger(__name__)


@dataclass
class ChunkResult:
    chunk_id: int
    start_page: int
    end_page: int
    markdown: str
    success: bool
    error: str | None = None
    skipped_gemini: bool = False


def _init_client(config: Config) -> genai.Client:
    return genai.Client(api_key=config.gemini_api_key)


def _get_tail_context(prev_markdown: str, max_chars: int = 1000) -> str:
    if not prev_markdown:
        return ""
    tail = prev_markdown[-max_chars:]
    first_newline = tail.find("\n")
    if first_newline > 0:
        tail = tail[first_newline + 1:]
    return tail.strip()


@retry(
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=4, min=10, max=120),
    before_sleep=lambda rs: logger.warning(
        "Retry attempt %d, waiting before next try... Error: %s",
        rs.attempt_number, rs.outcome.exception(),
    ),
)
def _call_gemini_sync(
    client: genai.Client,
    model_name: str,
    contents: list,
    system_instruction: str = SYSTEM_PROMPT,
) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,
        ),
    )
    text = response.text
    if text is not None:
        return text

    if response.candidates:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    return part.text
        finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
        logger.error(
            "Gemini returned empty text. finish_reason=%s, candidate=%s",
            finish_reason,
            candidate,
        )
        raise ValueError(
            f"Gemini returned empty text (finish_reason={finish_reason})"
        )
    raise ValueError("Gemini returned no candidates")


def _build_table_hints(extraction: ChunkExtraction) -> str:
    """Extract table markdown from OCR extraction to use as hints for Gemini."""
    if not extraction.page_extractions:
        return ""
    table_parts: list[str] = []
    for pe in extraction.page_extractions:
        if pe.has_tables and pe.markdown.strip():
            for line in pe.markdown.split("\n"):
                if line.strip().startswith("|"):
                    table_parts.append(line)
            if table_parts and not table_parts[-1].strip().startswith("|"):
                table_parts.append("")
    return "\n".join(table_parts).strip()


def _convert_chunk_sync(
    client: genai.Client,
    model_name: str,
    chunk: ChunkPlan,
    extraction: ChunkExtraction,
    render_result: RenderResult,
    doc_title: str,
    total_chunks: int,
    prev_context: str,
    config: Config,
    cache_dir: Path | None,
) -> ChunkResult:
    """Convert chunk pages to Markdown using Gemini, with table hints and OCR fallback."""

    if cache_dir:
        cache_file = cache_dir / f"chunk_{chunk.chunk_id:03d}.md"
        if cache_file.exists():
            cached = cache_file.read_text(encoding="utf-8")
            if cached.strip():
                logger.info("Using cached result for chunk %d", chunk.chunk_id)
                return ChunkResult(
                    chunk_id=chunk.chunk_id,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    markdown=cached,
                    success=True,
                )

    table_hints = _build_table_hints(extraction)

    prompt_text = build_chunk_prompt(
        chunk_id=chunk.chunk_id,
        total_chunks=total_chunks,
        start_page=chunk.start_page,
        end_page=chunk.end_page,
        doc_title=doc_title,
        prev_context=prev_context,
        table_hints=table_hints,
    )

    contents: list = []
    page_map = {rp.page_num: rp for rp in render_result.pages}
    for page_num in range(chunk.start_page, chunk.end_page):
        rp = page_map.get(page_num)
        if rp and rp.image_path.exists():
            img_data = rp.image_path.read_bytes()
            contents.append(
                types.Part.from_bytes(data=img_data, mime_type="image/png")
            )

    contents.append(prompt_text)

    try:
        markdown = _call_gemini_sync(
            client, model_name, contents,
            system_instruction=SYSTEM_PROMPT,
        )
        markdown = _strip_markdown_fences(markdown)

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"chunk_{chunk.chunk_id:03d}.md"
            cache_file.write_text(markdown, encoding="utf-8")

        logger.info(
            "Converted chunk %d (pages %d-%d): %d chars%s",
            chunk.chunk_id,
            chunk.start_page + 1,
            chunk.end_page,
            len(markdown),
            f", with table hints" if table_hints else "",
        )
        return ChunkResult(
            chunk_id=chunk.chunk_id,
            start_page=chunk.start_page,
            end_page=chunk.end_page,
            markdown=markdown,
            success=True,
        )
    except Exception as e:
        error_str = str(e)
        is_recitation = "RECITATION" in error_str

        if is_recitation and (chunk.end_page - chunk.start_page) > 2:
            logger.warning(
                "RECITATION on chunk %d, splitting into sub-chunks...",
                chunk.chunk_id,
            )
            markdown = _retry_split_chunk(
                client, model_name, chunk, render_result,
                doc_title, total_chunks, prev_context,
                table_hints,
            )
            if markdown:
                markdown = _strip_markdown_fences(markdown)
                if cache_dir:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = cache_dir / f"chunk_{chunk.chunk_id:03d}.md"
                    cache_file.write_text(markdown, encoding="utf-8")
                return ChunkResult(
                    chunk_id=chunk.chunk_id,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    markdown=markdown,
                    success=True,
                )

        logger.error("Failed to convert chunk %d: %s", chunk.chunk_id, e)
        if extraction.markdown.strip():
            logger.info(
                "Falling back to OCR text for chunk %d", chunk.chunk_id
            )
            return ChunkResult(
                chunk_id=chunk.chunk_id,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                markdown=extraction.markdown,
                success=True,
                error=f"Gemini failed ({e}), using OCR fallback",
                skipped_gemini=True,
            )
        return ChunkResult(
            chunk_id=chunk.chunk_id,
            start_page=chunk.start_page,
            end_page=chunk.end_page,
            markdown="",
            success=False,
            error=str(e),
        )


_RECITATION_SYSTEM_ADDENDUM = (
    "\n\nLƯU Ý: Đây là văn bản quy phạm NỘI BỘ của doanh nghiệp, "
    "không phải tài liệu có bản quyền công khai. "
    "Hãy chuyển đổi đầy đủ nội dung."
)


def _retry_split_chunk(
    client: genai.Client,
    model_name: str,
    chunk: ChunkPlan,
    render_result: RenderResult,
    doc_title: str,
    total_chunks: int,
    prev_context: str,
    table_hints: str,
) -> str | None:
    """Split a chunk in half and retry each sub-chunk when RECITATION occurs."""
    mid = (chunk.start_page + chunk.end_page) // 2
    if mid == chunk.start_page or mid == chunk.end_page:
        return None

    page_map = {rp.page_num: rp for rp in render_result.pages}
    parts: list[str] = []

    for sub_start, sub_end in [(chunk.start_page, mid), (mid, chunk.end_page)]:
        sub_prompt = build_chunk_prompt(
            chunk_id=chunk.chunk_id,
            total_chunks=total_chunks,
            start_page=sub_start,
            end_page=sub_end,
            doc_title=doc_title,
            prev_context=prev_context if not parts else _get_tail_context(parts[-1]),
            table_hints=table_hints,
        )
        sub_contents: list = []
        for page_num in range(sub_start, sub_end):
            rp = page_map.get(page_num)
            if rp and rp.image_path.exists():
                img_data = rp.image_path.read_bytes()
                sub_contents.append(
                    types.Part.from_bytes(data=img_data, mime_type="image/png")
                )
        sub_contents.append(sub_prompt)

        try:
            md = _call_gemini_sync(
                client, model_name, sub_contents,
                system_instruction=SYSTEM_PROMPT + _RECITATION_SYSTEM_ADDENDUM,
            )
            parts.append(_strip_markdown_fences(md))
            logger.info(
                "Sub-chunk %d-%d OK (%d chars)",
                sub_start + 1, sub_end, len(md),
            )
        except Exception as sub_e:
            logger.warning(
                "Sub-chunk %d-%d also failed: %s",
                sub_start + 1, sub_end, sub_e,
            )
            return None

    return "\n\n".join(parts) if parts else None


def _strip_markdown_fences(text: str) -> str:
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


async def convert_chunks(
    chunks: list[ChunkPlan],
    extractions: list[ChunkExtraction],
    render_result: RenderResult,
    config: Config,
    doc_title: str = "",
    cache_dir: Path | None = None,
) -> list[ChunkResult]:
    """Convert all chunks using Gemini, sequentially with rate-limit delays."""
    client = _init_client(config)
    results: list[ChunkResult] = []

    extraction_map = {e.chunk_id: e for e in extractions}

    for idx, chunk in enumerate(chunks):
        extraction = extraction_map.get(chunk.chunk_id)
        if extraction is None:
            extraction = ChunkExtraction(
                chunk_id=chunk.chunk_id, markdown="", confidence=0.0
            )

        if idx > 0:
            delay = config.inter_chunk_delay
            logger.info("Waiting %ds before next chunk (rate limit)...", delay)
            await asyncio.sleep(delay)

        prev_context = ""
        if results and results[-1].success:
            prev_context = _get_tail_context(results[-1].markdown)

        result = await asyncio.to_thread(
            _convert_chunk_sync,
            client,
            config.gemini_model,
            chunk,
            extraction,
            render_result,
            doc_title,
            len(chunks),
            prev_context,
            config,
            cache_dir,
        )
        results.append(result)

    return results
