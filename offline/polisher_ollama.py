"""Optional Ollama-based polish for offline pipeline.

Uses a local vision LLM (e.g. Qwen3-VL) to verify and improve markdown
output by comparing against rendered PDF page images.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = """\
Bạn là chuyên gia kiểm tra văn bản pháp quy tiếng Việt dạng Markdown.

Đối chiếu đoạn Markdown bên dưới với hình ảnh trang PDF gốc. Sửa lỗi:
- Sai chính tả, thiếu dấu thanh
- Sai số hiệu, ngày tháng
- Thiếu nội dung so với hình ảnh
- Bảng biểu bị vỡ cấu trúc

CHỈ trả về Markdown đã sửa. KHÔNG giải thích.

Đoạn Markdown cần kiểm tra:
```
{chunk}
```
"""


def _check_ollama_available(base_url: str) -> bool:
    try:
        import ollama
        client = ollama.Client(host=base_url)
        client.list()
        return True
    except Exception:
        return False


def _split_into_chunks(text: str, max_chars: int = 3000) -> list[str]:
    """Split markdown into chunks, trying to break at paragraph boundaries."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))
    return chunks


def polish_with_ollama(
    markdown: str,
    pdf_path: Path | str | None = None,
    model: str = "qwen3-vl:8b",
    base_url: str = "http://localhost:11434",
    render_dpi: int = 150,
) -> str:
    """Polish markdown using local Ollama vision model.

    If pdf_path is provided and the model supports vision, page images
    will be sent alongside the markdown for verification.
    Without pdf_path, only text-based correction is performed.
    """
    import ollama

    if not _check_ollama_available(base_url):
        logger.warning("Ollama not available at %s, skipping polish", base_url)
        return markdown

    client = ollama.Client(host=base_url)

    page_images: list[bytes] = []
    if pdf_path:
        pdf_path = Path(pdf_path)
        if pdf_path.exists():
            try:
                import fitz
                doc = fitz.open(str(pdf_path))
                for page in doc:
                    pix = page.get_pixmap(dpi=render_dpi)
                    page_images.append(pix.tobytes("png"))
                doc.close()
                logger.info("Rendered %d page images for Ollama polish", len(page_images))
            except Exception as e:
                logger.warning("Could not render PDF pages: %s", e)

    chunks = _split_into_chunks(markdown)
    polished_chunks: list[str] = []

    for i, chunk in enumerate(chunks):
        prompt = _VERIFY_PROMPT.replace("{chunk}", chunk[:4000])

        messages = [{"role": "user", "content": prompt}]

        if page_images:
            page_idx = min(i * len(page_images) // len(chunks), len(page_images) - 1)
            img_b64 = base64.b64encode(page_images[page_idx]).decode()
            messages = [{
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }]

        try:
            response = client.chat(
                model=model,
                messages=messages,
                options={"temperature": 0.1},
            )
            result = response.get("message", {}).get("content", "")
            if result and len(result) > len(chunk) * 0.3:
                polished_chunks.append(result)
            else:
                polished_chunks.append(chunk)
        except Exception as e:
            logger.warning("Ollama polish failed for chunk %d: %s", i, e)
            polished_chunks.append(chunk)

        if (i + 1) % 5 == 0:
            logger.info("Polished %d/%d chunks", i + 1, len(chunks))

    logger.info("Ollama polish complete: %d chunks processed", len(polished_chunks))
    return "\n".join(polished_chunks)
