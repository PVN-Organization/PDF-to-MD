from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEMP_DIR = PROJECT_ROOT / "temp"


@dataclass
class Config:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = "gemini-3-flash-preview"
    gemini_verify_model: str = "gemini-2.5-flash-lite"

    # Chunking
    default_chunk_pages: int = 10
    max_concurrent_requests: int = 1
    inter_chunk_delay: int = 12  # seconds between chunks

    # Rendering
    render_dpi: int = 300

    # OCR / Extraction
    use_tesseract: bool = True
    tesseract_lang: str = "vie"
    skip_gemini_confidence: float = 0.98

    # Paths
    input_dir: Path = INPUTS_DIR
    output_dir: Path = OUTPUTS_DIR
    temp_dir: Path = TEMP_DIR

    # Quality check
    quality_sample_pages: int = 5

    # Spot check
    spot_check_count: int = 5
    spot_check_dpi: int = 150

    def validate(self) -> None:
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Create a .env file with GEMINI_API_KEY=your_key or export it."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def check_quota(self) -> bool:
        """Pre-flight check: verify the API key has available quota."""
        from google import genai
        try:
            client = genai.Client(api_key=self.gemini_api_key)
            response = client.models.generate_content(
                model=self.gemini_model,
                contents="Respond with just: OK",
            )
            return bool(response.text)
        except Exception:
            return False
