from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = PROJECT_ROOT / "inputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs_offline"


@dataclass
class OfflineConfig:
    marker_device: str = "mps"
    marker_languages: tuple[str, ...] = ("vi", "en")

    use_ollama: bool = False
    ollama_model: str = "qwen3-vl:8b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chunk_tokens: int = 2000

    input_dir: Path = INPUTS_DIR
    output_dir: Path = OUTPUTS_DIR

    render_dpi: int = 150

    def validate(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
