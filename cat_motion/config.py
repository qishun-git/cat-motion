from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from .paths import ProjectPaths, resolve_project_paths


class RecorderConfig(BaseModel):
    motion_drop_dir: Path = Field(
        default=Path("/var/lib/motion-clips"),
        description="Directory where Motion daemon saves finished clips.",
    )
    settle_time_s: float = Field(default=3.0, ge=0.5, description="Seconds to wait after a file stops growing.")
    min_duration_s: float = Field(default=2.0, ge=0.0)
    target_extension: str = Field(default=".mp4")
    enforce_extension: bool = Field(default=True)


class DetectionConfig(BaseModel):
    model: Path = Path("models/detection.onnx")
    input_size: tuple[int, int] | int = 640
    class_ids: Optional[list[int]] = None
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    providers: Optional[list[str]] = None

    @field_validator("input_size")
    @classmethod
    def _validate_input_size(cls, value: tuple[int, int] | int) -> tuple[int, int] | int:
        if isinstance(value, tuple):
            return (int(value[0]), int(value[1]))
        return int(value)


class RecognitionConfig(BaseModel):
    embeddings: Path = Path("models/embeddings.npz")
    labels: Path = Path("models/labels.json")
    embedding_input: int = Field(default=224, ge=32)
    threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    face_size: int = Field(default=100, ge=32)
    device: Optional[str] = None


class ProcessingConfig(BaseModel):
    batch_size: int = Field(default=2, ge=1)
    frame_stride: int = Field(default=2, ge=1)
    detection_interval: float = Field(default=0.0, ge=0.0, description="Seconds between detector runs (0 = every frame).")
    trim_padding_seconds: float = Field(default=1.0, ge=0.0)
    recognition_margin: float = Field(default=0.05, ge=0.0)
    training_refresh_count: int = Field(default=10, ge=0)
    unlabeled_save_limit: int = Field(default=25, ge=0)
    unlabeled_per_second: bool = True
    trim_clip: bool = True


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)
    stream_url: Optional[str] = None
    stream_port: Optional[int] = Field(default=8081, ge=1, le=65535)
    stream_path: str = "/?action=stream"
    stream_content_type: str = "image/jpeg"
    templates_dir: Optional[Path] = None


class AppConfig(BaseModel):
    base_dir: Path = Path(".")
    recorder: RecorderConfig = RecorderConfig()
    detection: DetectionConfig = DetectionConfig()
    recognition: RecognitionConfig = RecognitionConfig()
    processing: ProcessingConfig = ProcessingConfig()
    web: WebConfig = WebConfig()

    def paths(self) -> ProjectPaths:
        return resolve_project_paths(self.base_dir)


def _load_from_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
        return data
    raise ValueError(f"Unsupported config extension: {path}")


def load_config(explicit_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from file or environment."""

    candidate: Optional[Path] = explicit_path
    if candidate is None:
        env_path = os.environ.get("CAT_MOTION_CONFIG")
        if env_path:
            candidate = Path(env_path)
    if candidate is None:
        default_candidate = Path("configs/cat_motion.yml")
        if not default_candidate.exists():
            repo_default = Path(__file__).resolve().parent.parent / "configs" / "cat_motion.yml"
            candidate = repo_default if repo_default.exists() else default_candidate
        else:
            candidate = default_candidate
    data = _load_from_file(candidate) if candidate.exists() else {}
    raw_base_dir = data.get("base_dir") or os.environ.get("CAT_MOTION_BASE", ".")
    base_path = Path(raw_base_dir)
    if not base_path.is_absolute():
        base_root = candidate.parent if candidate else Path.cwd()
        base_path = (base_root / base_path).resolve()
    else:
        base_path = base_path.resolve()
    data["base_dir"] = str(base_path)
    return AppConfig(**data)
