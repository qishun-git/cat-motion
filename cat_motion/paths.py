from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class ProjectPaths:
    base: Path
    data: Path
    clips: Path
    recognized: Path
    unknown: Path
    unlabeled: Path
    training: Path
    models: Path
    templates: Path
    static: Path


def resolve_project_paths(base_dir: Path | str) -> ProjectPaths:
    base = Path(base_dir).resolve()
    data = ensure_dir(base / "data")
    clips = ensure_dir(data / "clips")
    recognized = ensure_dir(data / "recognized_clips")
    unknown = ensure_dir(data / "unknown_clips")
    unlabeled = ensure_dir(data / "unlabeled")
    training = ensure_dir(data / "training")
    models = ensure_dir(base / "models")
    templates = ensure_dir(base / "web" / "templates")
    static = ensure_dir(base / "web" / "static")
    return ProjectPaths(
        base=base,
        data=data,
        clips=clips,
        recognized=recognized,
        unknown=unknown,
        unlabeled=unlabeled,
        training=training,
        models=models,
        templates=templates,
        static=static,
    )
