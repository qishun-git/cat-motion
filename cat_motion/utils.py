from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .paths import ensure_dir


def load_label_map(path: Path | str) -> Dict[int, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(key): value for key, value in payload.items()}


def save_label_map(mapping: Dict[int, str], path: Path | str) -> None:
    serializable = {int(idx): name for idx, name in sorted(mapping.items())}
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def preprocess_face(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize/cast face crops for recognition."""

    if image is None or image.size == 0:
        raise ValueError("Empty face crop.")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def move_unique(src: Path, dest_dir: Path) -> Path:
    dest_dir = ensure_dir(dest_dir)
    candidate = dest_dir / src.name
    counter = 1
    while candidate.exists():
        candidate = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    src.rename(candidate)
    return candidate


def safe_join(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise ValueError(f"Path {relative} escapes {root}")
    return candidate
