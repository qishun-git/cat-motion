from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import typer
from rich.logging import RichHandler

from cat_motion import AppConfig, load_config
from cat_motion.paths import ensure_dir
from cat_motion.utils import preprocess_face, save_label_map
from cat_motion.vision.embedding import EmbeddingExtractor, compute_centroids

logger = logging.getLogger("cat_motion.training")
app = typer.Typer(help="Train/update embedding centroids from labeled face crops.")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MAX_IMAGES_PER_LABEL = 1000
SIMILARITY_THRESHOLD = 0.02


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else (base / value)


def _gather_samples(training_dir: Path) -> Dict[str, List[Path]]:
    samples: Dict[str, List[Path]] = {}
    if not training_dir.exists():
        return samples
    for label_dir in sorted(p for p in training_dir.iterdir() if p.is_dir()):
        images = [img for img in sorted(label_dir.glob("*")) if img.suffix.lower() in IMAGE_EXTS]
        if images:
            samples[label_dir.name] = images
    return samples


def _load_embedding(image_path: Path, face_size: int, extractor: EmbeddingExtractor) -> Optional[np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning("Failed to load %s; skipping.", image_path)
        return None
    face = preprocess_face(image, size=(face_size, face_size))
    return extractor.extract(face)


def _dedup_and_cap_label(
    label: str,
    images: List[Path],
    face_size: int,
    extractor: EmbeddingExtractor,
) -> List[np.ndarray]:
    kept: List[Tuple[Path, np.ndarray, float]] = []
    duplicates_removed = 0
    for image_path in images:
        embedding = _load_embedding(image_path, face_size, extractor)
        if embedding is None:
            continue
        is_duplicate = False
        for _, existing_emb, _ in kept:
            if float(np.linalg.norm(existing_emb - embedding)) < SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
        if is_duplicate:
            try:
                image_path.unlink()
            except OSError:
                pass
            duplicates_removed += 1
            continue
        kept.append((image_path, embedding, image_path.stat().st_mtime))
    if duplicates_removed:
        logger.info("Removed %s near-duplicate image(s) for %s.", duplicates_removed, label)

    trimmed = 0
    if len(kept) > MAX_IMAGES_PER_LABEL:
        kept.sort(key=lambda entry: entry[2])
        overflow = len(kept) - MAX_IMAGES_PER_LABEL
        victims = kept[:overflow]
        for victim_path, _, _ in victims:
            try:
                victim_path.unlink()
            except OSError:
                pass
        kept = kept[overflow:]
        trimmed = len(victims)
    if trimmed:
        logger.info(
            "Trimmed %s old image(s) for %s to enforce %s-image cap.",
            trimmed,
            label,
            MAX_IMAGES_PER_LABEL,
        )
    return [embedding for _, embedding, _ in kept]


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config TOML."),
    training_dir: Optional[Path] = typer.Option(None, "--training-dir", help="Override training directory."),
    embeddings_path: Optional[Path] = typer.Option(None, "--embeddings-out", help="Output embeddings path."),
    labels_path: Optional[Path] = typer.Option(None, "--labels-out", help="Output labels path."),
    device: Optional[str] = typer.Option(None, "--device", help="Torch device override, e.g. cpu"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _setup_logging(verbose)
    cfg = load_config(config_path)
    paths = cfg.paths()
    training_root = training_dir or paths.training
    if not training_root.exists():
        raise FileNotFoundError(f"Training directory not found: {training_root}")
    samples = _gather_samples(training_root)
    if not samples:
        raise RuntimeError(f"No labeled images found under {training_root}")

    embeddings_out = _resolve(paths.base, embeddings_path or cfg.recognition.embeddings)
    labels_out = _resolve(paths.base, labels_path or cfg.recognition.labels)

    extractor = EmbeddingExtractor(
        input_size=cfg.recognition.embedding_input,
        device=device or cfg.recognition.device,
    )

    embeddings: List = []
    label_ids: List[int] = []
    label_index: Dict[str, int] = {}
    next_label_id = 0

    for label_name, images in samples.items():
        if label_name not in label_index:
            label_index[label_name] = next_label_id
            next_label_id += 1
        label_id = label_index[label_name]
        logger.info("Processing %s (%s images)", label_name, len(images))
        label_embeddings = _dedup_and_cap_label(
            label_name,
            images,
            cfg.recognition.face_size,
            extractor,
        )
        if not label_embeddings:
            logger.warning("No usable embeddings for %s after cleanup; skipping.", label_name)
            continue
        embeddings.extend(label_embeddings)
        label_ids.extend([label_id] * len(label_embeddings))

    if not embeddings:
        raise RuntimeError("No embeddings generated; ensure images are readable.")

    model = compute_centroids(embeddings, label_ids)
    ensure_dir(embeddings_out.parent)
    model.save(embeddings_out)
    reverse_map = {idx: name for name, idx in label_index.items()}
    save_label_map(reverse_map, labels_out)
    logger.info("Saved embeddings to %s", embeddings_out)
    logger.info("Saved labels to %s", labels_out)


def cli() -> None:
    app()


__all__ = ["run", "cli"]
