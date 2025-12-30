from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import typer
from rich.logging import RichHandler

from cat_motion import AppConfig, load_config
from cat_motion.paths import ensure_dir
from cat_motion.utils import load_label_map, preprocess_face
from cat_motion.vision.detection import BaseDetector, Box, create_detector
from cat_motion.vision.embedding import EmbeddingExtractor, EmbeddingModel, EmbeddingRecognizer

logger = logging.getLogger("cat_motion.processor")
app = typer.Typer(help="Run clip processing + recognition.")


@dataclass
class DetectionSample:
    frame_idx: int
    detection_idx: int
    face: Any


@dataclass
class ClipResult:
    summary: Dict[str, object]
    best_label: Optional[str]
    detections: List[DetectionSample]
    recognized_samples: List[Tuple[str, float, Any]]
    fps: float
    total_frames: int
    trim_start: int
    trim_end: int
    frame_annotations: Dict[int, List[Tuple[Box, Optional[str], float]]]


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )


def _resolve_path(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path)


class ClipProcessor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.paths = config.paths()
        self.processing_cfg = config.processing
        self._detector = self._build_detector()
        self._recognizer = self._build_recognizer()
        self._label_map = self._load_label_map()
        self.frame_stride = max(1, int(self.processing_cfg.frame_stride))
        self.face_size = (config.recognition.face_size, config.recognition.face_size)
        self.detection_interval = float(self.processing_cfg.detection_interval)
        self.trim_padding = max(0.0, float(self.processing_cfg.trim_padding_seconds))
        self.training_refresh_count = max(0, int(self.processing_cfg.training_refresh_count))
        self.recognition_margin = max(0.0, float(self.processing_cfg.recognition_margin))
        self.unlabeled_save_limit = max(0, int(self.processing_cfg.unlabeled_save_limit))

    def _build_detector(self) -> BaseDetector:
        detection_cfg = self.config.detection
        model_path = _resolve_path(self.paths.base, Path(detection_cfg.model))
        return create_detector(
            model_path=model_path,
            input_size=detection_cfg.input_size,
            class_ids=detection_cfg.class_ids,
            conf_threshold=detection_cfg.conf_threshold,
            iou_threshold=detection_cfg.iou_threshold,
            providers=detection_cfg.providers,
        )

    def _build_recognizer(self) -> Optional[EmbeddingRecognizer]:
        recognition_cfg = self.config.recognition
        embeddings_path = _resolve_path(self.paths.base, Path(recognition_cfg.embeddings))
        try:
            model = EmbeddingModel.load(embeddings_path)
        except FileNotFoundError:
            logger.warning("Embedding model %s not found; recognition disabled.", embeddings_path)
            return None
        extractor = EmbeddingExtractor(input_size=recognition_cfg.embedding_input, device=recognition_cfg.device)
        return EmbeddingRecognizer(model=model, extractor=extractor, threshold=recognition_cfg.threshold)

    def _load_label_map(self) -> Dict[int, str]:
        path = _resolve_path(self.paths.base, Path(self.config.recognition.labels))
        if not path.exists():
            logger.warning("Label map %s not found; all detections treated as unknown.", path)
            return {}
        return load_label_map(path)

    def process_pending(self, limit: Optional[int] = None) -> int:
        clips = sorted(self.paths.clips.glob("*.mp4"))
        processed = 0
        for clip in clips:
            if limit is not None and processed >= limit:
                break
            try:
                result = self._analyze_clip(clip)
                if result is None:
                    continue
                self._finalize_clip(clip, result)
                processed += 1
            except Exception as exc:  # pragma: no cover
                logger.exception("Failed to process %s: %s", clip.name, exc)
        logger.info("Processed %d clips", processed)
        return processed

    def _analyze_clip(self, clip: Path) -> Optional[ClipResult]:
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            logger.warning("Unable to open %s, skipping.", clip)
            return None
        detection_samples: List[DetectionSample] = []
        recognized_samples: List[Tuple[str, float, any]] = []
        frame_annotations: Dict[int, List[Tuple[Box, Optional[str], float]]] = {}
        frame_idx = 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        last_detection_ts = -float("inf")
        last_detections: List[Box] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if ((frame_idx - 1) % self.frame_stride) != 0:
                continue
            timestamp = frame_idx / (fps or 30.0)
            run_detector = False
            if self.detection_interval <= 0:
                run_detector = True
            elif timestamp - last_detection_ts >= self.detection_interval:
                run_detector = True
            if run_detector:
                faces = self._detector.detect(frame)
                last_detection_ts = timestamp
                last_detections = faces
            else:
                faces = last_detections
            for idx, (x, y, w, h) in enumerate(faces):
                crop = frame[max(0, y) : max(0, y) + max(1, h), max(0, x) : max(0, x) + max(1, w)]
                if crop.size == 0:
                    continue
                processed_face = preprocess_face(crop, self.face_size)
                detection_samples.append(DetectionSample(frame_idx=frame_idx, detection_idx=idx, face=processed_face))
                label_name: Optional[str] = None
                score_value = 0.0
                if self._recognizer:
                    prepped = processed_face
                    label_id, score_value = self._recognizer.predict(prepped)
                    if label_id != -1:
                        label_name = self._label_map.get(label_id)
                        if label_name:
                            recognized_samples.append((label_name, score_value, prepped))
                frame_annotations.setdefault(frame_idx, []).append(((x, y, w, h), label_name, float(score_value)))
        cap.release()

        detection_total = len(detection_samples)
        recognized_total = len(recognized_samples)
        if detection_total == 0:
            clip.unlink(missing_ok=True)
            logger.info("Deleted %s (no detections).", clip.name)
            return None

        counts: Dict[str, int] = {}
        label_best_score: Dict[str, float] = {}
        recognized_majority_needed = recognized_total // 2 + 1 if recognized_total else 0
        detection_majority_needed = detection_total // 2 + 1 if detection_total else 0
        best_label: Optional[str] = None
        best_score = float("-inf")

        if self._recognizer and recognized_samples:
            for label_name, score, _ in recognized_samples:
                counts[label_name] = counts.get(label_name, 0) + 1
                label_best_score[label_name] = max(score, label_best_score.get(label_name, float("-inf")))
            for label_name, count in counts.items():
                if count < recognized_majority_needed:
                    continue
                if detection_total and count < detection_majority_needed:
                    continue
                score = label_best_score[label_name]
                if score >= self._recognizer.threshold + self.recognition_margin and score > best_score:
                    best_label = label_name
                    best_score = score

        fps_value = fps or 30.0
        padding_frames = int(round(self.trim_padding * fps_value))
        frame_ids = [sample.frame_idx for sample in detection_samples]
        first_det = min(frame_ids)
        last_det = max(frame_ids)
        start_frame = max(1, first_det - padding_frames)
        end_frame = min(last_det + padding_frames, total_frames or last_det + padding_frames)

        summary = {
            "clip": clip.name,
            "fps": fps_value,
            "frames": total_frames,
            "detections_total": detection_total,
            "recognized_total": recognized_total,
            "recognized_majority_needed": recognized_majority_needed,
            "detection_majority_needed": detection_majority_needed,
            "highlight_label": best_label,
            "label_counts": counts,
            "label_best_scores": label_best_score,
            "frames_kept": (start_frame, end_frame),
        }
        serialized_frames = {
            str(idx): [
                {
                    "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    "label": label_name,
                    "score": float(score),
                }
                for (box, label_name, score) in entries
            ]
            for idx, entries in sorted(frame_annotations.items())
        }
        summary["frame_annotations"] = serialized_frames

        return ClipResult(
            summary=summary,
            best_label=best_label,
            detections=detection_samples,
            recognized_samples=recognized_samples,
            fps=fps_value,
            total_frames=total_frames,
            trim_start=start_frame,
            trim_end=end_frame,
            frame_annotations=frame_annotations,
        )

    def _finalize_clip(self, clip: Path, result: ClipResult) -> None:
        if self.processing_cfg.trim_clip:
            trim_clip_to_range(clip, result.trim_start, result.trim_end, result.fps)
        dest_dir = self.paths.recognized / result.best_label if result.best_label else self.paths.unknown
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_clip = self._move_clip_unique(clip, dest_dir)
        sidecar = dest_clip.with_suffix(dest_clip.suffix + ".json")
        sidecar.write_text(json.dumps(result.summary, indent=2), encoding="utf-8")
        if result.best_label:
            self._export_training_frames(result.best_label, result.recognized_samples, dest_clip.stem)
        else:
            self._export_unlabeled_faces(dest_clip.stem, result.detections, result.fps)
        if self.processing_cfg.enable_compression:
            self._write_compressed_copy(dest_clip)

    def _move_clip_unique(self, clip: Path, dest_dir: Path) -> Path:
        candidate = dest_dir / clip.name
        counter = 1
        while candidate.exists():
            candidate = dest_dir / f"{clip.stem}_{counter}{clip.suffix}"
            counter += 1
        shutil.move(str(clip), candidate)
        return candidate

    def _export_training_frames(self, label: str, samples: List[Tuple[str, float, any]], clip_stem: str) -> None:
        if not samples:
            logger.info("No recognized samples available for %s.", clip_stem)
            return
        candidates = [face for (label_name, _, face) in samples if label_name == label]
        if not candidates:
            candidates = [face for (_, _, face) in samples]
        if not candidates:
            return
        if self.training_refresh_count > 0 and len(candidates) > self.training_refresh_count:
            candidates = random.sample(candidates, self.training_refresh_count)
        target_dir = ensure_dir(self.paths.training / label)
        for idx, face in enumerate(candidates):
            filename = target_dir / f"{clip_stem}_auto_{idx}.png"
            cv2.imwrite(str(filename), face)
        logger.info("Promoted %s frame(s) to training/%s.", len(candidates), label)

    def _export_unlabeled_faces(self, clip_stem: str, samples: List[DetectionSample], fps: float) -> None:
        if not samples:
            logger.info("No detection samples to export for %s.", clip_stem)
            return
        chosen = samples
        if self.processing_cfg.unlabeled_per_second and fps > 0:
            seen_seconds: set[int] = set()
            dedup: List[DetectionSample] = []
            for sample in samples:
                bucket = int((sample.frame_idx - 1) / fps)
                if bucket in seen_seconds:
                    continue
                seen_seconds.add(bucket)
                dedup.append(sample)
            chosen = dedup
        if self.unlabeled_save_limit > 0 and len(chosen) > self.unlabeled_save_limit:
            chosen = random.sample(chosen, self.unlabeled_save_limit)
        folder = ensure_dir(self.paths.unlabeled / clip_stem)
        saved = 0
        for sample in chosen:
            filename = folder / f"f{sample.frame_idx}_n{sample.detection_idx}.png"
            cv2.imwrite(str(filename), sample.face)
            saved += 1
        logger.info("Saved %s face(s) from %s into %s.", saved, clip_stem, folder)

    def _write_compressed_copy(self, clip_path: Path) -> None:
        dest_dir = ensure_dir(self.paths.compressed)
        dest_path = dest_dir / clip_path.name
        if dest_path.exists():
            return
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if width == 0 or height == 0:
            cap.release()
            return
        scale = float(self.processing_cfg.compression_scale)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dest_path), fourcc, fps, new_size)
        if not writer.isOpened():
            cap.release()
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            writer.write(resized)
        cap.release()
        writer.release()


def trim_clip_to_range(clip_path: Path, start_frame: int, end_frame: int, fps: float) -> None:
    if start_frame <= 1 and end_frame <= 0:
        return
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width == 0 or height == 0:
        cap.release()
        return
    if fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    safe_start = max(1, min(start_frame, total_frames or start_frame))
    safe_end = max(safe_start, min(end_frame, total_frames or end_frame))
    if safe_start == 1 and (total_frames == 0 or safe_end >= total_frames):
        cap.release()
        return
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_path = clip_path.with_suffix(".trim_tmp" + clip_path.suffix)
    writer = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx < safe_start:
                continue
            if frame_idx > safe_end:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    temp_path.replace(clip_path)


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Process at most N clips."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _setup_logging(verbose)
    cfg = load_config(config_path)
    processor = ClipProcessor(cfg)
    processor.process_pending(limit=limit)


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
