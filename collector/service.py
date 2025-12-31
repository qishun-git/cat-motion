from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import cv2
import typer
from rich.logging import RichHandler
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from cat_motion import AppConfig, load_config

logger = logging.getLogger("cat_motion.collector")
app = typer.Typer(help="Ingest finished Motion clips into the cat-motion workspace.")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )


def probe_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return float(frames / fps) if fps > 0 else 0.0
    finally:
        cap.release()


class MotionClipCollector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.paths = config.paths()
        self.recorder_cfg = config.recorder
        self._pending: Dict[Path, float] = {}
        self._processed: Dict[Path, float] = {}

    def run(self, poll_interval: float = 2.0, use_watchdog: bool = True) -> None:
        drop_dir = self.recorder_cfg.motion_drop_dir
        logger.info("Collecting clips from %s", drop_dir)
        observer: Optional[Observer] = None
        if use_watchdog and drop_dir.exists():
            observer = Observer()
            handler = _MotionEventHandler(self)
            observer.schedule(handler, str(drop_dir), recursive=False)
            observer.start()
        elif use_watchdog:
            logger.warning("Watchdog requested but %s does not exist; falling back to polling.", drop_dir)
        try:
            while True:
                self.scan_once()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Collector stopped by user.")
        finally:
            if observer:
                observer.stop()
                observer.join()

    def scan_once(self) -> None:
        drop_dir = self.recorder_cfg.motion_drop_dir
        if not drop_dir.exists():
            logger.warning("Motion drop dir %s not found.", drop_dir)
            return
        self._cleanup_processed()
        for path in drop_dir.glob(f"*{self.recorder_cfg.target_extension}"):
            self._track_candidate(path)
        self._flush_ready()

    def _cleanup_processed(self) -> None:
        missing = [path for path in self._processed if not path.exists()]
        for path in missing:
            self._processed.pop(path, None)

    def _track_candidate(self, path: Path) -> None:
        if not path.is_file():
            return
        if self.recorder_cfg.enforce_extension and path.suffix.lower() != self.recorder_cfg.target_extension.lower():
            return
        resolved = path.resolve()
        if self._already_processed(resolved):
            return
        self._pending.setdefault(resolved, 0.0)

    def _already_processed(self, path: Path) -> bool:
        processed_ts = self._processed.get(path)
        if processed_ts is None:
            return False
        try:
            return abs(processed_ts - path.stat().st_mtime) < 0.5
        except FileNotFoundError:
            return False

    def _flush_ready(self) -> None:
        for path in list(self._pending.keys()):
            if not path.exists():
                self._pending.pop(path, None)
                continue
            last_size = self._pending[path]
            stat_size = path.stat().st_size
            if stat_size <= 0:
                self._pending[path] = stat_size
                continue
            if stat_size != last_size:
                self._pending[path] = stat_size
                continue
            age = time.time() - path.stat().st_mtime
            if age < self.recorder_cfg.settle_time_s:
                continue
            self._ingest(path)
            self._pending.pop(path, None)
            self._processed[path] = path.stat().st_mtime

    def _ingest(self, source: Path) -> None:
        duration = probe_duration(source)
        if duration < self.recorder_cfg.min_duration_s:
            logger.debug("Skip %s (duration %.2fs < %.2fs)", source.name, duration, self.recorder_cfg.min_duration_s)
            return
        timestamp = datetime.fromtimestamp(source.stat().st_mtime, tz=timezone.utc)
        safe_name = timestamp.strftime("%Y%m%dT%H%M%SZ")
        dest_name = f"{safe_name}_{source.name}"
        dest_path = self.paths.clips / dest_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists():
            logger.debug("Destination %s already exists; skipping copy.", dest_path)
            self._remove_source(source)
            return
        try:
            shutil.copy2(source, dest_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to copy %s -> %s: %s", source, dest_path, exc)
            return
        sidecar = dest_path.with_suffix(dest_path.suffix + ".json")
        payload = {
            "source": str(source),
            "captured_at": timestamp.isoformat(),
            "duration": duration,
            "size_bytes": dest_path.stat().st_size,
        }
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Copied %s -> %s", source.name, dest_path.name)
        self._remove_source(source)

    def _remove_source(self, source: Path) -> None:
        try:
            source.unlink()
            logger.debug("Removed source clip %s", source)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Failed to remove source clip %s: %s", source, exc)


class _MotionEventHandler(FileSystemEventHandler):
    def __init__(self, collector: MotionClipCollector) -> None:

        self.collector = collector

    def on_created(self, event) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        self.collector._track_candidate(Path(event.src_path))

    def on_moved(self, event) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        self.collector._track_candidate(Path(event.dest_path))

    def on_modified(self, event) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        self.collector._track_candidate(Path(event.src_path))


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to cat_motion config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
    poll_interval: float = typer.Option(2.0, "--interval", "-i", help="Polling interval seconds."),
    watch: bool = typer.Option(True, "--watch/--no-watch", help="Use watchdog file notifications."),
) -> None:
    """Start the collector loop."""

    _setup_logging(verbose)
    cfg = load_config(config_path)
    collector = MotionClipCollector(cfg)
    collector.run(poll_interval=poll_interval, use_watchdog=watch)


def cli() -> None:
    app()


if __name__ == "__main__":
    cli()
