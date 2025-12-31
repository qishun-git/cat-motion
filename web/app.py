from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from rich.logging import RichHandler

from cat_motion import AppConfig, load_config
from cat_motion.paths import ensure_dir
from cat_motion.utils import safe_join
from processor.pipeline import ClipProcessor

logger = logging.getLogger("cat_motion.web")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=False)],
    )


class WebState:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.paths = config.paths()
        self.templates = Jinja2Templates(directory=str(self.paths.templates))

    def refresh(self) -> None:
        self.__init__(load_config())

    def list_clips(self, root: Path, limit: int = 50) -> List[Dict[str, object]]:
        clips: List[Dict[str, object]] = []
        if not root.exists():
            return clips
        for clip in sorted(root.glob("**/*.mp4"), reverse=True)[:limit]:
            sidecar = clip.with_suffix(clip.suffix + ".json")
            summary: Optional[Dict[str, object]] = None
            if sidecar.exists():
                try:
                    summary = json.loads(sidecar.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    summary = None
            clips.append(
                {
                    "path": clip.relative_to(root).as_posix(),
                    "size_mb": round(clip.stat().st_size / (1024 * 1024), 2),
                    "summary": summary,
                }
            )
        return clips

    def list_unlabeled(self, root: Path) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        if not root.exists():
            return entries
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            entries.append(
                {
                    "name": folder.name,
                    "count": sum(1 for _ in folder.glob("*")),
                }
            )
        return entries

    def known_labels(self) -> List[str]:
        labels: List[str] = []
        for path in sorted(self.paths.training.iterdir()):
            if path.is_dir():
                labels.append(path.name)
        return labels


class ProcessRequest(BaseModel):
    limit: Optional[int] = None


class AssignRequest(BaseModel):
    label: str


def create_app(state: WebState) -> FastAPI:
    fastapi_app = FastAPI(title="Cat Motion Web")
    static_dir = state.paths.static
    static_dir.mkdir(parents=True, exist_ok=True)
    fastapi_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _category_root(category: str) -> Path:
        if category == "recognized":
            return state.paths.recognized
        if category == "unknown":
            return state.paths.unknown
        raise HTTPException(status_code=404, detail="Category not found")

    @fastapi_app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        recognized = state.list_clips(state.paths.recognized)
        unknown = state.list_clips(state.paths.unknown)
        unlabeled = state.list_unlabeled(state.paths.unlabeled)
        stream_cfg = state.config.web
        context = {
            "request": request,
            "recognized": recognized,
            "unknown": unknown,
            "unlabeled": unlabeled,
            "stream_url": stream_cfg.stream_url,
            "stream_port": stream_cfg.stream_port,
            "stream_path": stream_cfg.stream_path,
            "known_labels": state.known_labels(),
        }
        return state.templates.TemplateResponse("index.html", context)

    @fastapi_app.get("/api/clips/{category}")
    async def clips(category: str) -> List[Dict[str, object]]:
        root = _category_root(category)
        return state.list_clips(root, limit=200)

    @fastapi_app.get("/clips/{category}/{clip_path:path}")
    async def download_clip(category: str, clip_path: str) -> FileResponse:
        root = _category_root(category)
        target = (root / clip_path).resolve()
        if not target.exists() or root.resolve() not in target.parents and target != root.resolve():
            raise HTTPException(status_code=404, detail="Clip not found")
        return FileResponse(target)

    @fastapi_app.post("/api/process")
    async def trigger_processing(payload: ProcessRequest) -> JSONResponse:
        def _run() -> int:
            processor = ClipProcessor(state.config)
            return processor.process_pending(limit=payload.limit)

        processed = await asyncio.to_thread(_run)
        return JSONResponse({"processed": processed})

    @fastapi_app.post("/api/refresh")
    async def refresh_config() -> JSONResponse:
        state.refresh()
        return JSONResponse({"status": "ok"})

    def _unlabeled_folder(folder: str) -> Path:
        try:
            return safe_join(state.paths.unlabeled, folder)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid folder")

    @fastapi_app.post("/api/unlabeled/{folder}/assign")
    async def assign_unlabeled(folder: str, payload: AssignRequest) -> JSONResponse:
        label = payload.label.strip()
        if not label:
            raise HTTPException(status_code=400, detail="Label is required")
        if any(ch in label for ch in "/\\"):
            raise HTTPException(status_code=400, detail="Label contains invalid characters")
        src = _unlabeled_folder(folder)
        if not src.exists():
            raise HTTPException(status_code=404, detail="Folder not found")
        dest_dir = ensure_dir(state.paths.training / label)
        for image in sorted(src.glob("*")):
            if image.is_file():
                shutil.move(str(image), dest_dir / image.name)
        try:
            src.rmdir()
        except OSError:
            pass
        return JSONResponse({"status": "moved", "label": label})

    @fastapi_app.post("/api/unlabeled/{folder}/delete")
    async def delete_unlabeled(folder: str) -> JSONResponse:
        src = _unlabeled_folder(folder)
        if not src.exists():
            raise HTTPException(status_code=404, detail="Folder not found")
        shutil.rmtree(src)
        return JSONResponse({"status": "deleted"})

    return fastapi_app


def serve(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    host: Optional[str] = typer.Option(None, "--host", "-h"),
    port: Optional[int] = typer.Option(None, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    _setup_logging(verbose)
    cfg = load_config(config_path)
    if host:
        cfg.web.host = host  # type: ignore[attr-defined]
    if port:
        cfg.web.port = port  # type: ignore[attr-defined]
    state = WebState(cfg)
    fastapi_app = create_app(state)
    uvicorn.run(fastapi_app, host=cfg.web.host, port=cfg.web.port, reload=reload)


def cli() -> None:
    typer.run(serve)


state = WebState(load_config())
app = create_app(state)


if __name__ == "__main__":
    cli()
