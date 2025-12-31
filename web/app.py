from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Literal, Optional

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from rich.logging import RichHandler

from cat_motion import AppConfig, load_config
from cat_motion.paths import ensure_dir
from cat_motion.training import run as train_embeddings_run
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

    def list_clips(self, root: Path, limit: Optional[int] = 50, offset: int = 0) -> List[Dict[str, object]]:
        clips: List[Dict[str, object]] = []
        if not root.exists():
            return clips
        skipped = 0
        for clip in sorted(root.glob("**/*.mp4"), reverse=True):
            if skipped < offset:
                skipped += 1
                continue
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
            if limit is not None and len(clips) >= limit:
                break
        return clips

    def count_clips(self, root: Path) -> int:
        if not root.exists():
            return 0
        return sum(1 for _ in root.glob("**/*.mp4"))

    def list_unlabeled_buckets(self) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        root = self.paths.unlabeled
        if not root.exists():
            return entries
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            count = sum(1 for _ in folder.glob("*.png"))
            if count > 0:
                entries.append({"name": folder.name, "count": count})
        return entries

    def list_unlabeled_images(self, folder: str, limit: Optional[int] = None, offset: int = 0) -> List[str]:
        root = self.paths.unlabeled / folder
        if not root.exists():
            return []
        images = sorted(img.name for img in root.glob("*.png"))
        if offset:
            images = images[offset:]
        if limit is not None:
            images = images[:limit]
        return images

    def known_labels(self) -> List[str]:
        labels: List[str] = []
        root = self.paths.training
        if not root.exists():
            return labels
        for path in sorted(root.iterdir()):
            if path.is_dir():
                labels.append(path.name)
        return labels

    def list_training_labels(self) -> List[Dict[str, object]]:
        entries: List[Dict[str, object]] = []
        root = self.paths.training
        if not root.exists():
            return entries
        for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            count = sum(1 for _ in label_dir.glob("*.png"))
            if count > 0:
                entries.append({"name": label_dir.name, "count": count})
        return entries

    def list_training_images(self, label: str, limit: Optional[int] = None, offset: int = 0) -> List[str]:
        target = self.paths.training / label
        if not target.exists():
            return []
        images = sorted(img.name for img in target.glob("*.png"))
        if offset:
            images = images[offset:]
        if limit is not None:
            images = images[:limit]
        return images


class ProcessRequest(BaseModel):
    limit: Optional[int] = None


class AssignRequest(BaseModel):
    label: str


class BulkAssignRequest(BaseModel):
    label: str
    images: List[str]


class BulkDeleteRequest(BaseModel):
    images: List[str]


class ClipAssignRequest(BaseModel):
    category: Literal["recognized", "unknown"]
    clip: str
    label: str


class ClipDeleteRequest(BaseModel):
    category: Literal["recognized", "unknown"]
    clip: str


class TrainingDeleteRequest(BaseModel):
    label: str
    image: str


class TrainingAssignRequest(BaseModel):
    current_label: str
    image: str
    new_label: str


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
        recognized_count = state.count_clips(state.paths.recognized)
        unknown_count = state.count_clips(state.paths.unknown)
        unlabeled_buckets = state.list_unlabeled_buckets()
        unlabeled_count = sum(bucket["count"] for bucket in unlabeled_buckets)
        training_labels = state.list_training_labels()
        training_count = sum(label["count"] for label in training_labels)
        stream_cfg = state.config.web
        context = {
            "request": request,
            "recognized_count": recognized_count,
            "unknown_count": unknown_count,
            "unlabeled_count": unlabeled_count,
            "training_count": training_count,
            "stream_url": stream_cfg.stream_url,
            "stream_port": stream_cfg.stream_port,
            "stream_path": stream_cfg.stream_path,
            "active_page": "dashboard",
        }
        return state.templates.TemplateResponse("index.html", context)

    @fastapi_app.get("/clips", response_class=HTMLResponse)
    async def clips_page(
        request: Request,
        recognized_page: int = Query(1, alias="recognized_page", ge=1),
        unknown_page: int = Query(1, alias="unknown_page", ge=1),
    ) -> HTMLResponse:
        per_page = 3

        def paginate_clips(root: Path, page: int) -> Dict[str, object]:
            offset = (page - 1) * per_page
            entries = state.list_clips(root, limit=per_page + 1, offset=offset)
            has_next = len(entries) > per_page
            display = entries[:per_page]
            return {
                "entries": display,
                "page": page,
                "has_prev": page > 1,
                "has_next": has_next,
                "next_url": str(
                    request.url.include_query_params(
                        **{
                            "recognized_page": page + 1 if root == state.paths.recognized else recognized_page,
                            "unknown_page": page + 1 if root == state.paths.unknown else unknown_page,
                        }
                    )
                )
                if has_next
                else None,
                "prev_url": str(
                    request.url.include_query_params(
                        **{
                            "recognized_page": page - 1 if root == state.paths.recognized else recognized_page,
                            "unknown_page": page - 1 if root == state.paths.unknown else unknown_page,
                        }
                    )
                )
                if page > 1
                else None,
            }

        recognized = paginate_clips(state.paths.recognized, recognized_page)
        unknown = paginate_clips(state.paths.unknown, unknown_page)
        context = {
            "request": request,
            "recognized": recognized,
            "unknown": unknown,
            "active_page": "clips",
        }
        return state.templates.TemplateResponse("clips.html", context)

    @fastapi_app.get("/unlabeled", response_class=HTMLResponse)
    async def unlabeled_page(
        request: Request,
        folder: Optional[str] = Query(None),
        page: int = Query(1, ge=1),
    ) -> HTMLResponse:
        buckets = state.list_unlabeled_buckets()
        per_page = 24
        selected_images: List[str] = []
        pagination: Dict[str, object] = {
            "page": page,
            "has_prev": False,
            "has_next": False,
            "prev_url": None,
            "next_url": None,
        }
        if folder:
            offset = (page - 1) * per_page
            entries = state.list_unlabeled_images(folder, limit=per_page + 1, offset=offset)
            has_next = len(entries) > per_page
            selected_images = entries[:per_page]
            pagination["has_prev"] = page > 1
            pagination["has_next"] = has_next
            if has_next:
                pagination["next_url"] = str(request.url.include_query_params(folder=folder, page=page + 1))
            if page > 1:
                pagination["prev_url"] = str(request.url.include_query_params(folder=folder, page=page - 1))
        context = {
            "request": request,
            "buckets": buckets,
            "selected_folder": folder,
            "selected_images": selected_images,
            "pagination": pagination,
            "known_labels": state.known_labels(),
            "active_page": "unlabeled",
        }
        return state.templates.TemplateResponse("unlabeled.html", context)

    @fastapi_app.get("/training", response_class=HTMLResponse)
    async def training_page(
        request: Request,
        label: Optional[str] = Query(None),
        page: int = Query(1, ge=1),
    ) -> HTMLResponse:
        labels = state.list_training_labels()
        per_page = 24
        selected_images: List[str] = []
        pagination: Dict[str, object] = {
            "page": page,
            "has_prev": False,
            "has_next": False,
            "prev_url": None,
            "next_url": None,
        }
        if label:
            offset = (page - 1) * per_page
            entries = state.list_training_images(label, limit=per_page + 1, offset=offset)
            has_next = len(entries) > per_page
            selected_images = entries[:per_page]
            pagination["has_prev"] = page > 1
            pagination["has_next"] = has_next
            if has_next:
                pagination["next_url"] = str(request.url.include_query_params(label=label, page=page + 1))
            if page > 1:
                pagination["prev_url"] = str(request.url.include_query_params(label=label, page=page - 1))
        context = {
            "request": request,
            "labels": labels,
            "selected_label": label,
            "selected_images": selected_images,
            "pagination": pagination,
            "active_page": "training",
        }
        return state.templates.TemplateResponse("training.html", context)

    @fastapi_app.get("/training/{label}/{image}")
    async def training_image(label: str, image: str) -> FileResponse:
        try:
            label_dir = safe_join(state.paths.training, label)
            target = safe_join(label_dir, image)
        except ValueError:
            raise HTTPException(status_code=404, detail="Invalid path")
        if not target.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(target)

    @fastapi_app.get("/api/clips/{category}")
    async def clips(
        category: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=200),
    ) -> List[Dict[str, object]]:
        root = _category_root(category)
        return state.list_clips(root, limit=limit, offset=offset)

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

    @fastapi_app.post("/api/train")
    async def trigger_training() -> JSONResponse:
        def _run() -> None:
            train_embeddings_run(
                config_path=None,
                training_dir=None,
                embeddings_path=None,
                labels_path=None,
                device=None,
                verbose=False,
            )

        try:
            await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training failed")
            raise HTTPException(status_code=500, detail=str(exc))
        return JSONResponse({"status": "trained"})

    @fastapi_app.post("/api/refresh")
    async def refresh_config() -> JSONResponse:
        state.refresh()
        return JSONResponse({"status": "ok"})

    def _validate_label(raw: str) -> str:
        label = raw.strip()
        if not label:
            raise HTTPException(status_code=400, detail="Label is required")
        if any(ch in label for ch in "/\\"):
            raise HTTPException(status_code=400, detail="Label contains invalid characters")
        return label

    def _unlabeled_folder(folder: str) -> Path:
        try:
            return safe_join(state.paths.unlabeled, folder)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid folder")

    def _unlabeled_image_path(folder: str, image: str) -> Path:
        folder_path = _unlabeled_folder(folder)
        try:
            return safe_join(folder_path, image)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid image path")

    def _cleanup_unlabeled_folder(folder: str) -> None:
        folder_path = _unlabeled_folder(folder)
        if not folder_path.exists():
            return
        has_png = any(child.is_file() and child.suffix.lower() == ".png" for child in folder_path.iterdir())
        if not has_png:
            shutil.rmtree(folder_path, ignore_errors=True)

    def _unique_image_names(names: List[str]) -> List[str]:
        seen: set[str] = set()
        image_names: List[str] = []
        for name in names:
            normalized = name.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            image_names.append(normalized)
        if not image_names:
            raise HTTPException(status_code=400, detail="No images provided")
        return image_names

    @fastapi_app.get("/unlabeled/{folder}/{image}")
    async def unlabeled_image(folder: str, image: str) -> FileResponse:
        target = _unlabeled_image_path(folder, image)
        if not target.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(target)

    def _assign_image_to_label(image_path: Path, label: str) -> Path:
        dest_dir = ensure_dir(state.paths.training / label)
        dest_path = dest_dir / image_path.name
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{image_path.stem}_{counter}{image_path.suffix}"
            counter += 1
        shutil.move(str(image_path), dest_path)
        return dest_path

    @fastapi_app.post("/api/unlabeled/{folder}/assign")
    async def assign_unlabeled(folder: str, payload: AssignRequest) -> JSONResponse:
        label = _validate_label(payload.label)
        src = _unlabeled_folder(folder)
        if not src.exists():
            raise HTTPException(status_code=404, detail="Folder not found")
        for image in sorted(src.glob("*")):
            if image.is_file():
                _assign_image_to_label(image, label)
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

    @fastapi_app.post("/api/unlabeled/{folder}/{image}/assign")
    async def assign_unlabeled_image(folder: str, image: str, payload: AssignRequest) -> JSONResponse:
        label = _validate_label(payload.label)
        image_path = _unlabeled_image_path(folder, image)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        _assign_image_to_label(image_path, label)
        _cleanup_unlabeled_folder(folder)
        return JSONResponse({"status": "assigned"})

    @fastapi_app.post("/api/unlabeled/{folder}/bulk-assign")
    async def bulk_assign_unlabeled(folder: str, payload: BulkAssignRequest) -> JSONResponse:
        label = _validate_label(payload.label)
        image_names = _unique_image_names(payload.images)
        assigned = 0
        for image_name in image_names:
            image_path = _unlabeled_image_path(folder, image_name)
            if not image_path.exists():
                raise HTTPException(status_code=404, detail=f"{image_name} not found")
            _assign_image_to_label(image_path, label)
            assigned += 1
        _cleanup_unlabeled_folder(folder)
        return JSONResponse({"status": "assigned", "count": assigned})

    @fastapi_app.post("/api/unlabeled/{folder}/bulk-delete")
    async def bulk_delete_unlabeled(folder: str, payload: BulkDeleteRequest) -> JSONResponse:
        image_names = _unique_image_names(payload.images)
        deleted = 0
        for image_name in image_names:
            image_path = _unlabeled_image_path(folder, image_name)
            if not image_path.exists():
                raise HTTPException(status_code=404, detail=f"{image_name} not found")
            image_path.unlink()
            deleted += 1
        _cleanup_unlabeled_folder(folder)
        return JSONResponse({"status": "deleted", "count": deleted})

    def _training_image_path(label: str, image: str) -> Path:
        label_dir = state.paths.training / label
        try:
            return safe_join(label_dir, image)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid image path")

    @fastapi_app.post("/api/unlabeled/{folder}/{image}/delete")
    async def delete_unlabeled_image(folder: str, image: str) -> JSONResponse:
        image_path = _unlabeled_image_path(folder, image)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        image_path.unlink()
        _cleanup_unlabeled_folder(folder)
        return JSONResponse({"status": "deleted"})

    def _clip_path(category: str, relative: str) -> Path:
        root = _category_root(category)
        try:
            return safe_join(root, relative)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid clip path")

    def _move_sidecar(src: Path, dest: Path) -> None:
        sidecar_src = src.with_suffix(src.suffix + ".json")
        if sidecar_src.exists():
            sidecar_dest = dest.with_suffix(dest.suffix + ".json")
            sidecar_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sidecar_src), sidecar_dest)

    @fastapi_app.post("/api/clips/assign")
    async def assign_clip(payload: ClipAssignRequest) -> JSONResponse:
        clip_path = _clip_path(payload.category, payload.clip)
        label = payload.label.strip()
        if not label:
            raise HTTPException(status_code=400, detail="Label is required")
        dest_dir = ensure_dir(state.paths.recognized / label)
        dest_path = dest_dir / clip_path.name
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{clip_path.stem}_{counter}{clip_path.suffix}"
            counter += 1
        shutil.move(str(clip_path), dest_path)
        _move_sidecar(clip_path, dest_path)
        return JSONResponse({"status": "assigned", "dest": dest_path.name})

    @fastapi_app.post("/api/clips/delete")
    async def delete_clip(payload: ClipDeleteRequest) -> JSONResponse:
        clip_path = _clip_path(payload.category, payload.clip)
        sidecar = clip_path.with_suffix(clip_path.suffix + ".json")
        clip_path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        return JSONResponse({"status": "deleted"})

    @fastapi_app.post("/api/training/delete")
    async def delete_training(payload: TrainingDeleteRequest) -> JSONResponse:
        image_path = _training_image_path(payload.label, payload.image)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        image_path.unlink()
        return JSONResponse({"status": "removed"})

    @fastapi_app.post("/api/training/{label}/bulk-delete")
    async def bulk_delete_training(label: str, payload: BulkDeleteRequest) -> JSONResponse:
        image_names = _unique_image_names(payload.images)
        deleted = 0
        for image_name in image_names:
            image_path = _training_image_path(label, image_name)
            if not image_path.exists():
                raise HTTPException(status_code=404, detail=f"{image_name} not found")
            image_path.unlink()
            deleted += 1
        return JSONResponse({"status": "deleted", "count": deleted})

    @fastapi_app.post("/api/training/assign")
    async def assign_training(payload: TrainingAssignRequest) -> JSONResponse:
        new_label = payload.new_label.strip()
        if not new_label:
            raise HTTPException(status_code=400, detail="Label is required")
        try:
            current_dir = safe_join(state.paths.training, payload.current_label)
            current_path = safe_join(Path(current_dir), payload.image)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid path")
        current_path = Path(current_path)
        if not current_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        dest_dir = ensure_dir(state.paths.training / new_label)
        dest_path = dest_dir / current_path.name
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{current_path.stem}_{counter}{current_path.suffix}"
            counter += 1
        shutil.move(str(current_path), dest_path)
        return JSONResponse({"status": "assigned"})

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
