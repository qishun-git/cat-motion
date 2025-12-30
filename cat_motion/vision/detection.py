from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

Box = Tuple[int, int, int, int]


class BaseDetector:
    def detect(self, frame: np.ndarray) -> List[Box]:
        raise NotImplementedError


class YoloOnnxDetector(BaseDetector):
    def __init__(
        self,
        model_path: Path,
        input_size: Tuple[int, int],
        class_ids: Optional[Sequence[int]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is required for YOLO detection.")
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.class_ids = set(int(cid) for cid in class_ids) if class_ids else None
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        sess_opts = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(model_path),
            sess_opts,
            providers=list(providers) if providers else None,
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = getattr(input_meta, "shape", None)
        if shape and len(shape) == 4:
            _, _, h_dim, w_dim = shape
            if isinstance(h_dim, (int, np.integer)) and isinstance(w_dim, (int, np.integer)):
                if h_dim > 0 and w_dim > 0:
                    self.input_size = (int(h_dim), int(w_dim))

    def detect(self, frame: np.ndarray) -> List[Box]:
        processed, ratio, pad = self._prepare_input(frame)
        preds = self.session.run(None, {self.input_name: processed})[0]
        if preds.ndim == 3:
            preds = preds[0]
        boxes: List[Box] = []
        scores: List[float] = []
        for det in preds:
            obj_conf = det[4]
            if obj_conf < 1e-5:
                continue
            cls_scores = det[5:]
            cls_id = int(np.argmax(cls_scores))
            if self.class_ids and cls_id not in self.class_ids:
                continue
            score = float(obj_conf * cls_scores[cls_id])
            if score < self.conf_threshold:
                continue
            box = self._xywh_to_xywh(det[:4], ratio, pad, frame.shape[1], frame.shape[0])
            boxes.append(box)
            scores.append(score)
        keep = self._nms(boxes, scores, self.iou_threshold)
        return [boxes[i] for i in keep]

    def _prepare_input(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        h, w = frame.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        dw = (target_w - new_w) / 2
        dh = (target_h - new_h) / 2
        top = int(math.floor(dh))
        left = int(math.floor(dw))
        padded[top : top + new_h, left : left + new_w] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor, scale, (dw, dh)

    def _xywh_to_xywh(
        self,
        box: np.ndarray,
        scale: float,
        pad: Tuple[float, float],
        image_w: int,
        image_h: int,
    ) -> Box:
        x_c, y_c, width, height = box
        pad_w, pad_h = pad
        x1 = (x_c - width / 2 - pad_w) / scale
        y1 = (y_c - height / 2 - pad_h) / scale
        x2 = (x_c + width / 2 - pad_w) / scale
        y2 = (y_c + height / 2 - pad_h) / scale
        x1 = max(0, min(image_w - 1, x1))
        y1 = max(0, min(image_h - 1, y1))
        x2 = max(0, min(image_w - 1, x2))
        y2 = max(0, min(image_h - 1, y2))
        return int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))

    def _nms(self, boxes: List[Box], scores: List[float], threshold: float) -> List[int]:
        if not boxes:
            return []
        boxes_arr = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=np.float32)
        scores_arr = np.array(scores, dtype=np.float32)
        order = scores_arr.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = self._iou(boxes_arr[i], boxes_arr[rest])
            rest = rest[ious <= threshold]
            order = rest
        return keep

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter + 1e-6
        return inter / union


def resolve_input_size(value: tuple[int, int] | int) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return (int(value[0]), int(value[1]))
    size = int(value)
    return size, size


def create_detector(
    *,
    model_path: Path,
    input_size: tuple[int, int] | int,
    class_ids: Optional[Sequence[int]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    providers: Optional[Sequence[str]] = None,
) -> BaseDetector:
    size = resolve_input_size(input_size)
    return YoloOnnxDetector(
        model_path=model_path,
        input_size=size,
        class_ids=class_ids,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        providers=providers,
    )
