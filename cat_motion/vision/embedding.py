from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torchvision import models


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _ensure_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class EmbeddingExtractor:
    def __init__(self, input_size: int = 224, device: str | None = None) -> None:
        self.input_size = int(input_size)
        self.device = _ensure_device(device)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        base_model = models.mobilenet_v3_small(weights=weights)
        base_model.eval()
        self.feature_extractor = torch.nn.Sequential(
            base_model.features,
            base_model.avgpool,
            torch.nn.Flatten(),
        ).to(self.device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.mean = IMAGENET_MEAN.to(self.device)
        self.std = IMAGENET_STD.to(self.device)

    def extract(self, image: np.ndarray) -> np.ndarray:
        rgb = _to_rgb(image)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = (tensor - self.mean) / self.std
        tensor = tensor.to(self.device)
        with torch.no_grad():
            embedding = self.feature_extractor(tensor)
        embedding = torch.nn.functional.normalize(embedding, dim=1)
        return embedding.cpu().numpy()[0]


@dataclass
class EmbeddingModel:
    centroids: np.ndarray
    label_ids: np.ndarray

    @classmethod
    def load(cls, path: Path | str) -> "EmbeddingModel":
        data = np.load(Path(path))
        return cls(centroids=data["centroids"], label_ids=data["label_ids"])

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, centroids=self.centroids, label_ids=self.label_ids)


class EmbeddingRecognizer:
    def __init__(self, model: EmbeddingModel, extractor: EmbeddingExtractor, threshold: float = 0.75) -> None:
        self.model = model
        self.extractor = extractor
        self.threshold = float(threshold)

    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        embedding = self.extractor.extract(image)
        centroids = self.model.centroids
        sims = centroids @ embedding
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score < self.threshold:
            return -1, best_score
        return int(self.model.label_ids[best_idx]), best_score


def compute_centroids(embeddings: List[np.ndarray], labels: List[int]) -> EmbeddingModel:
    per_label: Dict[int, List[np.ndarray]] = {}
    for embedding, label in zip(embeddings, labels):
        per_label.setdefault(label, []).append(embedding)
    centroids: List[np.ndarray] = []
    label_ids: List[int] = []
    for label, vectors in per_label.items():
        stacked = np.stack(vectors)
        centroid = stacked.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        centroids.append(centroid)
        label_ids.append(label)
    return EmbeddingModel(centroids=np.stack(centroids), label_ids=np.array(label_ids, dtype=np.int64))
