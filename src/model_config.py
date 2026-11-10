from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MarkerConfig:
    """Параметры ArUco-маркеров модели."""

    dictionary: int
    dictionary_name: str
    size: float
    ids: tuple[int, ...]
    corners_local: np.ndarray


@dataclass(frozen=True)
class CameraConfig:
    """Параметры одной камеры модели."""

    name: str
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray
    T_WC: np.ndarray


@dataclass(frozen=True)
class ModelConfig:
    """Все параметры модели, необходимые photo-opu."""

    path: Path
    schema: str
    name: str
    fps: int
    marker: MarkerConfig
    cameras: dict[str, CameraConfig]

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        """Загрузка конфигурации из scene.json."""
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))

        markers = data["markers"]
        first_marker = markers[0]
        dictionary_name = data["dictionary"]

        cameras: dict[str, CameraConfig] = {}
        for item in data["cameras"]:
            name = str(item["name"])
            width, height = item["resolution"]
            cameras[name] = CameraConfig(
                name=name,
                width=int(width),
                height=int(height),
                K=np.asarray(item["K"], dtype=np.float64),
                dist=np.asarray(item["dist"], dtype=np.float64).reshape(-1),
                T_WC=np.asarray(item["T_WC"], dtype=np.float64),
            )

        return cls(
            path=path,
            schema=str(data.get("schema", "")),
            name=str(data.get("name", path.stem)),
            fps=int(data["render"]["fps"]),
            marker=MarkerConfig(
                dictionary=int(getattr(cv2.aruco, dictionary_name)),
                dictionary_name=dictionary_name,
                size=float(first_marker["size"]),
                ids=tuple(int(marker["id"]) for marker in markers),
                corners_local=np.asarray(
                    first_marker["corners_local"], dtype=np.float64
                ),
            ),
            cameras=cameras,
        )

    def camera(self, name: str) -> CameraConfig:
        """Камера по имени."""
        return self.cameras[name]

    def ground_truth_dir(self, camera_name: str) -> Path:
        """Папка ground truth модели для камеры."""
        return self.path.parent / "ground_truth" / camera_name

    def detector_kwargs(self, camera_name: str) -> dict[str, object]:
        """Параметры для VideoArucoDetector."""
        camera = self.cameras[camera_name]
        return {
            "marker_size": self.marker.size,
            "dict_id": self.marker.dictionary,
            "camera_matrix": camera.K,
            "dist_coeffs": camera.dist,
            "camera_name": camera.name,
            "world_from_camera": camera.T_WC,
            "scene_path": self.path,
        }

    def validate_video(self, camera_name: str, video_path: str | Path) -> None:
        """Проверка, что видео открывается и совпадает с камерой модели."""
        camera = self.cameras[camera_name]
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео {video_path}")

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()

        if (width, height) != (camera.width, camera.height):
            raise ValueError(
                f"Видео {width}x{height} не совпадает с камерой "
                f"{camera.name} {camera.width}x{camera.height}. "
                "Для уменьшенного видео масштабируйте K."
            )
