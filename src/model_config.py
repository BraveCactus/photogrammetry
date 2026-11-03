from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _matrix(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Преобразование JSON-массива в матрицу нужной формы."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(
            f"{label}: ожидалась форма {shape}, получено {array.shape}"
        )
    return array


def _dictionary_id(name: str | int) -> int:
    """Числовой идентификатор словаря ArUco по имени или числу."""
    if isinstance(name, int):
        return name

    full_name = name if name.startswith("DICT_") else f"DICT_{name}"
    value = getattr(cv2.aruco, full_name, None)
    if value is None:
        raise ValueError(f"Неизвестный словарь ArUco: {name}")
    return int(value)


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

        units = str(data.get("units", "m"))
        if units != "m":
            raise ValueError(f"{path}: поддерживаются только метры, units={units!r}")

        markers = data.get("markers", [])
        if not markers:
            raise ValueError(f"{path}: в конфиге нет маркеров")

        cameras_data = data.get("cameras", [])
        if not cameras_data:
            raise ValueError(f"{path}: в конфиге нет камер")

        dictionary_name = data.get("dictionary")
        if not isinstance(dictionary_name, str):
            raise TypeError(
                f"{path}: нет строкового поля 'dictionary'. "
                "Добавьте scene.dictionary = m.dictionary в scene_definition.m."
            )

        marker_size = float(markers[0]["size"])
        corners_local = _matrix(
            markers[0]["corners_local"], (4, 3), "corners_local"
        )
        for marker in markers[1:]:
            if not np.isclose(float(marker["size"]), marker_size):
                raise ValueError(
                    f"{path}: маркеры разного размера; поддерживается "
                    "только одинаковый размер"
                )
            if not np.allclose(
                np.asarray(marker["corners_local"], dtype=np.float64),
                corners_local,
            ):
                raise ValueError(
                    f"{path}: маркеры с разными corners_local "
                    "не поддерживаются"
                )

        cameras: dict[str, CameraConfig] = {}
        for item in cameras_data:
            name = str(item["name"])
            if name in cameras:
                raise ValueError(f"{path}: камера {name!r} описана дважды")

            resolution = item["resolution"]
            if len(resolution) != 2:
                raise ValueError(
                    f"{path}: {name}.resolution должен содержать [width, height]"
                )
            width, height = int(resolution[0]), int(resolution[1])

            dist = np.asarray(item["dist"], dtype=np.float64).reshape(-1)
            if dist.size == 0:
                dist = np.zeros(5, dtype=np.float64)

            cameras[name] = CameraConfig(
                name=name,
                width=width,
                height=height,
                K=_matrix(item["K"], (3, 3), f"{name}.K"),
                dist=dist,
                T_WC=_matrix(item["T_WC"], (4, 4), f"{name}.T_WC"),
            )

        render = data.get("render", {})
        fps = int(render.get("fps", 30))
        if fps <= 0:
            raise ValueError(f"{path}: fps должен быть положительным")

        return cls(
            path=path,
            schema=str(data.get("schema", "")),
            name=str(data.get("name", path.stem)),
            fps=fps,
            marker=MarkerConfig(
                dictionary=_dictionary_id(dictionary_name),
                dictionary_name=dictionary_name,
                size=marker_size,
                ids=tuple(int(marker["id"]) for marker in markers),
                corners_local=corners_local,
            ),
            cameras=cameras,
        )

    def camera(self, name: str) -> CameraConfig:
        """Камера по имени с понятной ошибкой, если имени нет."""
        try:
            return self.cameras[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.cameras))
            raise ValueError(
                f"Камера {name!r} не найдена. Доступны: {available}"
            ) from exc

    def ground_truth_dir(self, camera_name: str) -> Path:
        """Папка ground truth модели для камеры."""
        self.camera(camera_name)
        return self.path.parent / "ground_truth" / camera_name

    def detector_kwargs(self, camera_name: str) -> dict[str, object]:
        """Параметры для VideoArucoDetector."""
        camera = self.camera(camera_name)
        return {
            "marker_size": self.marker.size,
            "dict_id": self.marker.dictionary,
            "camera_matrix": camera.K,
            "dist_coeffs": camera.dist,
            "camera_name": camera.name,
            "world_from_camera": camera.T_WC,
        }

    def validate_video(self, camera_name: str, video_path: str | Path) -> None:
        """Проверка, что разрешение видео совпадает с камерой модели."""
        camera = self.camera(camera_name)
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
