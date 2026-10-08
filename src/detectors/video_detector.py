import csv
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from src.base.aruco_detector import ArucoDetect, DetectionResult, MarkerPose

SMOOTHING = 0.9

class VideoArucoDetector(ArucoDetect):
    def __init__(
        self,
        marker_size: float = 0.2,
        dict_id: int = cv2.aruco.DICT_6X6_250,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        axis_length: float = 0.03,
        draw_distances: bool = True,
        smoothing: float = SMOOTHING,
        track_margin: float = 0.5,
        max_missing: int = 10,
    ) -> None:
        """Инициализация видеодетектора ArUco.

        params
            marker_size: физический размер метки в метрах
            dict_id: идентификатор словаря ArUco
            camera_matrix: матрица камеры (3x3)
            dist_coeffs: коэффициенты дисторсии
            axis_length: длина осей координат
            draw_distances: рисовать ли расстояния до меток
            smoothing: коэффициент EMA-сглаживания
            track_margin: запас области поиска при отслеживании
            max_missing: число кадров, после которых метка считается пропавшей
        """
        super().__init__(marker_size, dict_id, camera_matrix, dist_coeffs)

        self.axis_length = axis_length
        self.draw_distances = draw_distances
        # ema-сглаживание
        self.smoothing = smoothing
        self._smooth_state: dict[int, np.ndarray] = {}
        # roi-поиск
        self.track_margin = track_margin
        self.max_missing = max_missing
        self._track_state: dict[int, np.ndarray] = {}
        self._missing: dict[int, int] = {}

        soft_params = self._create_detector_params()
        soft_params.minMarkerPerimeterRate = 0.005
        soft_params.adaptiveThreshWinSizeMax = 23
        soft_params.adaptiveThreshWinSizeStep = 3

        self._soft_detector = cv2.aruco.ArucoDetector(self.aruco_dict, soft_params)

    def process(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Iterator[DetectionResult]:
        """Покадровая обработка видео и выдача результатов детекции.

        params
            video_path: путь к видео
            output_path: путь для сохранения результата
            show: показывать ли результат в окне OpenCV

        returns:
            результаты: генератор результатов DetectionResult по кадрам
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        camera_matrix, dist_coeffs = self.get_camera_matrix(width, height)

        stats_path: Path | None = None
        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            stats_path = output_path.with_suffix(".csv")

        frame_index = 0
        frames_with_markers = 0
        seen_ids: set[int] = set()
        total_frame_time = 0.0

        try:
            self._smooth_state.clear()
            self._track_state.clear()
            self._missing.clear()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1

                start = time.perf_counter()
                
                corners, ids = self.detect(frame)
                corners, ids = self._track(frame, corners, ids)
                corners = self._smooth(corners, ids)
                poses = self.estimate_poses(corners, ids, camera_matrix, dist_coeffs)
                output = self.draw(
                    frame, corners, ids, poses, axis_length=self.axis_length
                )

                if self.draw_distances:
                    self._draw_distances(output, corners, ids, poses)

                if writer is not None:
                    writer.write(output)

                total_frame_time += time.perf_counter() - start

                if ids is not None:
                    frames_with_markers += 1
                    seen_ids.update(int(marker_id) for marker_id in ids)
                    print(f"Кадр {frame_index}/{total_frames}: ID {ids.tolist()}")

                if show:
                    cv2.imshow("Aruco detection", output)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                yield DetectionResult(
                    image=output,
                    corners=corners,
                    ids=ids,
                    poses=poses,
                    frame_index=frame_index,
                )
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if stats_path is not None:
                avg_frame_ms = (
                    total_frame_time / frame_index * 1000 if frame_index else 0.0
                )
                self._save_stats(
                    stats_path,
                    Path(video_path).name,
                    frame_index,
                    frames_with_markers,
                    seen_ids,
                    avg_frame_ms,
                )

    def _save_stats(
        self,
        path: Path,
        video_name: str,
        frames_total: int,
        frames_with_markers: int,
        marker_ids: set[int],
        avg_frame_ms: float,
        
    ) -> None:
        """Сохранение статистики прогона видео в CSV.

        params
            path: путь к CSV-файлу
            video_name: имя исходного видео
            frames_total: число обработанных кадров
            frames_with_markers: число кадров с найденными метками
            marker_ids: множество найденных ID меток
            avg_frame_ms: количество милисекунд, которое тратит алгоритм на обработку одного кадра
        """
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(
                ["video", "marker type", "detected markers", "frames amount", "frames with markers", "average ms per frame"]
            )
            writer.writerow(
                [
                    video_name,
                    self.marker_type,
                    len(marker_ids),
                    frames_total,
                    frames_with_markers,
                    round(avg_frame_ms, 2)
                ]
            )

    @staticmethod
    def _draw_distances(
        image: np.ndarray,
        corners: tuple[np.ndarray, ...],
        ids: np.ndarray | None,
        poses: list[MarkerPose],
    ) -> None:
        """Отрисовка расстояний до каждой метки на изображении.

        params
            image: изображение для подписей
            corners: углы маркеров
            ids: идентификаторы маркеров
            poses: оценённые позы
        """
        if ids is None or not poses:
            return

        corners_by_id = {
            int(marker_id): corner for marker_id, corner in zip(ids, corners)
        }
        for pose in poses:
            corner = corners_by_id.get(pose.id)
            if corner is None:
                continue
            distance = float(np.linalg.norm(pose.tvec))
            cx, cy = corner[0].mean(axis=0).astype(int)
            cv2.putText(
                image,
                f"{distance:.2f} m",
                (cx - 40, cy + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _smooth(
            self,
            corners: tuple[np.ndarray, ...],
            ids: np.ndarray | None,
    ) -> tuple[np.ndarray, ...]:
        """Сглаживание углов маркеров по предыдущим кадрам (EMA).

        params
            corners: углы маркеров
            ids: идентификаторы маркеров

        returns:
            result: сглаженные углы маркеров
        """
        if ids is None or self.smoothing >= 1.0:
            return corners
        result: list[np.ndarray] = []
        for corner, marker_id in zip(corners, ids):
            key = int(marker_id)
            pts = corner[0].astype(np.float32)
            previous = self._smooth_state.get(key)
            if previous is not None:
                pts = self.smoothing * pts + (1.0 - self.smoothing) * previous
            self._smooth_state[key] = pts
            result.append(pts.reshape(1, 4, 2))
        return tuple(result)

    def _track(
            self,
            image: np.ndarray,
            corners: tuple[np.ndarray, ...],
            ids: np.ndarray | None,
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray | None]:
        """Отслеживание пропавших маркеров по области последнего кадра.

        params
            image: текущий кадр
            corners: углы маркеров
            ids: идентификаторы маркеров

        returns:
            tracked_corners: обновлённые углы маркеров
            tracked_ids: обновлённые идентификаторы маркеров
        """
        current: dict[int, np.ndarray] = {}
        if ids is not None:
            for corner, marker_id in zip(corners, ids):
                current[int(marker_id)] = corner[0].astype(np.float32)

        for marker_id in list(self._track_state):
            if marker_id in current:
                self._missing[marker_id] = 0
                continue

            self._missing[marker_id] = self._missing.get(marker_id, 0) + 1
            if self._missing[marker_id] > self.max_missing:
                del self._track_state[marker_id]
                del self._missing[marker_id]
                continue

            recovered = self._detect_in_roi(
                image, marker_id, self._track_state[marker_id]
            )
            if recovered is not None:
                current[marker_id] = recovered
                self._missing[marker_id] = 0

        self._track_state.update(current)
        if not current:
            return (), None

        items = sorted(current.items())
        tracked_corners = tuple(
            pts.reshape(1, 4, 2).astype(np.float32) for _, pts in items
        )
        tracked_ids = np.array([marker_id for marker_id, _ in items], dtype=np.int32)
        return tracked_corners, tracked_ids

    def _detect_in_roi(
        self,
        image: np.ndarray,
        marker_id: int,
        last_corners: np.ndarray,
    ) -> np.ndarray | None:
        """Поиск маркера в ограниченной области вокруг последнего положения.

        params
            image: текущий кадр
            marker_id: идентификатор искомого маркера
            last_corners: последние известные углы маркера

        returns:
            corners: найденные углы маркера (None, если не найден)
        """
        x, y, w, h = cv2.boundingRect(last_corners)
        mx = int(w * self.track_margin)
        my = int(h * self.track_margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1 = min(image.shape[1], x + w + mx)
        y1 = min(image.shape[0], y + h + my)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None

        roi = image[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        roi_corners, roi_ids, _ = self._soft_detector.detectMarkers(gray)
        if roi_ids is None:
            return None

        offset = np.array([x0, y0], dtype=np.float32)
        for corner, rid in zip(roi_corners, roi_ids):
            if int(rid) == marker_id:
                return corner[0].astype(np.float32) + offset
        return None




