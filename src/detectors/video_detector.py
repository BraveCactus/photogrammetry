import csv
import json
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from src.base.aruco_detector import ArucoDetect, DetectionResult, MarkerPose

# Пояснение:  Blender-камера имеет ориентацию (X вправо, Y вверх, -Z вперёд)
# а OpenCV-камера (X вправо, Y вниз, +Z вперёд)
MODEL_CAM_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])

POSE_COLUMNS = [
    "frame_index", "time_s", "marker_id", "is_reference",
    "x_cam", "y_cam", "z_cam", "dist_cam",
    "rvec_cam_x", "rvec_cam_y", "rvec_cam_z",
    "x_world", "y_world", "z_world",
    "rvec_world_x", "rvec_world_y", "rvec_world_z",
]

MEASUREMENT_PROJECTION_COLUMNS = [
    "frame", "t", "camera", "marker_id", "visible",
    "u0", "v0", "u1", "v1", "u2", "v2", "u3", "v3",
]

MEASUREMENT_CORNER_COLUMNS = [
    "frame", "t", "camera", "marker_id", "visible",
    *[f"{axis}{i}" for i in range(4) for axis in ("x", "y", "z")],
]

class VideoArucoDetector(ArucoDetect):
    """Покадровая детекция ArUco-маркеров на видео."""    

    def __init__(
        self,
        marker_size: float = 0.2,
        dict_id: int = cv2.aruco.DICT_6X6_250,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        axis_length: float = 0.03,
        draw_distances: bool = True,
        reference_marker_id: int | None = None,   # если None, то мир = камера
        camera_name: str | None = None,
        world_from_camera: np.ndarray | None = None,
        save_measurements: bool = False,
    ) -> None:
        """Инициализация видеодетектора ArUco.

        params:
            marker_size: физический размер метки в метрах
            dict_id: идентификатор словаря ArUco
            camera_matrix: матрица камеры (3x3)
            dist_coeffs: коэффициенты дисторсии
            axis_length: длина осей координат
            draw_distances: рисовать ли расстояния до меток
            reference_marker_id: ID опорного маркера для мировой СК
            camera_name: имя камеры для экспорта измерений
            world_from_camera: матрица T_WC (камера -> мир)
            save_measurements: сохранять углы маркеров в формате model
        """
        super().__init__(marker_size, dict_id, camera_matrix, dist_coeffs)

        self.axis_length = axis_length
        self.draw_distances = draw_distances
        self.reference_marker_id = reference_marker_id
        self.camera_name = camera_name
        self.world_from_camera = (
            None
            if world_from_camera is None
            else np.asarray(world_from_camera, dtype=np.float64)
        )

        self.save_measurements = save_measurements

    def process(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Iterator[DetectionResult]:
        """Покадровая обработка видео и выдача результатов детекции.

        params:
            video_path: путь к видео
            output_path: путь для сохранения результата
            show: показывать ли результат в окне OpenCV

        returns:
            результаты: генератор результатов DetectionResult по кадрам
        """
        if self.save_measurements and output_path is None:
            raise ValueError("save_measurements=True требует output_path")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        camera_matrix, dist_coeffs = self.get_camera_matrix(width, height)

        stats_path: Path | None = None
        pose_csv_path: Path | None = None
        pose_json_path: Path | None = None
        measurement_dir: Path | None = None

        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            stats_path = output_path.with_suffix(".csv")
            pose_csv_path = output_path.with_name(f"{output_path.stem}_poses.csv")
            pose_json_path = output_path.with_name(f"{output_path.stem}_poses.json")
            if self.save_measurements:
                measurement_name = self.camera_name or Path(video_path).stem
                measurement_dir = output_path.parent / "measurements" / measurement_name

        frame_index = 0
        frames_with_markers = 0
        seen_ids: set[int] = set()
        total_frame_time = 0.0

        pose_rows: list[dict] = []
        measurement_rows: list[
            tuple[
                int, float, int, np.ndarray, np.ndarray, np.ndarray | None
            ]
        ] = []
        write_world = (
            self.world_from_camera is not None
            or self.reference_marker_id is not None
        )
        world_transform: np.ndarray | None
        if self.world_from_camera is not None:
            # T_WC задана в СК Blender, поэтому переводим в OpenCV-камеру.
            world_transform = self.world_from_camera @ MODEL_CAM_TO_CV
        elif self.reference_marker_id is None:
            world_transform = np.eye(4)
        else:
            world_transform = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1

                start = time.perf_counter()
                
                corners, ids = self.detect(frame)
                poses = self.estimate_poses(corners, ids, camera_matrix, dist_coeffs)

                if world_transform is None:
                    reference = next(
                        (p for p in poses if p.id == self.reference_marker_id), None
                    )
                    if reference is not None:
                        world_transform = np.linalg.inv(
                            self.pose_matrix(reference.rvec, reference.tvec)
                        )
                time_s = (frame_index - 1) / fps if fps else 0.0
                pose_rows.extend(
                    self._pose_row(frame_index, time_s, pose, world_transform)
                    for pose in poses
                )

                if self.save_measurements and ids is not None:
                    corners_by_id = {
                        int(marker_id): corner[0]
                        for corner, marker_id in zip(corners, ids)
                    }
                    object_points = np.hstack(
                        [self._object_points(), np.ones((4, 1))]
                    )
                    for pose in poses:
                        uv = corners_by_id.get(pose.id)
                        if uv is None:
                            continue
                        T_cv_m = self.pose_matrix(pose.rvec, pose.tvec)
                        x_c = (
                            MODEL_CAM_TO_CV @ T_cv_m @ object_points.T
                        )[:3].T
                        x_w = None
                        if write_world and world_transform is not None:
                            x_w = (
                                world_transform @ T_cv_m @ object_points.T
                            )[:3].T
                        measurement_rows.append(
                            (
                                frame_index - 1,
                                time_s,
                                pose.id,
                                uv,
                                x_c,
                                x_w,
                            )
                        )

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
            if (
                stats_path is not None
                and pose_csv_path is not None
                and pose_json_path is not None
            ):
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
                self._save_poses(
                    pose_rows,
                    pose_csv_path,
                    pose_json_path,
                    Path(video_path).name,
                    self.reference_marker_id,
                )

            if self.save_measurements and measurement_dir is not None:
                measurement_name = self.camera_name or Path(video_path).stem
                self._save_measurements(
                    measurement_rows, measurement_dir, measurement_name, write_world
                )
                print(f"Measurements: {measurement_dir}")

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

        params:
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

    def _pose_row(
        self,
        frame_index: int,
        time_s: float,
        pose: MarkerPose,
        world_transform: np.ndarray | None,
    ) -> dict:
        """Формирование строки с положением одного маркера.

        params:
            frame_index: номер кадра
            time_s: время кадра, с
            pose: оценённое положение маркера
            world_transform: переход камера -> мир или None

        returns:
            row: словарь с полями для CSV
        """
        rvec, tvec = pose.rvec.flatten(), pose.tvec.flatten()
        row: dict = {
            "frame_index": frame_index,
            "time_s": round(time_s, 3),
            "marker_id": pose.id,
            "is_reference": pose.id == self.reference_marker_id,
            "x_cam": round(float(tvec[0]), 4),
            "y_cam": round(float(tvec[1]), 4),
            "z_cam": round(float(tvec[2]), 4),
            "dist_cam": round(float(np.linalg.norm(tvec)), 4),
            "rvec_cam_x": round(float(rvec[0]), 4),
            "rvec_cam_y": round(float(rvec[1]), 4),
            "rvec_cam_z": round(float(rvec[2]), 4),
            "x_world": None, "y_world": None, "z_world": None,
            "rvec_world_x": None, "rvec_world_y": None, "rvec_world_z": None,
        }
        if world_transform is not None:
            rvec_world, tvec_world = self.transform_pose(pose.rvec, pose.tvec, world_transform)
            row.update(
                x_world=round(float(tvec_world[0]), 4),
                y_world=round(float(tvec_world[1]), 4),
                z_world=round(float(tvec_world[2]), 4),
                rvec_world_x=round(float(rvec_world[0]), 4),
                rvec_world_y=round(float(rvec_world[1]), 4),
                rvec_world_z=round(float(rvec_world[2]), 4),
            )
        return row

    @staticmethod
    def _save_poses(
        rows: list[dict],
        csv_path: Path,
        json_path: Path,
        video_name: str,
        reference_marker_id: int | None,
    ) -> None:
        """Сохранение положения маркеров в CSV и JSON.

        params:
            rows: строки с положениями маркеров
            csv_path: путь к CSV-файлу
            json_path: путь к JSON-файлу
            video_name: имя исходного видео
            reference_marker_id: ID опорного маркера или None
        """
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(POSE_COLUMNS)
            for row in rows:
                writer.writerow(
                    "" if row[c] is None else row[c] for c in POSE_COLUMNS
                )

        frames: dict[int, dict] = {}
        for row in rows:
            frame = frames.setdefault(
                row["frame_index"],
                {"frame_index": row["frame_index"], "time_s": row["time_s"], "markers": []},
            )
            frame["markers"].append(
                {k: v for k, v in row.items() if k not in ("frame_index", "time_s")}
            )
        payload = {
            "video": video_name,
            "reference_marker_id": reference_marker_id,
            "frames": list(frames.values()),
        }
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @staticmethod
    def _save_measurements(
        rows: list[
            tuple[
                int, float, int, np.ndarray, np.ndarray, np.ndarray | None
            ]
        ],
        measurement_dir: Path,
        camera_name: str,
        write_world: bool,
    ) -> None:
        """Сохранение углов маркеров в формате model.

        params:
            rows: (frame, t, marker_id, 2D-углы 4x2,
                  3D-углы 4x3 в СК камеры,
                  3D-углы 4x3 в мировой СК или None)
            measurement_dir: папка measurements/<camera>
            camera_name: имя камеры для колонки camera
            write_world: писать ли world_corners.csv
        """
        measurement_dir.mkdir(parents=True, exist_ok=True)

        with (measurement_dir / "projections.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(MEASUREMENT_PROJECTION_COLUMNS)
            for frame, t, marker_id, uv, _, _ in rows:
                writer.writerow(
                    [
                        frame,
                        f"{t:.9f}",
                        camera_name,
                        marker_id,
                        1,
                        *[f"{value:.6f}" for value in uv.reshape(-1)],
                    ]
                )

        with (measurement_dir / "camera_corners.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(MEASUREMENT_CORNER_COLUMNS)
            for frame, t, marker_id, _, x_c, _ in rows:
                writer.writerow(
                    [
                        frame,
                        f"{t:.9f}",
                        camera_name,
                        marker_id,
                        1,
                        *[f"{value:.9f}" for value in x_c.reshape(-1)],
                    ]
                )

        if not write_world:
            print(
                "world_corners.csv не создан: не задана T_WC "
                "или reference_marker_id"
            )
            return

        with (measurement_dir / "world_corners.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(MEASUREMENT_CORNER_COLUMNS)
            written = 0
            for frame, t, marker_id, _, _, x_w in rows:
                if x_w is None:
                    continue
                writer.writerow(
                    [
                        frame,
                        f"{t:.9f}",
                        camera_name,
                        marker_id,
                        1,
                        *[f"{value:.9f}" for value in x_w.reshape(-1)],
                    ]
                )
                written += 1
            if written == 0:
                print(
                    "world_corners.csv: нет данных, "
                    "мировая СК не определена"
                )

    @staticmethod
    def _draw_distances(
        image: np.ndarray,
        corners: tuple[np.ndarray, ...],
        ids: np.ndarray | None,
        poses: list[MarkerPose],
    ) -> None:
        """Отрисовка расстояний до каждой метки на изображении.

        params:
            image: изображение для подписей
            corners: углы маркеров
            ids: идентификаторы маркеров
            poses: оценённые положения
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
