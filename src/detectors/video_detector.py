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
    ) -> None:
        super().__init__(marker_size, dict_id, camera_matrix, dist_coeffs)
        self.axis_length = axis_length
        self.draw_distances = draw_distances
        self.smoothing = smoothing
        self._smooth_state: dict[int, np.ndarray] = {}

    def process(
        self,
        video_path: str | Path,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> Iterator[DetectionResult]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Не удалось открыть видео {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        camera_matrix, dist_coeffs = self.get_camera_matrix(width, height)

        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_index = 0

        try:
            self._smooth_state.clear()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                corners, ids = self.detect(frame)
                corners = self._smooth(corners, ids)
                poses = self.estimate_poses(corners, ids, camera_matrix, dist_coeffs)
                output = self.draw(
                    frame, corners, ids, poses, axis_length=self.axis_length
                )

                if self.draw_distances:
                    self._draw_distances(output, corners, ids, poses)

                if writer is not None:
                    writer.write(output)

                if ids is not None:
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

    @staticmethod
    def _draw_distances(
        image: np.ndarray,
        corners: tuple[np.ndarray, ...],
        ids: np.ndarray | None,
        poses: list[MarkerPose],
    ) -> None:
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
