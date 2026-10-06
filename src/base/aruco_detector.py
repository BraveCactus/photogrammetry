from dataclasses import dataclass, field
from typing import cast

import cv2
import numpy as np

ARUCO_DICT_NAMES: dict[int, str] = {
    value: name.removeprefix("DICT_").split("_")[0]
    for name, value in vars(cv2.aruco).items()
    if name.startswith("DICT_") and isinstance(value, int)
}


@dataclass
class MarkerPose:
    id: int
    rvec: np.ndarray
    tvec: np.ndarray
    center_3d: np.ndarray

@dataclass
class DetectionResult:
    image: np.ndarray
    corners: tuple[np.ndarray, ...]
    ids: np.ndarray | None
    poses: list[MarkerPose] = field(default_factory=list)
    frame_index: int | None = None

    @property
    def found(self):
        """Обнаружение хотя бы одной метки.

        returns:
            найдено: True, если обнаружена хотя бы одна метка
        """
        return self.ids is not None and len(self.ids) > 0

class ArucoDetect:
    def __init__(
        self,
        marker_size: float = 0.2,
        dict_id: int = cv2.aruco.DICT_6X6_250,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> None:
        """Инициализация детектора ArUco.

        params
            marker_size: физический размер метки в метрах
            dict_id: идентификатор словаря ArUco
            camera_matrix: матрица камеры (3x3)
            dist_coeffs: коэффициенты дисторсии
        """
        self.marker_size = marker_size
        self.dict_id = dict_id
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.detector_params = self._create_detector_params()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = (dist_coeffs if dist_coeffs is not None 
                             else np.zeros((5, 1), dtype=np.float64))

    @property
    def marker_type(self) -> str:
        """Короткий вид словаря меток, например 4X4.

        returns:
            name: короткое имя типа меток
        """
        return ARUCO_DICT_NAMES.get(self.dict_id, str(self.dict_id))

    @staticmethod
    def _create_detector_params() -> cv2.aruco.DetectorParameters:
        """Создание параметров детектора ArUco.

        returns:
            params: настроенные параметры детектора
        """
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 3
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minMarkerDistanceRate = 0.05
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.2
        return params

    def get_camera_matrix(
            self, 
            width: int,
            height: int,
        ) -> tuple[np.ndarray, np.ndarray]:
        """Получение матрицы камеры и коэффициентов дисторсии.

        params
            width: ширина кадра
            height: высота кадра

        returns:
            camera_matrix: матрица камеры (3x3)
            dist_coeffs: коэффициенты дисторсии
        """
        if self.camera_matrix is None:
            focal_length = float(width)
            self.camera_matrix = np.array(
                [
                    [focal_length, 0.0, width / 2],
                    [0.0, focal_length, height / 2],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
        return self.camera_matrix, self.dist_coeffs

    def detect(
            self,
            image: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray | None]:
        """Нахождение маркеров на изображении.

        params
            image: входное изображение

        returns:
            corners: найденные углы маркеров (2D координаты пикселей на изображении)
            ids: идентификаторы найденных маркеров (None, если не найдены)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self.detector.detectMarkers(gray)
        detected = cast(tuple[np.ndarray, ...], corners)
        if ids is None:
            return detected, None
        return detected, cast(np.ndarray, ids).flatten()

    def _object_points(self) -> np.ndarray:
        """Получение 3D-координат углов метки в её системе координат.

        returns:
            points: массив точек углов (4x3)
        """
        half = self.marker_size / 2
        return np.array(
            [
                [-half, half, 0],
                [half, half, 0],
                [half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float32,
        )

    def estimate_pose(
        self,
        img_points: np.ndarray,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Оценивание позы метки через solvePnP.

        params
            img_points: 2D-координаты углов метки на изображении
            camera_matrix: матрица камеры
            dist_coeffs: коэффициенты дисторсии

        returns:
            ok: удалось ли оценить позу
            rvec: вектор вращения
            tvec: вектор переноса
            center_3d: центр метки в координатах камеры
        """
        matrix = camera_matrix if camera_matrix is not None else self.camera_matrix
        coeffs = dist_coeffs if dist_coeffs is not None else self.dist_coeffs

        if matrix is None:
            raise ValueError(
                "Матрица камеры не задана: вызовите get_camera_matrix(width, height)"
            )

        ok, rvec, tvec = cv2.solvePnP(
            self._object_points(),
            img_points.astype(np.float32),
            matrix,
            coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return False, None, None, None

        rotation, _ = cv2.Rodrigues(rvec)
        center_3d = (-rotation.T @ tvec).flatten()
        return True, rvec, tvec, center_3d

    def estimate_poses(
        self,
        corners: tuple[np.ndarray, ...],
        ids: np.ndarray | None,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
    ) -> list[MarkerPose]:
        """Оценивание позы всех найденных маркеров.

        params
            corners: углы маркеров
            ids: идентификаторы маркеров
            camera_matrix: матрица камеры
            dist_coeffs: коэффициенты дисторсии

        returns:
            poses: список оценённых поз MarkerPose
        """
        poses: list[MarkerPose] = []
        if ids is None:
            return poses

        for i, marker_id in enumerate(ids):
            ok, rvec, tvec, center_3d = self.estimate_pose(
                corners[i][0], camera_matrix, dist_coeffs
            )
            if not ok or rvec is None or tvec is None or center_3d is None:
                print(f"Маркер {marker_id}: расположение не вычислено")
                continue
            poses.append(MarkerPose(int(marker_id), rvec, tvec, center_3d))
        return poses

    def draw(
        self,
        image: np.ndarray,
        corners: tuple[np.ndarray, ...],
        ids: np.ndarray | None,
        poses: list[MarkerPose] | None = None,
        axis_length: float | None = None,
    ) -> np.ndarray:
        """Отрисовка найденных маркеров и осей координат на изображении.

        params
            image: исходное изображение
            corners: углы маркеров
            ids: идентификаторы маркеров
            poses: оценённые позы для отрисовки осей
            axis_length: длина осей координат

        returns:
            output: изображение с нарисованными маркерами
        """
        output = image.copy()
        if ids is None or len(ids) == 0:
            return output

        cv2.aruco.drawDetectedMarkers(output, corners, ids)

        if poses:
            matrix = self.camera_matrix
            if matrix is None:
                raise ValueError(
                    "Матрица камеры не задана: вызовите get_camera_matrix(width, height)"
                )
            length = self.marker_size if axis_length is None else axis_length
            for pose in poses:
                cv2.drawFrameAxes(
                    output, matrix, self.dist_coeffs,
                    pose.rvec, pose.tvec, length,
                )
        return output

    def close(self) -> None:
        """Закрытие всех окон OpenCV."""
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()