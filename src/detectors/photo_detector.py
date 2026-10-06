from pathlib import Path

import cv2

from src.base.aruco_detector import ArucoDetect, DetectionResult


class PhotoArucoDetector(ArucoDetect):
    def process(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
        show: bool = False,
        save: bool = True,
        axis_length: float | None = None,
    ) -> DetectionResult: 
        """Обработка одиночного фото: детекция, позы, отрисовка и сохранение.

        params
            image_path: путь к изображению
            output_path: путь для сохранения результата
            show: показывать ли результат в окне OpenCV
            save: сохранять ли результат
            axis_length: длина осей координат

        returns:
            result: результат детекции DetectionResult
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Не удалось открыть {image_path}")

        height, width = image.shape[:2]
        camera_matrix, dist_coeffs = self.get_camera_matrix(width, height)

        corners, ids = self.detect(image)
        poses = self.estimate_poses(corners, ids, camera_matrix, dist_coeffs)
        output = self.draw(image, corners, ids, poses, axis_length=axis_length)

        if ids is None:
            print("Маркеры не найдены.")
        else:
            print(f"Найдено маркеров: {len(ids)}")
            print(f"ID: {ids.tolist()}")
            for pose in poses:
                print(
                    f"Маркер ID={pose.id}: "
                    f"tvec={pose.tvec.flatten().round(3)}, "
                    f"центр в координатах камеры={pose.center_3d.round(3)}"
                )
        if save and output_path is not None:
            cv2.imwrite(str(output_path), output)

        if show:
            cv2.imshow("ArUco detection", output)
            cv2.waitKey(0)

        return DetectionResult(image=output, corners=corners, ids=ids, poses=poses)

