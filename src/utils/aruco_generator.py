from pathlib import Path

import cv2

from src.config import settings

ARUCO_DICTS: dict[str, int] = {
    "4x4": cv2.aruco.DICT_4X4_50,
    "5x5": cv2.aruco.DICT_5X5_50,
    "6x6": cv2.aruco.DICT_6X6_250,
    "7x7": cv2.aruco.DICT_7X7_250,
}


def generate_aruco_markers(
    count: int = 6,
    marker_type: str = "6x6",
    image_size: int = 512,
    output_dir: Path | str | None = None,
    start_id: int = 0,
) -> list[Path]:
    """Генерирует ArUco-метки заданного типа и сохраняет их в PNG."""    
    
    if marker_type not in ARUCO_DICTS:
        available = ", ".join(ARUCO_DICTS)
        raise ValueError(f"Неизвестный тип метки: {marker_type}. Доступны: {available}")

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[marker_type])
    capacity = int(dictionary.bytesList.shape[0])
    if start_id + count > capacity:
        raise ValueError(
            f"Тип {marker_type} содержит {capacity} меток, "
            f"запрошено {count} (начиная с {start_id})"
        )

    target_dir = Path(output_dir) if output_dir is not None else settings.aruco_markers
    target_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for marker_id in range(start_id, start_id + count):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, image_size)
        path = target_dir / f"aruco_{marker_id}.png"
        cv2.imwrite(str(path), marker)
        saved.append(path)
    return saved
