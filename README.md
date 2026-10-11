# photogrammetry

Дипломный проект: детекция ArUco-меток и оценка их положения
на фотографиях и видео с помощью OpenCV.

## Возможности

- Детекция ArUco-меток на фото и видео (словари 4x4, 5x5, 6x6, 7x7)
- Оценка 3D-позы меток (solvePnP, SOLVEPNP_IPPE_SQUARE)
- Трекинг пропадающих меток с поиском в ROI и EMA-сглаживание
- Пересчёт координат в мировую систему относительно опорной метки
- Экспорт результатов: аннотированное видео, статистика (CSV), позы меток (CSV/JSON)
- Генерация ArUco-меток в PNG

## Стек

Python >= 3.11, OpenCV (opencv-contrib-python), NumPy, pandas, matplotlib, Jupyter (ipykernel).
Управление зависимостями: uv (pyproject.toml).

## Структура

- `src/base/aruco_detector.py` — базовый детектор, оценка поз, отрисовка
- `src/detectors/photo_detector.py` — обработка одиночного фото
- `src/detectors/video_detector.py` — покадровая обработка видео, трекинг, экспорт
- `src/utils/aruco_generator.py` — генерация меток
- `src/config.py` — пути к `data/` и `output/`
- `main.ipynb`, `aruco_generation.ipynb` — примеры запуска
