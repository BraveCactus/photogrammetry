from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MATRIX_COLUMNS = [f"m{i}{j}" for i in range(3) for j in range(3)]


def angles_from_rotation(matrix: np.ndarray) -> tuple[float, float]:
    """Азимут и угол места из матрицы поворота ротора.

    params:
        matrix: матрица R_BR (ротор -> база/статор)

    returns:
        phi: азимут, рад
        theta: угол места, рад
    """
    phi = math.atan2(-matrix[0, 1], matrix[1, 1])
    theta = math.atan2(matrix[2, 0], matrix[2, 2])
    return phi, theta


def load_marker_corners(path: Path) -> dict[int, np.ndarray]:
    """Углы маркеров в системе координат ротора.

    params:
        path: путь к scene.json

    returns:
        corners: словарь marker_id -> массив 4x3 в СК ротора
    """
    scene = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, np.ndarray] = {}
    for marker in scene["markers"]:
        transform = np.asarray(marker["T_BM"], dtype=float)
        corners = np.asarray(marker["corners_local"], dtype=float)
        corners_h = np.hstack([corners, np.ones((4, 1))])
        result[int(marker["id"])] = (transform @ corners_h.T).T[:, :3]
    return result


def fit_rigid(
    local: np.ndarray,
    world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Реализация алгоритма Кабша.

    params:
        local: точки в СК ротора, Nx3
        world: те же точки в мировой СК, Nx3

    returns:
        rotation: матрица поворота R_WR
        translation: вектор переноса t_WR
        rms: среднеквадратичная невязка, м
    """
    local_center = local.mean(axis=0)
    world_center = world.mean(axis=0)
    covariance = (local - local_center).T @ (world - world_center)
    u, _, vt = np.linalg.svd(covariance)
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    translation = world_center - rotation @ local_center
    residual = local @ rotation.T + translation - world
    rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return rotation, translation, rms


def angles_from_animation(path: Path) -> pd.DataFrame:
    """Истинные углы ротора из ground_truth/animation.csv.

    params:
        path: путь к ground_truth/animation.csv

    returns:
        angles: таблица с frame, t, phi и theta
    """
    data = pd.read_csv(path)
    rotor = data[data["body"] == "Rotor"]
    rows = []
    for _, row in rotor.iterrows():
        matrix = row[MATRIX_COLUMNS].to_numpy(dtype=float).reshape(3, 3)
        phi, theta = angles_from_rotation(matrix)
        rows.append(
            {
                "frame": int(row["frame"]),
                "t": float(row["t"]),
                "phi_rad": phi,
                "theta_rad": theta,
                "phi_deg": math.degrees(phi),
                "theta_deg": math.degrees(theta),
            }
        )
    return pd.DataFrame(rows)


def angles_from_world_corners(
    path: Path,
    marker_corners: dict[int, np.ndarray],
) -> pd.DataFrame:
    """Углы ротора по измеренным 3D-углам маркеров.

    params:
        path: путь к world_corners.csv
        marker_corners: локальные углы маркеров в СК ротора

    returns:
        angles: таблица с frame, t, camera, phi, theta, n_points и rms_m
    """
    data = pd.read_csv(path)
    rows = []
    for frame, group in data.groupby("frame", sort=True):
        local = []
        world = []
        for marker_id, marker_rows in group.groupby("marker_id"):
            marker_id = int(marker_id)
            if marker_id not in marker_corners:
                continue
            row = marker_rows.iloc[0]
            corners_world = np.array(
                [
                    [row[f"x{i}"], row[f"y{i}"], row[f"z{i}"]]
                    for i in range(4)
                ],
                dtype=float,
            )
            local.append(marker_corners[marker_id])
            world.append(corners_world)

        local_array = np.vstack(local)
        world_array = np.vstack(world)
        rotation, _, rms = fit_rigid(local_array, world_array)
        phi, theta = angles_from_rotation(rotation)

        first = group.iloc[0]
        rows.append(
            {
                "frame": int(frame),
                "t": float(first["t"]),
                "camera": str(first["camera"]),
                "phi_rad": phi,
                "theta_rad": theta,
                "phi_deg": math.degrees(phi),
                "theta_deg": math.degrees(theta),
                "n_points": len(local_array),
                "rms_m": rms,
            }
        )
    return pd.DataFrame(rows)
