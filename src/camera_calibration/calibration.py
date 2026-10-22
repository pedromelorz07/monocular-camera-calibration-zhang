"""Detecção, calibração Zhang/OpenCV e avaliação das observações."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .board import chessboard_object_points, detect_chessboard
from .geometry import project_points
from .metrics import reprojection_metrics


def calibrate_directory(image_dir: Path, inner_corners: tuple[int, int], square_size: float,
                        minimum_views: int = 5, window: tuple[int, int] = (11, 11)) -> dict:
    """Detecta cantos em imagens e calibra com cv2.calibrateCamera.

    Imagens que não contenham o padrão são listadas e ignoradas; tamanho inconsistente é erro.
    """
    paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in
                   {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise ValueError(f"Nenhuma imagem compatível encontrada em {image_dir}")
    object_template = chessboard_object_points(inner_corners, square_size)
    object_points, image_points, used, rejected = [], [], [], []
    image_size = None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"file": path.name, "reason": "não foi possível carregar"})
            continue
        size = (image.shape[1], image.shape[0])
        if image_size is not None and size != image_size:
            raise ValueError(f"Resolução inconsistente em {path.name}: {size} != {image_size}")
        image_size = size
        corners = detect_chessboard(image, inner_corners, window)
        if corners is None:
            rejected.append({"file": path.name, "reason": "tabuleiro não detectado"})
            continue
        object_points.append(object_template.copy())
        image_points.append(corners)
        used.append(path)
    if len(used) < minimum_views:
        raise ValueError(f"Somente {len(used)} vistas válidas; mínimo exigido: {minimum_views}. Rejeitadas: {rejected}")
    rms_opencv, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None)
    projected = [project_points(obj, rv, tv, matrix, distortion)
                 for obj, rv, tv in zip(object_points, rvecs, tvecs)]
    observed = [p.reshape(-1, 2) for p in image_points]
    metrics = reprojection_metrics(observed, projected)
    poses = [{"file": p.name, "rvec": rv.reshape(-1).tolist(), "tvec": tv.reshape(-1).tolist()}
             for p, rv, tv in zip(used, rvecs, tvecs)]
    return {"camera_matrix": matrix, "distortion_coefficients": distortion.reshape(-1),
            "rvecs": rvecs, "tvecs": tvecs, "image_size": image_size, "used_files": used,
            "rejected_files": rejected, "object_points": object_points, "image_points": image_points,
            "projected_points": projected, "poses": poses, "opencv_rms": float(rms_opencv),
            "metrics": metrics}
