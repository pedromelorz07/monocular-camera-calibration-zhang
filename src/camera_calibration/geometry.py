"""Rotinas de projeção e conversão de pose usando convenções do OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


def project_points(object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                   camera_matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    """Projeta pontos 3D com modelo pinhole e distorção padrão do OpenCV."""
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    return projected.reshape(-1, 2)


def rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Converte vetor Rodrigues em matriz de rotação 3×3."""
    return cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))[0]
