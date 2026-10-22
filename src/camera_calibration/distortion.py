"""Correção de distorção para o modelo padrão de cinco parâmetros do OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


def undistort_image(image: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray,
                    alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Corrige distorção; devolve imagem e nova matriz intrínseca para o tamanho original."""
    height, width = image.shape[:2]
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion, (width, height), alpha,
                                                     (width, height))
    corrected = cv2.undistort(image, camera_matrix, distortion, None, new_matrix)
    return corrected, new_matrix
