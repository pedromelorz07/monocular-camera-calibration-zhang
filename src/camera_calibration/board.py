"""Geometria e detecção de tabuleiros de xadrez."""

from __future__ import annotations

import cv2
import numpy as np


def chessboard_object_points(inner_corners: tuple[int, int], square_size: float) -> np.ndarray:
    """Cria N×1 pontos 3D (x,y,0), em ordem por linha, para cantos internos."""
    columns, rows = inner_corners
    if columns < 2 or rows < 2 or square_size <= 0:
        raise ValueError("O tabuleiro precisa de ao menos 2×2 cantos e tamanho positivo.")
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    return points


def detect_chessboard(image: np.ndarray, inner_corners: tuple[int, int],
                      window: tuple[int, int] = (11, 11)) -> np.ndarray | None:
    """Detecta cantos internos e os refina em escala de cinza; retorna N×1×2."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if gray.ndim != 2:
        raise ValueError("A imagem deve ser em tons de cinza ou BGR.")
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, inner_corners, flags)
    if not found or corners is None:
        return None
    return cv2.cornerSubPix(gray, corners, window, (-1, -1),
                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
