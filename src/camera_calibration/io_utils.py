"""Leitura e escrita de relatórios e parâmetros de calibração."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def write_calibration_yaml(path: Path, camera_matrix: np.ndarray, distortion: np.ndarray,
                           image_size: tuple[int, int], square_unit: str = "m") -> None:
    """Salva matriz, distorção e metadados em YAML legível pelo OpenCV/ROS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise OSError(f"Não foi possível escrever YAML: {path}")
    fs.write("camera_matrix", np.asarray(camera_matrix, dtype=np.float64))
    fs.write("distortion_coefficients", np.asarray(distortion, dtype=np.float64).reshape(1, -1))
    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("distortion_model", "plumb_bob")
    fs.write("square_unit", square_unit)
    fs.release()


def read_calibration_yaml(path: Path) -> dict:
    """Lê e valida os campos essenciais do YAML de calibração."""
    if not path.is_file():
        raise FileNotFoundError(path)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"YAML inválido ou ilegível: {path}")
    matrix = fs.getNode("camera_matrix").mat()
    distortion = fs.getNode("distortion_coefficients").mat()
    width = int(fs.getNode("image_width").real())
    height = int(fs.getNode("image_height").real())
    model = fs.getNode("distortion_model").string()
    unit = fs.getNode("square_unit").string()
    fs.release()
    if matrix is None or matrix.shape != (3, 3) or distortion is None or distortion.size < 4:
        raise ValueError("YAML não contém matriz 3×3 e coeficientes de distorção válidos.")
    return {"camera_matrix": matrix, "distortion_coefficients": distortion.reshape(-1),
            "image_size": (width, height), "distortion_model": model, "square_unit": unit}


def write_json(path: Path, data: dict) -> None:
    """Escreve JSON com criação automática dos diretórios."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
