"""Gera um conjunto controlado de imagens de xadrez com modelo pinhole distorcido."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from camera_calibration.board import chessboard_object_points
from camera_calibration.io_utils import write_json


def render_view(inner: tuple[int, int], square_size: float, square_pixels: int,
                image_size: tuple[int, int], matrix: np.ndarray, distortion: np.ndarray,
                rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renderiza textura plana via homografia e remapeia pixels para simular distorção."""
    columns, rows = inner
    texture = np.full(((rows + 1) * square_pixels, (columns + 1) * square_pixels), 235, np.uint8)
    for row in range(rows + 1):
        for col in range(columns + 1):
            color = 30 if (row + col) % 2 == 0 else 235
            y0, x0 = row * square_pixels, col * square_pixels
            texture[y0:y0 + square_pixels, x0:x0 + square_pixels] = color
    plane = np.array([[-square_size, -square_size, 0], [columns * square_size, -square_size, 0],
                      [columns * square_size, rows * square_size, 0], [-square_size, rows * square_size, 0]], np.float32)
    # A homografia descreve o plano ideal sem distorção; o remapeamento abaixo
    # aplica a distorção uma única vez à imagem inteira.
    projected, _ = cv2.projectPoints(plane, rvec, tvec, matrix, np.zeros(5, dtype=np.float64))
    source = np.array([[0, 0], [texture.shape[1] - 1, 0], [texture.shape[1] - 1, texture.shape[0] - 1],
                       [0, texture.shape[0] - 1]], np.float32)
    width, height = image_size
    homography = cv2.getPerspectiveTransform(source, projected.reshape(-1, 2).astype(np.float32))
    rendered = cv2.warpPerspective(texture, homography, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=127)
    # OpenCV undistortPoints maps distorted destination pixels back to undistorted source pixels.
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    destination = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(destination, matrix, distortion, P=matrix).reshape(height, width, 2)
    distorted_image = cv2.remap(rendered, undistorted[:, :, 0], undistorted[:, :, 1], cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=127)
    inner_points = cv2.projectPoints(chessboard_object_points(inner, square_size), rvec, tvec,
                                     matrix, distortion)[0].reshape(-1, 2)
    return cv2.cvtColor(distorted_image, cv2.COLOR_GRAY2BGR), inner_points


def generate(output: Path, count: int, image_size: tuple[int, int], inner: tuple[int, int],
             square_size: float, seed: int, square_pixels: int) -> None:
    """Gera vistas variadas, ground truth separado e manifesto de reprodutibilidade."""
    if count < 5 or min(image_size) < 160 or square_size <= 0:
        raise ValueError("Use pelo menos 5 imagens, resolução >=160 px e tamanho positivo.")
    config = json.loads((ROOT / "config" / "calibration_config.json").read_text(encoding="utf-8"))
    synthetic_config = config["synthetic"]
    rng = np.random.default_rng(seed)
    width, height = image_size
    matrix = np.array([[synthetic_config["focal_ratio_x"] * width, 0, width / 2],
                       [0, synthetic_config["focal_ratio_y"] * width, height / 2], [0, 0, 1]], np.float64)
    distortion = np.asarray(synthetic_config["distortion_coefficients"], dtype=np.float64)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    poses, views = [], []
    for idx in range(count):
        angles = np.deg2rad([rng.uniform(-22, 22), rng.uniform(-26, 26), rng.uniform(-14, 14)])
        rvec = angles.reshape(3, 1).astype(np.float64)
        tvec = np.array([[rng.uniform(-0.09, 0.02)], [rng.uniform(-0.075, 0.02)],
                         [rng.uniform(0.40, 0.58)]], np.float64)
        image, points = render_view(inner, square_size, square_pixels, image_size, matrix, distortion, rvec, tvec)
        if (points[:, 0].min() < 20 or points[:, 1].min() < 20 or points[:, 0].max() > width - 20 or
                points[:, 1].max() > height - 20):
            raise RuntimeError(f"Pose {idx} projetou cantos fora da região segura; ajuste parâmetros.")
        path = image_dir / f"view_{idx:03d}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Falha ao salvar {path}")
        poses.append({"rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist()})
        views.append({"file": path.name, "projected_corner_bounds_px": [points.min(0).tolist(), points.max(0).tolist()]})
    ground_truth = {"note": "Referência apenas para avaliação; não deve ser fornecida ao calibrador.",
                    "image_size": [width, height], "camera_matrix": matrix.tolist(),
                    "distortion_coefficients_k1_k2_p1_p2_k3": distortion.tolist(),
                    "board": {"inner_corners": list(inner), "square_size": square_size, "unit": "m"}, "poses": poses}
    write_json(output / "synthetic_ground_truth.json", ground_truth)
    write_json(output / "manifest.json", {"seed": seed, "image_count": count, "image_size": list(image_size),
                                          "interpolation": "OpenCV INTER_LINEAR; quantização raster pode afetar detecção",
                                          "views": views})
    sample = cv2.imread(str(image_dir / "view_000.png"))
    cv2.imwrite(str(output / "sample.png"), sample)
    print(f"Geradas {count} imagens em {image_dir}")


def main() -> None:
    config = json.loads((ROOT / "config" / "calibration_config.json").read_text(encoding="utf-8"))
    board_config, synthetic_config = config["board"], config["synthetic"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "synthetic")
    parser.add_argument("--images", type=int, default=synthetic_config["number_of_images"])
    parser.add_argument("--width", type=int, default=synthetic_config["image_size"][0])
    parser.add_argument("--height", type=int, default=synthetic_config["image_size"][1])
    parser.add_argument("--corners", type=int, nargs=2, default=board_config["inner_corners"], metavar=("COLUNAS", "LINHAS"))
    parser.add_argument("--square-size", type=float, default=board_config["square_size"], help="Tamanho físico dos quadrados")
    parser.add_argument("--square-pixels", type=int, default=synthetic_config["square_pixels"])
    parser.add_argument("--seed", type=int, default=synthetic_config["seed"])
    args = parser.parse_args()
    generate(args.output.resolve(), args.images, (args.width, args.height), tuple(args.corners),
             args.square_size, args.seed, args.square_pixels)


if __name__ == "__main__": main()
