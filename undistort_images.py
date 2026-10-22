"""Aplica correção de distorção a um diretório de imagens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from camera_calibration.distortion import undistort_image
from camera_calibration.io_utils import read_calibration_yaml
from camera_calibration.visualization import save_undistortion_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "synthetic" / "images")
    parser.add_argument("--calibration", type=Path, default=ROOT / "results" / "camera_calibration.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--alpha", type=float, default=0.0, help="0 recorta bordas pretas; 1 preserva campo de visão")
    args = parser.parse_args()
    calibration = read_calibration_yaml(args.calibration.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images.resolve().iterdir() if p.suffix.lower() in
                    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not images:
        raise ValueError(f"Nenhuma imagem encontrada em {args.images}")
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Ignorada imagem ilegível: {path.name}"); continue
        if (image.shape[1], image.shape[0]) != calibration["image_size"]:
            raise ValueError(f"Resolução de {path.name} diverge da calibração.")
        corrected, _ = undistort_image(image, calibration["camera_matrix"],
                                       calibration["distortion_coefficients"], args.alpha)
        cv2.imwrite(str(args.output / path.name), corrected)
        if path == images[0]:
            save_undistortion_comparison(image, corrected, args.output / "comparison.png")
    print(f"Imagens corrigidas salvas em {args.output.resolve()}")


if __name__ == "__main__": main()
