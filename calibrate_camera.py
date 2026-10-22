"""Calibra imagens de tabuleiro e exporta métricas, poses, YAML e figuras."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from camera_calibration.calibration import calibrate_directory
from camera_calibration.board import detect_chessboard
from camera_calibration.io_utils import write_calibration_yaml, write_json
from camera_calibration.visualization import save_error_histogram, save_sample


def main() -> None:
    config = json.loads((ROOT / "config" / "calibration_config.json").read_text(encoding="utf-8"))
    board_config = config["board"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "synthetic" / "images")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--corners", type=int, nargs=2, default=board_config["inner_corners"], metavar=("COLUNAS", "LINHAS"))
    parser.add_argument("--square-size", type=float, default=board_config["square_size"])
    parser.add_argument("--unit", default=board_config["unit"])
    parser.add_argument("--minimum-views", type=int, default=config["minimum_views"])
    parser.add_argument("--window", type=int, nargs=2, default=config["subpixel_window"])
    args = parser.parse_args()
    result = calibrate_directory(args.images.resolve(), tuple(args.corners), args.square_size,
                                 args.minimum_views, tuple(args.window))
    args.output.mkdir(parents=True, exist_ok=True)
    write_calibration_yaml(args.output / "camera_calibration.yaml", result["camera_matrix"],
                           result["distortion_coefficients"], result["image_size"], args.unit)
    report = {"image_size": list(result["image_size"]), "camera_matrix": result["camera_matrix"].tolist(),
              "distortion_coefficients_k1_k2_p1_p2_k3": result["distortion_coefficients"].tolist(),
              "distortion_model": "OpenCV standard pinhole, 5 coefficients", "square_size": args.square_size,
              "square_unit": args.unit, "opencv_calibrateCamera_rms_px": result["opencv_rms"],
              "metrics": result["metrics"], "poses": result["poses"],
              "used_files": [p.name for p in result["used_files"]], "rejected_files": result["rejected_files"]}
    write_json(args.output / "calibration_report.json", report)
    sample_path = result["used_files"][0]
    image = cv2.imread(str(sample_path))
    corners = detect_chessboard(image, tuple(args.corners), tuple(args.window))
    save_sample(sample_path, corners, args.output / "figures" / "detected_corners.png", tuple(args.corners))
    save_error_histogram(result["metrics"]["global"]["point_errors_px"],
                         args.output / "figures" / "reprojection_errors.png")
    print(json.dumps({"views_used": len(result["used_files"]), "opencv_rms_px": result["opencv_rms"],
                      "mean_per_point_px": result["metrics"]["global"]["mean_per_point_px"],
                      "rms_coordinates_px": result["metrics"]["global"]["rms_coordinates_px"],
                      "yaml": str(args.output / "camera_calibration.yaml")}, indent=2))


if __name__ == "__main__": main()
