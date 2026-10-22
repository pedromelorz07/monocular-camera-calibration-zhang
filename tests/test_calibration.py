import numpy as np
from pathlib import Path

from camera_calibration.calibration import calibrate_directory
from camera_calibration.metrics import reprojection_metrics


def test_metrics_definitions():
    observed = [np.array([[0., 0.], [3., 4.]])]
    projected = [np.zeros((2, 2))]
    metrics = reprojection_metrics(observed, projected)
    assert metrics["global"]["mean_per_point_px"] == 2.5
    assert metrics["global"]["max_per_point_px"] == 5
    assert metrics["global"]["rms_coordinates_px"] == np.sqrt(12.5)


def test_synthetic_generator_has_cli():
    # Verifica importabilidade sem gerar arquivos ou depender do diretório atual.
    import generate_synthetic_board  # noqa: F401


def test_synthetic_calibration_recovers_intrinsics(tmp_path):
    from generate_synthetic_board import generate
    import json

    output = Path(tmp_path) / "synthetic"
    generate(output, 12, (800, 600), (8, 6), 0.025, 73, 90)
    truth = json.loads((output / "synthetic_ground_truth.json").read_text(encoding="utf-8"))
    result = calibrate_directory(output / "images", (8, 6), 0.025, minimum_views=8)
    expected = np.asarray(truth["camera_matrix"])
    estimated = result["camera_matrix"]
    assert len(result["used_files"]) >= 8
    assert np.all(np.isfinite(estimated))
    assert result["distortion_coefficients"].shape == (5,)
    assert np.isfinite(result["metrics"]["global"]["mean_per_point_px"])
    assert result["metrics"]["global"]["mean_per_point_px"] >= 0
    assert abs(estimated[0, 0] / expected[0, 0] - 1) < 0.05
    assert abs(estimated[1, 1] / expected[1, 1] - 1) < 0.05
    assert np.linalg.norm(estimated[:2, 2] - expected[:2, 2]) < 35
    assert result["metrics"]["global"]["rms_coordinates_px"] < 0.6
