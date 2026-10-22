import numpy as np

from camera_calibration.io_utils import read_calibration_yaml, write_calibration_yaml


def test_yaml_round_trip(tmp_path):
    matrix = np.array([[800., 0, 640], [0, 810, 480], [0, 0, 1]])
    distortion = np.array([-.1, .02, .001, -.002, 0.])
    path = tmp_path / "nested" / "calibration.yaml"
    write_calibration_yaml(path, matrix, distortion, (1280, 960))
    loaded = read_calibration_yaml(path)
    np.testing.assert_allclose(loaded["camera_matrix"], matrix)
    np.testing.assert_allclose(loaded["distortion_coefficients"], distortion)
    assert loaded["image_size"] == (1280, 960)


def test_missing_yaml_raises(tmp_path):
    try:
        read_calibration_yaml(tmp_path / "missing.yaml")
        assert False, "Deveria informar ausência de arquivo"
    except FileNotFoundError:
        pass
