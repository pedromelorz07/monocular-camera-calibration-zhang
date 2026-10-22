import numpy as np

from camera_calibration.distortion import undistort_image


def test_undistortion_preserves_dimensions():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    matrix = np.array([[100., 0, 80], [0, 100, 60], [0, 0, 1]])
    corrected, new_matrix = undistort_image(image, matrix, np.zeros(5))
    assert corrected.shape == image.shape
    assert new_matrix.shape == (3, 3)
