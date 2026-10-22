import numpy as np

from camera_calibration.board import chessboard_object_points
from camera_calibration.geometry import project_points


def test_board_geometry_and_projection():
    points = chessboard_object_points((8, 6), 0.025)
    assert points.shape == (48, 3)
    assert np.all(points[:, 2] == 0)
    projected = project_points(points[:2], np.zeros((3, 1)), np.array([[0], [0], [1.]]),
                               np.array([[800., 0, 640], [0, 800, 480], [0, 0, 1]]), np.zeros(5))
    np.testing.assert_allclose(projected, [[640, 480], [660, 480]])
