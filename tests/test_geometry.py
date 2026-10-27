import numpy as np

from tennis_analyze.geometry import apply_homography, foot_point


def test_apply_homography_identity_and_scale():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(apply_homography(np.eye(3), pts), pts)
    assert np.allclose(apply_homography(np.diag([2.0, 2.0, 1.0]), pts), pts * 2)


def test_foot_point_is_bottom_center_of_box():
    assert foot_point((10.0, 20.0, 30.0, 80.0)) == (20.0, 80.0)
