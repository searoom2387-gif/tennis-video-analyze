import numpy as np
import pytest
from synthetic import make_homography, render_court

from tennis_analyze.court_detect import (
    CourtDetectionError,
    court_from_corners,
    detect_court,
    render_court_check,
)
from tennis_analyze.geometry import apply_homography


def test_recovers_homography_from_synthetic_court():
    h_true = make_homography(seed=0)  # コート座標 -> px
    img = render_court(h_true)
    result = detect_court([img] * 5)
    # 検出結果は px -> コート座標。四隅を往復して誤差を見る
    corners_m = np.array([[0, 0], [8.23, 0], [8.23, 23.77], [0, 23.77]])
    px = apply_homography(h_true, corners_m)
    back = apply_homography(result.homography, px)
    assert np.abs(back - corners_m).max() < 0.15  # 15cm以内
    assert result.confidence >= 0.8


def test_doubles_lines_do_not_pull_detection_outward():
    h_true = make_homography(seed=1)
    img = render_court(h_true, doubles_lines=True)
    result = detect_court([img] * 5)
    px = apply_homography(h_true, np.array([[0.0, 0.0], [8.23, 23.77]]))
    back = apply_homography(result.homography, px)
    assert np.abs(back - np.array([[0.0, 0.0], [8.23, 23.77]])).max() < 0.3


def test_blank_image_raises_court_detection_error():
    with pytest.raises(CourtDetectionError):
        detect_court([np.zeros((1080, 1920, 3), np.uint8)] * 5)


def test_inconsistent_frames_raise_instead_of_guessing():
    imgs = [render_court(make_homography(s)) for s in (0, 1, 2, 3, 4)]
    with pytest.raises(CourtDetectionError, match="一致"):
        detect_court(imgs)


def test_court_from_corners_maps_corners_exactly():
    corners = [(300, 900), (1600, 900), (1200, 300), (700, 300)]
    r = court_from_corners(corners)
    out = apply_homography(r.homography, np.array(corners, float))
    assert np.allclose(out, [[0, 0], [8.23, 0], [8.23, 23.77], [0, 23.77]], atol=1e-6)


def test_render_court_check_returns_same_shape_image():
    img = render_court(make_homography(0))
    r = detect_court([img] * 3)
    assert render_court_check(img, r).shape == img.shape
