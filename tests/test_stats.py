import numpy as np
import pytest

from tennis_analyze.stats import compute_stats, heatmap
from tennis_analyze.types import PlayerSample


def _samples(player, pts, dt=0.2):
    return [PlayerSample(i * dt, player, x, y) for i, (x, y) in enumerate(pts)]


def test_straight_walk_distance_and_speed():
    pts = [(4.0, float(i) * 0.5) for i in range(21)]  # 0.5m刻みで10m、20ステップ
    s = compute_stats(_samples("A", pts))["A"]
    assert s.distance_m == pytest.approx(10.0, abs=0.3)
    assert s.mean_speed_mps == pytest.approx(2.5, abs=0.2)  # 0.5m / 0.2s


def test_stationary_player_with_tracking_jitter_has_near_zero_distance():
    rng = np.random.default_rng(0)
    pts = [(4 + rng.normal(0, 0.05), 5 + rng.normal(0, 0.05)) for _ in range(300)]
    s = compute_stats(_samples("A", pts))["A"]
    assert s.distance_m < 3.0  # 生のままだと 300 * 約0.09m = 約27m になる


def test_zone_shares_sum_to_one_per_axis():
    pts = [(1.0, 2.0)] * 3 + [(7.0, 20.0)]
    s = compute_stats(_samples("A", pts))["A"]
    assert s.share_left + s.share_right == pytest.approx(1.0)
    assert s.share_front + s.share_back == pytest.approx(1.0)
    assert s.share_left == pytest.approx(0.75) and s.share_front == pytest.approx(0.75)


def test_player_without_samples_is_absent():
    assert "B" not in compute_stats(_samples("A", [(1.0, 1.0), (1.0, 2.0)]))


def test_single_sample_has_zero_distance_and_speed():
    s = compute_stats(_samples("A", [(1.0, 1.0)]))["A"]
    assert s.distance_m == 0.0 and s.mean_speed_mps == 0.0


def test_heatmap_counts_samples_and_ignores_outside_court():
    pts = [(1.0, 1.0)] * 4 + [(-3.0, 5.0)]
    hm = heatmap(_samples("A", pts), "A")
    assert hm.shape == (24, 48) and hm.sum() == 4
