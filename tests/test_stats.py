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
    # (1,2) ベースライン側・左 / (7,8) ネット側・右 / (7,20) ベースライン側・左(x を反転して 1.23)
    pts = [(1.0, 2.0)] * 2 + [(7.0, 8.0), (7.0, 20.0)]
    s = compute_stats(_samples("A", pts))["A"]
    assert s.share_left + s.share_right == pytest.approx(1.0)
    assert s.share_front + s.share_back == pytest.approx(1.0)
    assert s.share_left == pytest.approx(0.75) and s.share_front == pytest.approx(0.25)


def test_mirrored_half_gives_same_shares_after_change_of_ends():
    near = [(7.0, 8.0), (6.5, 3.0), (2.0, 1.0)]
    far = [(8.23 - x, 23.77 - y) for x, y in near]  # 点対称(サイドチェンジ後の同じ動き)
    a = compute_stats(_samples("A", near))["A"]
    b = compute_stats(_samples("A", far))["A"]
    assert (a.share_front, a.share_left) == pytest.approx((b.share_front, b.share_left))


def test_front_boundary_is_exclusive_at_6_40m_from_net():
    # 浮動小数点の誤差を避けるため、境界のちょうど内側(6.39)と外側(6.41)で確認する
    net_y = 11.885
    inside = compute_stats(_samples("A", [(4.0, net_y - 6.39)]))["A"]
    outside = compute_stats(_samples("A", [(4.0, net_y - 6.41)]))["A"]
    assert inside.share_front == 1.0 and outside.share_front == 0.0
    far_inside = compute_stats(_samples("A", [(4.0, net_y + 6.39)]))["A"]
    far_outside = compute_stats(_samples("A", [(4.0, net_y + 6.41)]))["A"]
    assert far_inside.share_front == 1.0 and far_outside.share_front == 0.0


def test_player_without_samples_is_absent():
    assert "B" not in compute_stats(_samples("A", [(1.0, 1.0), (1.0, 2.0)]))


def test_single_sample_has_zero_distance_and_speed():
    s = compute_stats(_samples("A", [(1.0, 1.0)]))["A"]
    assert s.distance_m == 0.0 and s.mean_speed_mps == 0.0


def test_heatmap_counts_samples_and_ignores_outside_court():
    pts = [(1.0, 1.0)] * 4 + [(-3.0, 5.0)]
    hm = heatmap(_samples("A", pts), "A")
    assert hm.shape == (24, 48) and hm.sum() == 4


def test_slow_drift_with_jitter_is_not_undercounted():
    rng = np.random.default_rng(1)
    pts = [(4 + rng.normal(0, 0.05), 2.0 + 0.4 * i * 0.2 + rng.normal(0, 0.05)) for i in range(51)]
    s = compute_stats(_samples("A", pts))["A"]
    assert s.distance_m == pytest.approx(4.0, abs=1.0)  # 0.4 m/s for 10 s
