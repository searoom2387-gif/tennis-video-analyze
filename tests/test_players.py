import numpy as np

from tennis_analyze.players import (
    PlayerTracker,
    select_players,
    stitch_tracks,
    to_track_samples,
)
from tennis_analyze.types import Detection, TrackedBox, TrackSample


def _walk(track_id, t0, t1, xy0, xy1, n=20):
    ts = np.linspace(t0, t1, n)
    return [
        TrackSample(
            float(t),
            track_id,
            float(np.interp(t, [t0, t1], [xy0[0], xy1[0]])),
            float(np.interp(t, [t0, t1], [xy0[1], xy1[1]])),
        )
        for t in ts
    ]


def test_select_players_keeps_two_longest_tracks_inside_court():
    players = _walk(1, 0, 60, (2, 3), (3, 4)) + _walk(2, 0, 60, (5, 20), (4, 19))
    ball_person = _walk(3, 0, 5, (-8.0, 10.0), (-8.0, 11.0))  # 8 m outside the court
    out, warns = select_players(players + ball_person)
    assert {s.player for s in out} == {"A", "B"}
    assert warns == []


def test_player_a_starts_on_near_side_y_small():
    out, _ = select_players(_walk(1, 0, 60, (2, 3), (3, 4)) + _walk(2, 0, 60, (5, 20), (4, 19)))
    first_a = next(s for s in out if s.player == "A")
    assert first_a.y_m < 23.77 / 2


def test_labels_follow_person_after_change_of_ends():
    p1 = _walk(1, 0, 30, (4, 3), (4, 5)) + _walk(1, 30, 60, (4, 20), (4, 19))
    p2 = _walk(2, 0, 30, (4, 20), (4, 19)) + _walk(2, 30, 60, (4, 3), (4, 5))
    out, _ = select_players(p1 + p2)
    a_late = [s for s in out if s.player == "A" and s.time_s > 35]
    assert a_late
    assert all(s.y_m > 23.77 / 2 for s in a_late)  # A is the person who moved to the far side


def test_one_person_found_returns_samples_and_warning():
    out, warns = select_players(_walk(1, 0, 60, (2, 3), (3, 4)))
    assert {s.player for s in out} == {"A"}
    assert any("1人" in w for w in warns)


def test_no_person_found_returns_empty_and_warning():
    out, warns = select_players([])
    assert out == [] and any("見つかりません" in w for w in warns)


def test_stitch_joins_id_switch_nearby_in_time_and_space():
    a = _walk(1, 0, 10, (2, 3), (2.5, 3.5))
    b = _walk(9, 10.5, 20, (2.6, 3.6), (3, 4))
    stitched = stitch_tracks(a + b)
    assert len({s.track_id for s in stitched}) == 1


def test_stitch_does_not_join_far_apart_tracks():
    a = _walk(1, 0, 10, (2, 3), (2, 3))
    b = _walk(9, 10.5, 20, (6, 20), (6, 20))
    assert len({s.track_id for s in stitch_tracks(a + b)}) == 2


def test_stitch_does_not_join_overlapping_tracks():
    a = _walk(1, 0, 10, (2, 3), (2, 3))
    b = _walk(9, 5, 15, (2.5, 3.5), (2.5, 3.5))
    assert len({s.track_id for s in stitch_tracks(a + b)}) == 2


def test_to_track_samples_maps_foot_point_through_homography():
    h = np.array([[0.01, 0, 0], [0, 0.02, 0], [0, 0, 1.0]])
    boxes = [TrackedBox(7, (90.0, 100.0, 110.0, 300.0))]
    (s,) = to_track_samples(1.5, boxes, h)
    assert (s.time_s, s.track_id) == (1.5, 7)
    assert np.isclose(s.x_m, 1.0) and np.isclose(s.y_m, 6.0)


def test_tracker_gives_stable_ids_to_two_moving_boxes():
    tracker = PlayerTracker()
    seen = []
    for i in range(20):
        dets = [
            Detection((100 + 4 * i, 200.0, 140 + 4 * i, 300.0), 0.9),
            Detection((500 - 4 * i, 100.0, 540 - 4 * i, 200.0), 0.9),
        ]
        seen.append({round(b.box_xyxy[0]) < 300: b.track_id for b in tracker.update(dets)})
    left = {d[True] for d in seen if True in d}
    right = {d[False] for d in seen if False in d}
    assert len(left) == 1 and len(right) == 1 and left != right
    assert all(len(d) == 2 for d in seen[2:])
