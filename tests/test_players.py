import pytest
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


def _run_crossing(speed, x_left=200.0, x_right=600.0, y_offset=0.0, w=40.0, h=100.0,
                  reverse=False):
    """Two equal-size boxes walk toward each other, cross, and walk apart.

    Returns, per frame, {person: track_id} where person 0 started on the left.
    Frames where both boxes are identical are skipped (the people are indistinguishable).
    """
    tracker = PlayerTracker()
    gap = x_right - x_left
    n = int(round(2 * gap / (2 * speed))) + 1  # until they are as far apart again
    per_frame = []
    for i in range(n):
        boxes = [
            (x_left + speed * i, 200.0, x_left + speed * i + w, 200.0 + h),
            (x_right - speed * i, 200.0 + y_offset, x_right - speed * i + w, 200.0 + y_offset + h),
        ]
        dets = [Detection(b, 0.9) for b in boxes]
        out = tracker.update(dets[::-1] if reverse else dets)
        if boxes[0] == boxes[1]:
            continue
        ids = {}
        for tb in out:
            person = boxes.index(tuple(tb.box_xyxy))
            ids[person] = tb.track_id
        per_frame.append(ids)
    return per_frame


def _assert_ids_follow_people(per_frame):
    first, last = per_frame[0], per_frame[-1]
    assert set(first) == {0, 1} and set(last) == {0, 1}
    assert first[0] != first[1]
    assert last == first  # same physical person keeps the same id after crossing
    assert all(f == first for f in per_frame)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("speed", [5.0, 10.0, 20.0])
def test_tracker_keeps_ids_when_two_people_cross(speed, reverse):
    # the boxes coincide exactly at the crossing frame; detection order must not matter
    _assert_ids_follow_people(_run_crossing(speed, reverse=reverse))


@pytest.mark.parametrize("speed", [5.0, 10.0, 20.0])
def test_tracker_keeps_ids_when_crossing_without_exact_coincidence(speed):
    # start 395 px apart so the boxes never coincide exactly, but overlap heavily
    _assert_ids_follow_people(_run_crossing(speed, x_right=595.0))


@pytest.mark.parametrize("speed", [5.0, 10.0, 20.0])
def test_tracker_keeps_ids_when_crossing_with_heavy_vertical_overlap(speed):
    _assert_ids_follow_people(_run_crossing(speed, x_right=595.0, y_offset=10.0))


def test_tracker_predicts_through_short_miss_while_crossing():
    tracker = PlayerTracker()
    first = None
    for i in range(41):
        boxes = [
            (200.0 + 10 * i, 200.0, 240.0 + 10 * i, 300.0),
            (600.0 - 10 * i, 205.0, 640.0 - 10 * i, 305.0),
        ]
        if 19 <= i <= 21:  # both missed around the crossing (occlusion)
            tracker.update([])
            continue
        out = tracker.update([Detection(b, 0.9) for b in boxes])
        ids = {boxes.index(tuple(tb.box_xyxy)): tb.track_id for tb in out}
        if first is None:
            first = ids
        assert ids == first


def test_tracker_new_far_detection_gets_new_id_and_empty_frames_are_ok():
    tracker = PlayerTracker()
    a = tracker.update([Detection((100.0, 100.0, 140.0, 200.0), 0.9)])
    assert tracker.update([]) == []
    b = tracker.update([Detection((104.0, 100.0, 144.0, 200.0), 0.9),
                        Detection((900.0, 500.0, 940.0, 600.0), 0.9)])
    ids = {round(tb.box_xyxy[0]): tb.track_id for tb in b}
    assert ids[104] == a[0].track_id
    assert ids[900] > a[0].track_id
