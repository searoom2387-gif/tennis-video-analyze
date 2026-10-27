"""Player detection, tracking, track stitching and player selection."""

from dataclasses import replace
from typing import Protocol

import numpy as np

from tennis_analyze.geometry import apply_homography, foot_point
from tennis_analyze.types import Detection, PlayerSample, TrackedBox, TrackSample

COURT_WIDTH_M = 8.23
COURT_LENGTH_M = 23.77
PERSON_LABEL = 1  # COCO "person"


class PersonDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class TorchvisionPersonDetector:
    """Person detector using torchvision's SSDlite320 MobileNetV3 (BSD, CPU friendly).

    torch/torchvision are imported lazily so the rest of the package works without them.
    """

    def __init__(self, score_threshold: float = 0.5):
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        self._torch = torch
        self._score_threshold = score_threshold
        self._model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self._model.to("cpu")
        self._model.eval()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """`image` is a BGR uint8 array (OpenCV order)."""
        torch = self._torch
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            out = self._model([tensor])[0]
        detections = []
        for box, label, score in zip(
            out["boxes"].tolist(), out["labels"].tolist(), out["scores"].tolist()
        ):
            if label == PERSON_LABEL and score >= self._score_threshold:
                detections.append(Detection(tuple(box), float(score)))
        return detections


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


class PlayerTracker:
    """Greedy IoU multi-object tracker (ByteTrack-style two-stage association).

    supervision 0.31.2 no longer ships ByteTrack (it moved to the separate
    `trackers` package), so a small self-contained tracker is used instead.
    Each track keeps a constant-velocity estimate (smoothed box-center shift per
    frame) and detections are matched against the box predicted for the current
    frame, so two people who cross keep their ids.
    High-score detections are matched first, then low-score ones are used to
    keep existing tracks alive; only high-score unmatched detections start tracks.
    Tracks are dropped after `max_lost` consecutive missed frames.
    """

    def __init__(
        self,
        high_score: float = 0.6,
        min_iou: float = 0.1,
        max_lost: int = 10,
        velocity_smoothing: float = 0.5,
    ):
        self._high_score = high_score
        self._min_iou = min_iou
        self._max_lost = max_lost
        self._alpha = velocity_smoothing
        self._next_id = 1
        # id -> {"box": last matched box, "lost": missed frames, "vel": (dx, dy) per frame or None}
        self._tracks: dict[int, dict] = {}

    def _predicted(self, t: int):
        tr = self._tracks[t]
        if tr["vel"] is None:
            return tr["box"]
        steps = tr["lost"] + 1  # frames since the last match
        dx, dy = tr["vel"][0] * steps, tr["vel"][1] * steps
        x1, y1, x2, y2 = tr["box"]
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def _associate(self, track_ids, dets):
        predicted = {t: self._predicted(t) for t in track_ids}
        pairs = sorted(
            (
                (_iou(predicted[t], d.box_xyxy), t, i)
                for t in track_ids
                for i, d in enumerate(dets)
            ),
            reverse=True,
        )
        used_t, used_d, matches = set(), set(), []
        for iou, t, i in pairs:
            if iou < self._min_iou:
                break
            if t in used_t or i in used_d:
                continue
            used_t.add(t)
            used_d.add(i)
            matches.append((t, i))
        return matches, used_t, used_d

    def _match(self, t: int, box) -> None:
        tr = self._tracks[t]
        steps = tr["lost"] + 1
        (cx0, cy0), (cx1, cy1) = _center(tr["box"]), _center(box)
        measured = ((cx1 - cx0) / steps, (cy1 - cy0) / steps)
        if tr["vel"] is None:
            vel = measured
        else:
            a = self._alpha
            vel = (
                a * measured[0] + (1 - a) * tr["vel"][0],
                a * measured[1] + (1 - a) * tr["vel"][1],
            )
        self._tracks[t] = {"box": box, "lost": 0, "vel": vel}

    def update(self, detections: list[Detection]) -> list[TrackedBox]:
        high = [d for d in detections if d.score >= self._high_score]
        low = [d for d in detections if d.score < self._high_score]
        out: list[TrackedBox] = []
        matched: set[int] = set()

        matches, used_t, used_d = self._associate(list(self._tracks), high)
        for t, i in matches:
            self._match(t, high[i].box_xyxy)
            out.append(TrackedBox(t, tuple(high[i].box_xyxy)))
        matched |= used_t

        rest = [t for t in self._tracks if t not in matched]
        matches, used_t, _ = self._associate(rest, low)
        for t, i in matches:
            self._match(t, low[i].box_xyxy)
            out.append(TrackedBox(t, tuple(low[i].box_xyxy)))
        matched |= used_t

        for t in list(self._tracks):
            if t not in matched:
                self._tracks[t]["lost"] += 1
                if self._tracks[t]["lost"] > self._max_lost:
                    del self._tracks[t]

        for i, d in enumerate(high):
            if i not in used_d:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"box": d.box_xyxy, "lost": 0, "vel": None}
                out.append(TrackedBox(tid, tuple(d.box_xyxy)))
        return out


def to_track_samples(time_s: float, boxes: list[TrackedBox], h: np.ndarray) -> list[TrackSample]:
    """Project each box's foot point into court coordinates (meters)."""
    if not boxes:
        return []
    feet = np.array([foot_point(b.box_xyxy) for b in boxes], dtype=float)
    court = apply_homography(h, feet)
    return [
        TrackSample(time_s, b.track_id, float(x), float(y))
        for b, (x, y) in zip(boxes, court)
    ]


def stitch_tracks(
    samples: list[TrackSample], max_gap_s: float = 2.0, max_dist_m: float = 3.0
) -> list[TrackSample]:
    """Relabel a track that starts shortly after, and near, where another ended."""
    by_id: dict[int, list[TrackSample]] = {}
    for s in samples:
        by_id.setdefault(s.track_id, []).append(s)
    for lst in by_id.values():
        lst.sort(key=lambda s: s.time_s)

    mapping = {tid: tid for tid in by_id}
    taken_ends: set[int] = set()  # tracks already continued by a later track
    for tid in sorted(by_id, key=lambda t: by_id[t][0].time_s):
        first = by_id[tid][0]
        best, best_dist = None, None
        for other, lst in by_id.items():
            if other == tid or other in taken_ends:
                continue
            last = lst[-1]
            gap = first.time_s - last.time_s
            if gap < 0 or gap > max_gap_s:
                continue
            dist = float(np.hypot(first.x_m - last.x_m, first.y_m - last.y_m))
            if dist <= max_dist_m and (best_dist is None or dist < best_dist):
                best, best_dist = other, dist
        if best is not None:
            taken_ends.add(best)
            mapping[tid] = mapping[best]
    return [replace(s, track_id=mapping[s.track_id]) for s in samples]


def _in_court(s: TrackSample, margin: float) -> bool:
    return (
        -margin <= s.x_m <= COURT_WIDTH_M + margin
        and -margin <= s.y_m <= COURT_LENGTH_M + margin
    )


def select_players(
    samples: list[TrackSample], court_margin_m: float = 2.0
) -> tuple[list[PlayerSample], list[str]]:
    """Pick the two most-seen people on court and label them A/B by track id."""
    in_court = [s for s in samples if _in_court(s, court_margin_m)]
    by_id: dict[int, list[TrackSample]] = {}
    for s in in_court:
        by_id.setdefault(s.track_id, []).append(s)
    for lst in by_id.values():
        lst.sort(key=lambda s: s.time_s)

    top = sorted(by_id, key=lambda t: (-len(by_id[t]), by_id[t][0].time_s))[:2]
    top.sort(key=lambda t: by_id[t][0].y_m)  # A = nearer the camera (smaller y) at first sight
    labels = dict(zip(top, ["A", "B"]))

    warnings: list[str] = []
    if not top:
        warnings.append("選手が見つかりません。コート内に人物が検出されませんでした。")
    elif len(top) == 1:
        warnings.append("選手が1人しか見つかりませんでした。")

    out = [
        PlayerSample(s.time_s, labels[s.track_id], s.x_m, s.y_m)
        for s in in_court
        if s.track_id in labels
    ]
    out.sort(key=lambda p: (p.time_s, p.player))
    return out, warnings
