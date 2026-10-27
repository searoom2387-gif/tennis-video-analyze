"""Line-based tennis court detection (classic CV, no learned models).

Homographies stored in CourtResult map PIXELS -> COURT METERS.
"""

from collections.abc import Sequence
from itertools import combinations

import cv2
import numpy as np

from tennis_analyze.court_model import COURT_LENGTH_M, COURT_WIDTH_M, court_lines
from tennis_analyze.geometry import apply_homography
from tennis_analyze.types import CourtResult


class CourtDetectionError(Exception):
    pass


# Provisional value; to be tuned on real video in Task 8.
MIN_CONFIDENCE = 0.5

_CORNERS_M = np.array(
    [
        [0.0, 0.0],
        [COURT_WIDTH_M, 0.0],
        [COURT_WIDTH_M, COURT_LENGTH_M],
        [0.0, COURT_LENGTH_M],
    ]
)
# Model lines used for the 2+2 assignment: horizontal ones are y = const, vertical x = const.
_MODEL_HORIZONTAL_Y = sorted({p[1] for p, q in court_lines() if p[1] == q[1]})
_MODEL_VERTICAL_X = sorted({p[0] for p, q in court_lines() if p[0] == q[0]})

_SAMPLE_STEP_M = 0.1
_SCORE_DILATE_PX = 2
_HOUGH_MAX_WIDTH = 960  # Hough runs on a downscaled mask for speed
_HORIZONTAL_MAX_DEG = 30.0  # segments flatter than this are "horizontal"
_MERGE_ANGLE_DEG = 3.0
_MERGE_DIST_PX = 8.0
_REFIT_BAND_PX = 4.0
_MAX_LINES_PER_FAMILY = 8
_MATCH_DIST_PX = 5.0
_VOTE_TOLERANCE_PX = 20.0


def extract_line_mask(image: np.ndarray) -> np.ndarray:
    """Bright, low-saturation pixels that are brighter than their surroundings (uint8, line=255)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local_mean = cv2.boxFilter(gray, cv2.CV_32F, (31, 31))
    is_line = (val >= 170) & (sat <= 70) & (gray.astype(np.float32) - local_mean >= 15)
    return is_line.astype(np.uint8) * 255


def _model_sample_points() -> np.ndarray:
    points = []
    for (x1, y1), (x2, y2) in court_lines():
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / _SAMPLE_STEP_M) + 1)
        t = np.linspace(0.0, 1.0, n)
        points.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1))
    return np.concatenate(points)


_MODEL_SAMPLES = _model_sample_points()


def _score_court_to_px(h_court_to_px: np.ndarray, dilated: np.ndarray) -> float:
    """Fraction of model line samples that land on the (already dilated) mask."""
    px = apply_homography(h_court_to_px, _MODEL_SAMPLES)
    if not np.all(np.isfinite(px)):
        return 0.0
    xi = np.round(px[:, 0]).astype(np.int64)
    yi = np.round(px[:, 1]).astype(np.int64)
    height, width = dilated.shape[:2]
    inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    hits = np.zeros(len(px), bool)
    hits[inside] = dilated[yi[inside], xi[inside]] > 0
    return float(hits.mean())


def _dilate(mask: np.ndarray) -> np.ndarray:
    k = 2 * _SCORE_DILATE_PX + 1
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def score_homography(h: np.ndarray, mask: np.ndarray) -> float:
    """Score a PIXELS -> COURT homography (CourtResult convention) against a line mask.

    Model lines are sampled every 0.1 m, projected into the image with the inverse of h,
    and the fraction landing on the mask dilated by 2 px is returned (0..1).
    """
    return _score_court_to_px(np.linalg.inv(h), _dilate(mask))


# --- line detection --------------------------------------------------------------------


def _line_through(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Total-least-squares line (a, b, c) with a^2 + b^2 = 1, a*x + b*y + c = 0."""
    w = np.ones(len(points)) if weights is None else weights
    centroid = (points * w[:, None]).sum(0) / w.sum()
    d = points - centroid
    cov = (d * w[:, None]).T @ d
    _, vecs = np.linalg.eigh(cov)
    normal = vecs[:, 0]  # smallest eigenvalue -> normal direction
    return np.array([normal[0], normal[1], -normal @ centroid])


def _segment_lines(mask: np.ndarray) -> list[np.ndarray]:
    """Hough segments (full-res coordinates) merged into lines, refit on full-res mask pixels."""
    width = mask.shape[1]
    scale = min(1.0, _HOUGH_MAX_WIDTH / width)
    small = (
        cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1
        else mask
    )
    small = (small > 0).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(
        small, 1, np.pi / 360, threshold=30, minLineLength=30, maxLineGap=8
    )
    if segs is None:
        return []
    segs = segs.reshape(-1, 4).astype(float) / scale
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])

    # Greedy clustering, longest segments first.
    clusters: list[list[int]] = []
    for i in np.argsort(-lengths):
        x1, y1, x2, y2 = segs[i]
        ang_i = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        mid_i = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        for cluster in clusters:
            j = cluster[0]
            a1, b1, a2, b2 = segs[j]
            ang_j = np.degrees(np.arctan2(b2 - b1, a2 - a1)) % 180
            d_ang = abs(ang_i - ang_j)
            if min(d_ang, 180 - d_ang) > _MERGE_ANGLE_DEG:
                continue
            line_j = _line_through(np.array([[a1, b1], [a2, b2]]))
            if abs(line_j[:2] @ mid_i + line_j[2]) <= _MERGE_DIST_PX:
                cluster.append(i)
                break
        else:
            clusters.append([i])

    ys, xs = np.nonzero(mask)
    mask_pts = np.stack([xs, ys], axis=1).astype(float)
    lines = []
    for cluster in clusters:
        pts = segs[cluster].reshape(-1, 2)
        line = _line_through(pts, np.repeat(lengths[cluster], 2))
        # Refine on full-resolution mask pixels in a band around the line, within its extent.
        direction = np.array([-line[1], line[0]])
        t = pts @ direction
        near = np.abs(mask_pts @ line[:2] + line[2]) <= _REFIT_BAND_PX
        along = mask_pts[near] @ direction
        band = mask_pts[near][(along >= t.min()) & (along <= t.max())]
        if len(band) >= 10:
            line = _line_through(band)
        lines.append((line, float(lengths[cluster].sum())))
    lines.sort(key=lambda item: -item[1])
    return [line for line, _ in lines]


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def _plausible(h_court_to_px: np.ndarray, shape: tuple[int, ...]) -> bool:
    """Projected court must be a convex, non-tiny quad near the image, in front of the camera."""
    corners_h = np.hstack([_CORNERS_M, np.ones((4, 1))]) @ h_court_to_px.T
    if np.any(corners_h[:, 2] * corners_h[0, 2] <= 0):
        return False
    quad = corners_h[:, :2] / corners_h[:, 2:3]
    height, width = shape[:2]
    margin = 0.5
    if np.any(quad[:, 0] < -margin * width) or np.any(
        quad[:, 0] > (1 + margin) * width
    ):
        return False
    if np.any(quad[:, 1] < -margin * height) or np.any(
        quad[:, 1] > (1 + margin) * height
    ):
        return False
    if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return False
    return cv2.contourArea(quad.astype(np.float32)) >= 0.01 * width * height


def _match_distance(line: np.ndarray, h_court_to_px: np.ndarray, p_m, q_m) -> float:
    """Mean pixel distance from the projected model segment p_m-q_m to a detected line."""
    px = apply_homography(h_court_to_px, np.linspace(p_m, q_m, 20))
    return float(np.mean(np.abs(px @ line[:2] + line[2])))


def _closest_line(lines, h_court_to_px, p_m, q_m) -> np.ndarray | None:
    if not lines:
        return None
    dists = [_match_distance(ln, h_court_to_px, p_m, q_m) for ln in lines]
    i = int(np.argmin(dists))
    return lines[i] if dists[i] <= _MATCH_DIST_PX else None


def _refine(h_court_to_px, horizontals, verticals) -> np.ndarray:
    """Re-estimate H from the intersections of the detected lines closest to each model line."""
    matched_h = [
        (y, ln)
        for y in _MODEL_HORIZONTAL_Y
        if (
            ln := _closest_line(
                horizontals, h_court_to_px, (0.0, y), (COURT_WIDTH_M, y)
            )
        )
        is not None
    ]
    matched_v = [
        (x, ln)
        for x in _MODEL_VERTICAL_X
        if (
            ln := _closest_line(verticals, h_court_to_px, (x, 0.0), (x, COURT_LENGTH_M))
        )
        is not None
    ]
    src, dst = [], []
    for y, lh in matched_h:
        for x, lv in matched_v:
            p = _intersect(lh, lv)
            if p is not None:
                src.append((x, y))
                dst.append(p)
    if len(src) < 5:
        return h_court_to_px
    refined, _ = cv2.findHomography(np.array(src, float), np.array(dst, float), 0)
    return h_court_to_px if refined is None else refined


def detect_court_frame(image: np.ndarray) -> CourtResult | None:
    """Detect the singles court in one frame; None when no candidate is found.

    Detected lines are split into horizontal and vertical families. Every assignment of
    2 horizontal + 2 vertical lines (in image order) to 2 + 2 model lines (in court order)
    gives a homography from the 4 intersections; all are scored with the line mask and the
    best is refined with every matched intersection. Enumeration is exhaustive, so the
    result is deterministic.
    """
    mask = extract_line_mask(image)
    lines = _segment_lines(mask)
    if not lines:
        return None
    horizontals, verticals = [], []
    for line in lines:
        # Normal (a, b): a horizontal line has a mostly vertical normal.
        angle_from_horizontal = np.degrees(np.arctan2(abs(line[0]), abs(line[1])))
        (
            horizontals if angle_from_horizontal < _HORIZONTAL_MAX_DEG else verticals
        ).append(line)
    horizontals = horizontals[:_MAX_LINES_PER_FAMILY]
    verticals = verticals[:_MAX_LINES_PER_FAMILY]
    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    mid_x, mid_y = image.shape[1] / 2, image.shape[0] / 2
    # Image order: horizontals bottom -> top (near -> far), verticals left -> right.
    horizontals.sort(key=lambda ln: -(-(ln[0] * mid_x + ln[2]) / ln[1]))
    verticals.sort(key=lambda ln: -(ln[1] * mid_y + ln[2]) / ln[0])

    dilated = _dilate(mask)
    best_score, best_h = 0.0, None
    for h1, h2 in combinations(horizontals, 2):
        for v1, v2 in combinations(verticals, 2):
            pts = [_intersect(h, v) for h in (h1, h2) for v in (v1, v2)]
            if any(p is None for p in pts):
                continue
            dst = np.array(pts, np.float32)
            for my1, my2 in combinations(_MODEL_HORIZONTAL_Y, 2):
                for mx1, mx2 in combinations(_MODEL_VERTICAL_X, 2):
                    src = np.array(
                        [[mx1, my1], [mx2, my1], [mx1, my2], [mx2, my2]], np.float32
                    )
                    h = cv2.getPerspectiveTransform(src, dst).astype(float)
                    if not _plausible(h, image.shape):
                        continue
                    score = _score_court_to_px(h, dilated)
                    if score > best_score:
                        best_score, best_h = score, h
    if best_h is None:
        return None

    refined = _refine(best_h, horizontals, verticals)
    refined_score = _score_court_to_px(refined, dilated)
    if _plausible(refined, image.shape) and refined_score >= best_score:
        best_h, best_score = refined, refined_score
    return CourtResult(homography=np.linalg.inv(best_h), confidence=best_score)


def _corners_px(result: CourtResult) -> np.ndarray:
    return apply_homography(np.linalg.inv(result.homography), _CORNERS_M)


def detect_court(
    images: Sequence[np.ndarray], min_confidence: float = MIN_CONFIDENCE
) -> CourtResult:
    """Detect the court across frames and take the majority of mutually consistent results.

    Two frames agree when all four projected court corners lie within 20 px. The largest
    agreeing group must hold more than half of the frames; its corners are median-combined.
    """
    if not images:
        raise CourtDetectionError("コート検出用のフレームがありません")
    results = [r for r in (detect_court_frame(img) for img in images) if r is not None]
    if not results:
        raise CourtDetectionError("どのフレームでもコートを検出できませんでした")

    corners = [_corners_px(r) for r in results]
    best_group: list[int] = []
    for i in range(len(results)):
        group = [
            j
            for j in range(len(results))
            if np.linalg.norm(corners[i] - corners[j], axis=1).max()
            <= _VOTE_TOLERANCE_PX
        ]
        if len(group) > len(best_group):
            best_group = group
    if 2 * len(best_group) <= len(images):
        raise CourtDetectionError(
            f"フレーム間で検出結果が一致しません(一致 {len(best_group)} / {len(images)} フレーム)"
        )

    median_corners = np.median(np.stack([corners[j] for j in best_group]), axis=0)
    confidence = float(np.median([results[j].confidence for j in best_group]))
    if confidence < min_confidence:
        raise CourtDetectionError(
            f"コート検出の信頼度が低すぎます: {confidence:.2f}(しきい値 {min_confidence:.2f})"
        )
    homography = cv2.getPerspectiveTransform(
        median_corners.astype(np.float32), _CORNERS_M.astype(np.float32)
    ).astype(float)
    return CourtResult(homography=homography, confidence=confidence)


def court_from_corners(corners_px: Sequence[tuple[float, float]]) -> CourtResult:
    """Court from 4 pixel corners: near-left, near-right, far-right, far-left. Confidence 1.0."""
    src = np.asarray(corners_px, float).reshape(4, 2)
    homography, _ = cv2.findHomography(src, _CORNERS_M, 0)
    if homography is None:
        raise CourtDetectionError("指定された四隅からホモグラフィを計算できません")
    return CourtResult(homography=homography, confidence=1.0)


def render_court_check(image: np.ndarray, result: CourtResult) -> np.ndarray:
    """Copy of image with the court model lines projected through the detected homography."""
    out = image.copy()
    h_court_to_px = np.linalg.inv(result.homography)
    for p, q in court_lines():
        (x1, y1), (x2, y2) = apply_homography(h_court_to_px, np.array([p, q]))
        if np.all(np.isfinite([x1, y1, x2, y2])):
            cv2.line(
                out,
                (round(x1), round(y1)),
                (round(x2), round(y2)),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    return out
