# 選手位置分析(フェーズ1)実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 固定カメラのシングルス試合映像(約60分)から、選手2人のコート上の位置を取得し、軌跡・ヒートマップ・走行距離をデータと HTML レポートで出力する。

**Architecture:** 映像読み込み → コート検出(ホモグラフィ)→ 選手検出・追跡 → スタッツ計算 → レポート生成の5部品を、型付きデータだけでつなぐ。検出器は Protocol の裏に置き、テストでは偽の検出器を差し込む。

**Tech Stack:** Python 3.11、numpy、opencv-python-headless、torch/torchvision、supervision(ByteTrack)、matplotlib、jinja2、pytest。

**Spec:** `docs/superpowers/specs/2026-10-27-player-position-analysis-design.md`

## Global Constraints

- シングルスコート寸法: 幅 8.23m、長さ 23.77m(コート座標は x=幅方向 0〜8.23、y=長さ方向 0〜23.77、メートル単位)。
- 映像は 5fps に間引いて処理する(60分で約18,000フレーム)。
- 実行環境は CPU のみ。GPU 前提のコードを書かない。
- ショット検出・ボール追跡・ダブルス・姿勢推定・リアルタイム処理は扱わない。
- コート検出は学習済みモデルを使わず、従来の画像処理(ライン抽出 + ハフ変換 + RANSAC)で行う。
- コート検出の信頼度が低い場合は停止して警告する。選手が2人見つからない場合は警告して出力を続ける。
- 出力ファイル名: `positions.csv`、`stats.json`、`heatmap_A.png`、`heatmap_B.png`、`court_check.png`、`report.html`。

## Review Focus

1. 選手のサイドチェンジ(コートチェンジ)後も、ラベル A/B が同じ人物に付いたままである。→ Task 4
2. ダブルスのライン付きコートでシングルスを撮った映像で、外側のダブルスラインに引きずられず、シングルスのラインに合わせる。→ Task 3
3. 検出された人物が0人、または1人だけのとき、例外で落ちず、警告付きで出力する。→ Task 4、Task 7
4. 空の動画、フレーム数0、fps メタデータが 0 の動画で、原因の分かるエラーで終了する。→ Task 2
5. 追跡のノイズで静止している選手の走行距離が膨らまない。→ Task 5

---

## ファイル構成

| ファイル | 責務 |
|---|---|
| `pyproject.toml` | 依存関係、`tennis-analyze` コマンド登録 |
| `src/tennis_analyze/types.py` | 部品間で渡すデータ型 |
| `src/tennis_analyze/court_model.py` | シングルスコートの寸法とライン定義 |
| `src/tennis_analyze/geometry.py` | ホモグラフィ適用、足元位置の算出 |
| `src/tennis_analyze/video_io.py` | 動画の読み込みと間引き |
| `src/tennis_analyze/court_detect.py` | コート検出、信頼度、確認用画像、手動指定 |
| `src/tennis_analyze/players.py` | 人物検出、追跡、つなぎ直し、選手の絞り込み |
| `src/tennis_analyze/stats.py` | 走行距離、速度、滞在割合、ヒートマップ |
| `src/tennis_analyze/report.py` | 出力ファイルの書き出し |
| `src/tennis_analyze/pipeline.py` | 部品をつなぐ実行関数 |
| `src/tennis_analyze/cli.py` | コマンドライン入口 |
| `tests/synthetic.py` | 合成コート画像・合成動画の生成(テスト用) |

---

### Task 1: プロジェクトの土台、型、コートモデル、座標変換

**Files:**
- Create: `pyproject.toml`, `src/tennis_analyze/__init__.py`, `src/tennis_analyze/types.py`, `src/tennis_analyze/court_model.py`, `src/tennis_analyze/geometry.py`
- Test: `tests/test_court_model.py`, `tests/test_geometry.py`

**Interfaces:**
- Produces(`types.py`、すべて `@dataclass(frozen=True)`):
  - `Frame(index: int, time_s: float, image: np.ndarray)`
  - `Detection(box_xyxy: tuple[float, float, float, float], score: float)`
  - `TrackedBox(track_id: int, box_xyxy: tuple[float, float, float, float])`
  - `TrackSample(time_s: float, track_id: int, x_m: float, y_m: float)`
  - `PlayerSample(time_s: float, player: str, x_m: float, y_m: float)`(`player` は `"A"` か `"B"`)
  - `CourtResult(homography: np.ndarray, confidence: float)`(3x3、ピクセルからコート座標メートルへ)
- Produces(`court_model.py`): 定数 `COURT_WIDTH_M = 8.23`、`COURT_LENGTH_M = 23.77`、`SERVICE_LINE_FROM_NET_M = 6.40`。`court_lines() -> list[tuple[tuple[float, float], tuple[float, float]]]`(ベースライン2本、シングルスサイドライン2本、サービスライン2本、センターサービスライン1本の、計7本の線分、コート座標)。
- Produces(`geometry.py`): `apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray`((N,2) から (N,2))、`foot_point(box_xyxy) -> tuple[float, float]`(ボックス下辺の中央)。

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_court_model.py
def test_court_has_seven_lines_with_spec_dimensions():
    lines = court_lines()
    assert len(lines) == 7
    xs = {p[0] for line in lines for p in line}
    ys = {p[1] for line in lines for p in line}
    assert min(xs) == 0 and max(xs) == 8.23
    assert min(ys) == 0 and max(ys) == 23.77

def test_service_lines_are_6_40m_from_net():
    ys = {p[1] for line in court_lines() for p in line}
    assert pytest.approx(23.77 / 2 - 6.40) in ys   # 5.485
    assert pytest.approx(23.77 / 2 + 6.40) in ys   # 18.285

# tests/test_geometry.py
def test_apply_homography_identity_and_scale():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(apply_homography(np.eye(3), pts), pts)
    assert np.allclose(apply_homography(np.diag([2.0, 2.0, 1.0]), pts), pts * 2)

def test_foot_point_is_bottom_center_of_box():
    assert foot_point((10.0, 20.0, 30.0, 80.0)) == (20.0, 80.0)
```

- [ ] **Step 2: 失敗を確認する**

Run: `pytest tests/test_court_model.py tests/test_geometry.py -v`
Expected: FAIL(`court_lines` などが未定義)

- [ ] **Step 3: 実装する**

`pyproject.toml` に上記依存と `[project.scripts] tennis-analyze = "tennis_analyze.cli:main"` を書く。`types.py`、`court_model.py`、`geometry.py` を上記の署名で実装する。`apply_homography` は同次座標で変換して w で割る。

- [ ] **Step 4: 通過を確認する**

Run: `pytest tests/test_court_model.py tests/test_geometry.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add pyproject.toml src tests
git commit -m "feat: scaffold project with court model and geometry helpers"
```

---

### Task 2: 映像読み込み

**Files:**
- Create: `src/tennis_analyze/video_io.py`
- Test: `tests/test_video_io.py`

**Interfaces:**
- Consumes: `Frame`(Task 1)
- Produces: `class VideoOpenError(Exception)`、`iter_frames(path: Path, target_fps: float = 5.0) -> Iterator[Frame]`。`Frame.time_s` は元動画の時刻(秒)。

- [ ] **Step 1: 失敗するテストを書く**

```python
# 30fps・3秒の動画(tmp_path に cv2.VideoWriter で生成)を使う
def test_iter_frames_downsamples_to_target_fps(video_30fps_3s):
    frames = list(iter_frames(video_30fps_3s, target_fps=5.0))
    assert len(frames) == 15
    assert frames[1].time_s == pytest.approx(0.2, abs=0.04)

def test_missing_file_raises_video_open_error(tmp_path):
    with pytest.raises(VideoOpenError, match="開けません"):
        list(iter_frames(tmp_path / "none.mp4"))

def test_empty_video_raises_video_open_error(empty_video):   # 0フレームの動画
    with pytest.raises(VideoOpenError, match="フレーム"):
        list(iter_frames(empty_video))

def test_zero_fps_metadata_raises_video_open_error(monkeypatch, video_30fps_3s):
    # cv2.VideoCapture.get(CAP_PROP_FPS) が 0 を返すよう差し替える
    with pytest.raises(VideoOpenError, match="fps"):
        list(iter_frames(video_30fps_3s))
```

- [ ] **Step 2: 失敗を確認する**

Run: `pytest tests/test_video_io.py -v`
Expected: FAIL

- [ ] **Step 3: `iter_frames` を実装する**

`cv2.VideoCapture` で開く。fps が 0 以下ならエラー。`round(src_fps / target_fps)` フレームごとに1枚返す。最初のフレームが読めなければ「フレームがありません」のエラーにする。エラーメッセージは日本語で、上のテストの `match` 文字列を含める。

- [ ] **Step 4: 通過を確認する**

Run: `pytest tests/test_video_io.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/tennis_analyze/video_io.py tests/test_video_io.py
git commit -m "feat: add frame-sampling video reader with clear errors"
```

---

### Task 3: コート検出

**Files:**
- Create: `src/tennis_analyze/court_detect.py`, `tests/synthetic.py`
- Test: `tests/test_court_detect.py`

**Interfaces:**
- Consumes: `CourtResult`(Task 1)、`court_lines`、`apply_homography`
- Produces(`court_detect.py`):
  - `class CourtDetectionError(Exception)`
  - `MIN_CONFIDENCE = 0.5`(仮の初期値。Task 8 で実映像により決める)
  - `extract_line_mask(image: np.ndarray) -> np.ndarray`(uint8、ライン=255)
  - `score_homography(h: np.ndarray, mask: np.ndarray) -> float`(コートモデルの各ラインを点列でサンプルし、画像へ射影して、マスク上に乗る点の割合を 0〜1 で返す)
  - `detect_court_frame(image: np.ndarray) -> CourtResult | None`
  - `detect_court(images: Sequence[np.ndarray], min_confidence: float = MIN_CONFIDENCE) -> CourtResult`(各フレームの結果から、ホモグラフィが近いもの同士の多数派を採る。多数派がなければ、または信頼度が下回れば `CourtDetectionError`)
  - `court_from_corners(corners_px: Sequence[tuple[float, float]]) -> CourtResult`(手前左、手前右、奥右、奥左の順。信頼度は 1.0)
  - `render_court_check(image: np.ndarray, result: CourtResult) -> np.ndarray`(ラインを重ねた画像)
- Produces(`tests/synthetic.py`): `make_homography(seed: int) -> np.ndarray`(コート座標からピクセルへの、後方やや高い位置から見た妥当な射影、1920x1080)、`render_court(h_court_to_px: np.ndarray, doubles_lines: bool = False, size=(1080, 1920)) -> np.ndarray`(緑地に白線のBGR画像。`doubles_lines=True` でシングルスの外側 1.37m にダブルスのサイドラインを加える)。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_recovers_homography_from_synthetic_court():
    h_true = make_homography(seed=0)                  # コート座標 -> px
    img = render_court(h_true)
    result = detect_court([img] * 5)
    # 検出結果は px -> コート座標。四隅を往復して誤差を見る
    corners_m = np.array([[0, 0], [8.23, 0], [8.23, 23.77], [0, 23.77]])
    px = apply_homography(h_true, corners_m)
    back = apply_homography(result.homography, px)
    assert np.abs(back - corners_m).max() < 0.15      # 15cm以内
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
```

- [ ] **Step 2: 失敗を確認する**

Run: `pytest tests/test_court_detect.py -v`
Expected: FAIL

- [ ] **Step 3: `tests/synthetic.py` を実装する**

署名どおり。`make_homography` は seed で四隅の位置を少し揺らす。`render_court` は `cv2.line` で `court_lines()` を射影して描く。

- [ ] **Step 4: `extract_line_mask`、`score_homography` を実装する**

マスクは、明るさの高い低彩度の画素を抽出する(色と明るさのしきい値)。`score_homography` は、ライン上の点を 0.1m 間隔でサンプルし、マスクを 2px 膨らませた上に乗る割合を返す。

- [ ] **Step 5: `detect_court_frame` を実装する**

ハフ変換(確率的)で線分を取り、傾きで横系・縦系に分ける。横系2本と縦系2本の組み合わせを、それぞれコートモデルの横ライン2本と縦ライン2本に割り当てた候補ごとに、4交点の対応からホモグラフィを作り、`score_homography` が最大のものを RANSAC 的に選ぶ。候補が無ければ `None`。シングルス幅のモデルに当てはめるため、外側のダブルスラインはスコアが低くなって負ける(Step 1 のダブルス テストで確認)。

- [ ] **Step 6: `detect_court`、`court_from_corners`、`render_court_check` を実装する**

多数決は、各フレームのホモグラフィでコート四隅を射影した結果が 20px 以内で一致するもの同士をグループにし、過半数のグループを採る。過半数が無ければ「フレーム間で検出結果が一致しません」で `CourtDetectionError`。多数派の信頼度の中央値が `min_confidence` 未満なら信頼度を含めて `CourtDetectionError`。

- [ ] **Step 7: 通過を確認する**

Run: `pytest tests/test_court_detect.py -v`
Expected: PASS

- [ ] **Step 8: コミット**

```bash
git add src/tennis_analyze/court_detect.py tests/synthetic.py tests/test_court_detect.py
git commit -m "feat: add line-based court detection with voting and confidence gate"
```

---

### Task 4: 選手検出・追跡・絞り込み

**Files:**
- Create: `src/tennis_analyze/players.py`
- Test: `tests/test_players.py`

**Interfaces:**
- Consumes: `Detection`、`TrackedBox`、`TrackSample`、`PlayerSample`(Task 1)、`apply_homography`、`foot_point`
- Produces:
  - `class PersonDetector(Protocol)`: `detect(self, image: np.ndarray) -> list[Detection]`
  - `class TorchvisionPersonDetector`: `PersonDetector` を実装。torchvision の `ssdlite320_mobilenet_v3_large`(BSD ライセンス、CPU 向け)の人物クラスを使う。
  - `class PlayerTracker`: `update(self, detections: list[Detection]) -> list[TrackedBox]`。supervision の ByteTrack を包む。
  - `to_track_samples(time_s: float, boxes: list[TrackedBox], h: np.ndarray) -> list[TrackSample]`(足元をコート座標へ)
  - `stitch_tracks(samples: list[TrackSample], max_gap_s: float = 2.0, max_dist_m: float = 3.0) -> list[TrackSample]`(切れた追跡IDを、時間差と位置の近さで同一IDにつなぎ直す)
  - `select_players(samples: list[TrackSample], court_margin_m: float = 2.0) -> tuple[list[PlayerSample], list[str]]`(戻り値の2つ目は警告メッセージ。`court_margin_m` はコート外周から広げる幅)

**ライセンスの判断:** 設計書は YOLO nano 系としていたが、ultralytics の YOLO は AGPL で、本プロジェクトのライセンス方針が未定のため、既定は BSD の torchvision モデルにする。検出器は `PersonDetector` の裏にあるので、後で差し替えられる。この判断はレビュー時にユーザーへ伝える。

- [ ] **Step 1: 失敗するテストを書く**

```python
def _walk(track_id, t0, t1, xy0, xy1, n=20):
    ts = np.linspace(t0, t1, n)
    return [TrackSample(t, track_id, *np.interp(t, [t0, t1], [xy0[i], xy1[i]]) ...) for t in ts]
    # x と y をそれぞれ線形補間して TrackSample を作る(実装は自由)

def test_select_players_keeps_two_longest_tracks_inside_court():
    players = _walk(1, 0, 60, (2, 3), (3, 4)) + _walk(2, 0, 60, (5, 20), (4, 19))
    ball_person = _walk(3, 0, 5, (-8.0, 10.0), (-8.0, 11.0))   # コート外 8m
    out, warns = select_players(players + ball_person)
    assert {s.player for s in out} == {"A", "B"}
    assert warns == []

def test_player_a_starts_on_near_side_y_small():
    out, _ = select_players(_walk(1, 0, 60, (2, 3), (3, 4)) + _walk(2, 0, 60, (5, 20), (4, 19)))
    first_a = next(s for s in out if s.player == "A")
    assert first_a.y_m < 23.77 / 2

def test_labels_follow_person_after_change_of_ends():
    # 選手1: 近側 -> 前半30秒で近側、後半30秒で奥側へ移動(サイドチェンジ)
    p1 = _walk(1, 0, 30, (4, 3), (4, 5)) + _walk(1, 30, 60, (4, 20), (4, 19))
    p2 = _walk(2, 0, 30, (4, 20), (4, 19)) + _walk(2, 30, 60, (4, 3), (4, 5))
    out, _ = select_players(p1 + p2)
    a_late = [s for s in out if s.player == "A" and s.time_s > 35]
    assert all(s.y_m > 23.77 / 2 for s in a_late)       # A は奥側へ移った人物のまま

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
```

- [ ] **Step 2: 失敗を確認する**

Run: `pytest tests/test_players.py -v`
Expected: FAIL

- [ ] **Step 3: `stitch_tracks`、`to_track_samples`、`select_players` を実装する**

`select_players` は、`court_margin_m` を広げたコート範囲内のサンプルだけ数え、追跡IDごとのサンプル数の上位2つを選手とする。A/B の割り当ては、全体の最初の時刻での y 座標が小さい方(近い側)を A とし、以後は追跡ID(人物)に固定する(側ではなく人物でラベルを保つ)。2人未満なら警告メッセージを返す。

- [ ] **Step 4: `PlayerTracker` と `TorchvisionPersonDetector` を実装する**

検出器は人物クラスかつスコア 0.5 以上だけ返す。追跡は supervision の ByteTrack に委ねる(API は導入バージョンで確認する)。この2つは実モデルが必要なため自動テストに含めず、Task 8 で実映像により確認する。

- [ ] **Step 5: 通過を確認する**

Run: `pytest tests/test_players.py -v`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add src/tennis_analyze/players.py tests/test_players.py
git commit -m "feat: add player tracking, stitching, and selection"
```

---

### Task 5: スタッツ計算

**Files:**
- Create: `src/tennis_analyze/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `PlayerSample`(Task 1)、`COURT_WIDTH_M`、`COURT_LENGTH_M`
- Produces:
  - `@dataclass(frozen=True) PlayerStats(distance_m: float, mean_speed_mps: float, share_left: float, share_right: float, share_front: float, share_back: float)`(前=手前半分 y<11.885、左=x<4.115)
  - `compute_stats(samples: list[PlayerSample]) -> dict[str, PlayerStats]`(キーは `"A"`、`"B"`。サンプルが無い選手はキーに含めない)
  - `heatmap(samples: list[PlayerSample], player: str, bins: tuple[int, int] = (24, 48)) -> np.ndarray`((幅方向, 長さ方向) の滞在秒数ではなくサンプル数、コート外は除く)
  - `smooth_path(xy: np.ndarray, window: int = 5) -> np.ndarray`(移動中央値でノイズを除く)

- [ ] **Step 1: 失敗するテストを書く**

```python
def _samples(player, pts, dt=0.2):
    return [PlayerSample(i * dt, player, x, y) for i, (x, y) in enumerate(pts)]

def test_straight_walk_distance_and_speed():
    pts = [(4.0, float(i) * 0.5) for i in range(21)]        # 0.5m刻みで10m、20ステップ
    s = compute_stats(_samples("A", pts))["A"]
    assert s.distance_m == pytest.approx(10.0, abs=0.3)
    assert s.mean_speed_mps == pytest.approx(2.5, abs=0.2)  # 0.5m / 0.2s

def test_stationary_player_with_tracking_jitter_has_near_zero_distance():
    rng = np.random.default_rng(0)
    pts = [(4 + rng.normal(0, 0.05), 5 + rng.normal(0, 0.05)) for _ in range(300)]
    s = compute_stats(_samples("A", pts))["A"]
    assert s.distance_m < 3.0       # 生のままだと 300 * 約0.09m = 約27m になる

def test_zone_shares_sum_to_one_per_axis():
    pts = [(1.0, 2.0)] * 3 + [(7.0, 20.0)]
    s = compute_stats(_samples("A", pts))["A"]
    assert s.share_left + s.share_right == pytest.approx(1.0)
    assert s.share_front + s.share_back == pytest.approx(1.0)
    assert s.share_left == pytest.approx(0.75) and s.share_front == pytest.approx(0.75)

def test_player_without_samples_is_absent():
    assert "B" not in compute_stats(_samples("A", [(1.0, 1.0), (1.0, 2.0)]))

def test_heatmap_counts_samples_and_ignores_outside_court():
    pts = [(1.0, 1.0)] * 4 + [(-3.0, 5.0)]
    hm = heatmap(_samples("A", pts), "A")
    assert hm.shape == (24, 48) and hm.sum() == 4
```

- [ ] **Step 2: 失敗を確認する**

Run: `pytest tests/test_stats.py -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

`smooth_path` は x・y それぞれに移動中央値をかける。`compute_stats` は平滑化した経路の隣接点間距離を合計して走行距離とし、平均速度は距離を最初から最後までの時間で割る。サンプルが1点だけなら距離0・速度0。

- [ ] **Step 4: 通過を確認する**

Run: `pytest tests/test_stats.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/tennis_analyze/stats.py tests/test_stats.py
git commit -m "feat: add position stats with jitter-robust distance"
```

---

### Task 6: レポート生成

**Files:**
- Create: `src/tennis_analyze/report.py`, `src/tennis_analyze/templates/report.html.j2`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `PlayerSample`、`PlayerStats`、`heatmap`(Task 5)、`COURT_WIDTH_M`、`COURT_LENGTH_M`
- Produces: `write_outputs(out_dir: Path, samples: list[PlayerSample], stats: dict[str, PlayerStats], court_check: np.ndarray, warnings: list[str], pdf: bool = False) -> list[Path]`(作ったファイルのパスを返す)

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_write_outputs_creates_all_files(tmp_path):
    samples = [PlayerSample(0.0, "A", 1.0, 2.0), PlayerSample(0.2, "A", 1.5, 2.5),
               PlayerSample(0.0, "B", 5.0, 20.0), PlayerSample(0.2, "B", 5.5, 20.5)]
    stats = compute_stats(samples)
    paths = write_outputs(tmp_path, samples, stats, np.zeros((10, 10, 3), np.uint8), [])
    names = {p.name for p in paths}
    assert names == {"positions.csv", "stats.json", "heatmap_A.png", "heatmap_B.png",
                     "court_check.png", "report.html"}

def test_positions_csv_has_header_and_one_row_per_sample(tmp_path):
    ...  # 先頭行が "time_s,player,x_m,y_m"、行数が サンプル数 + 1

def test_report_html_shows_warnings(tmp_path):
    write_outputs(tmp_path, samples, stats, img, ["選手が1人しか見つかりません"])
    assert "選手が1人しか見つかりません" in (tmp_path / "report.html").read_text(encoding="utf-8")

def test_missing_player_skips_that_heatmap(tmp_path):
    samples_a_only = [PlayerSample(0.0, "A", 1.0, 2.0)]
    paths = write_outputs(tmp_path, samples_a_only, compute_stats(samples_a_only), img, ["w"])
    assert "heatmap_B.png" not in {p.name for p in paths}
```

- [ ] **Step 2: 失敗を確認する**

Run: `pytest tests/test_report.py -v`
Expected: FAIL

- [ ] **Step 3: 実装する**

ヒートマップ画像は matplotlib でコート線を描いた上に `heatmap` の結果を重ねる。HTML は Jinja2 のテンプレートで、スタッツ表・ヒートマップ・確認用画像・警告を載せる(画像は相対パスで参照)。`pdf=True` のときだけ weasyprint で `report.pdf` を作る(インストールされていなければ警告を `warnings` に足し、処理は続ける)。

- [ ] **Step 4: 通過を確認する**

Run: `pytest tests/test_report.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add src/tennis_analyze/report.py src/tennis_analyze/templates tests/test_report.py
git commit -m "feat: add report and data file writer"
```

---

### Task 7: パイプラインとコマンドライン

**Files:**
- Create: `src/tennis_analyze/pipeline.py`, `src/tennis_analyze/cli.py`
- Modify: `tests/synthetic.py`(合成動画の関数を追加)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1〜6 のすべて
- Produces:
  - `run(video: Path, out_dir: Path, detector: PersonDetector, corners: Sequence[tuple[float, float]] | None = None, target_fps: float = 5.0, pdf: bool = False) -> list[str]`(戻り値は警告メッセージ。`corners` を渡すとコート自動検出の代わりに手動指定を使う)
  - `main(argv: Sequence[str] | None = None) -> int`: `tennis-analyze VIDEO --out DIR [--corners x1,y1,x2,y2,x3,y3,x4,y4] [--pdf]`。コート検出失敗時は、確認用画像を `DIR/court_check.png` に書いて終了コード 2 を返す。
  - `tests/synthetic.py` に追加: `write_synthetic_match(path: Path, seconds: int = 6, fps: int = 30) -> list[Detection-by-frame]`(合成コート上に、2つの白い四角を動かして描いた動画を書き、各フレームの正解ボックスを返す)、`class FakeDetector`(`PersonDetector` を実装し、`write_synthetic_match` が返した正解ボックスを順に返す)。

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_end_to_end_on_synthetic_match_writes_all_outputs(tmp_path):
    video = tmp_path / "m.mp4"
    boxes = write_synthetic_match(video)
    warnings = run(video, tmp_path / "out", FakeDetector(boxes))
    assert (tmp_path / "out" / "report.html").exists()
    stats = json.loads((tmp_path / "out" / "stats.json").read_text())
    assert set(stats) == {"A", "B"} and stats["A"]["distance_m"] > 0
    assert warnings == []

def test_blank_court_exits_with_code_2_and_writes_check_image(tmp_path):
    video = tmp_path / "blank.mp4"
    write_blank_video(video)            # 黒一色の動画を書く補助関数
    code = main([str(video), "--out", str(tmp_path / "out")])
    assert code == 2

def test_missing_video_exits_nonzero_with_message(tmp_path, capsys):
    code = main([str(tmp_path / "none.mp4"), "--out", str(tmp_path / "out")])
    assert code != 0 and "開けません" in capsys.readouterr().err

def test_no_players_still_writes_report_with_warning(tmp_path):
    video = tmp_path / "m.mp4"
    write_synthetic_match(video)
    warnings = run(video, tmp_path / "out", FakeDetector([]))
    assert any("見つかりません" in w for w in warnings)
    assert (tmp_path / "out" / "report.html").exists()
```

- [ ] **Step 2: 失敗を確認する**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: 合成動画の補助関数を実装する**

`write_synthetic_match` は `render_court` の背景に、2つの四角をコート座標で動かして描く(射影して画像に描画)。`write_blank_video` と `FakeDetector` も `tests/synthetic.py` に置く。

- [ ] **Step 4: `run` と `main` を実装する**

`run` は、先頭から等間隔に 30 フレームを選んで `detect_court` に渡す(`corners` 指定時は `court_from_corners`)。続いて全フレームについて検出 → 追跡 → `to_track_samples` を行い、`stitch_tracks` と `select_players` を通して `compute_stats` し、`write_outputs` で書く。`main` は `VideoOpenError` と `CourtDetectionError` を捕まえ、メッセージを標準エラーに出す。

- [ ] **Step 5: 通過を確認する**

Run: `pytest -v`
Expected: すべて PASS

- [ ] **Step 6: コミット**

```bash
git add src/tennis_analyze/pipeline.py src/tennis_analyze/cli.py tests/synthetic.py tests/test_pipeline.py
git commit -m "feat: add end-to-end pipeline and CLI"
```

---

### Task 8: 実映像での確認と調整(手動)

**Files:**
- Modify: `src/tennis_analyze/court_detect.py`(`MIN_CONFIDENCE` ほか、しきい値の調整のみ)
- Create: `docs/real-video-check.md`(結果の記録)

**Interfaces:**
- Consumes: Task 7 の `main`

この Task は自動テストの対象外で、ユーザーから試合映像を1本借りて行う。映像が無い間は、ここで止めてユーザーに依頼する。

- [ ] **Step 1: 映像を受け取り、解像度・長さ・fps を記録する**

`ffprobe` の出力を `docs/real-video-check.md` に書く。

- [ ] **Step 2: 実行して処理時間を測る**

Run: `time tennis-analyze <映像> --out out_real`
Expected: 完了する。実測時間を記録する(設計書の概算 15〜45分と比べる)。

- [ ] **Step 3: `court_check.png` を目で確認する**

重ねたラインがコートのラインに合っているか。ずれる場合は、`MIN_CONFIDENCE`、マスクのしきい値、ハフ変換のパラメータを調整し、合成映像のテスト(Task 3)が通り続けることも確認する。

- [ ] **Step 4: 選手の絞り込みを確認する**

`positions.csv` の軌跡が、実際の選手の動きと合うか。ボールパーソンや観客が選手になっていないか。コートチェンジ後にラベルが保たれているか。問題があれば `court_margin_m` などを調整する。

- [ ] **Step 5: 結果を記録してコミット**

```bash
git add docs/real-video-check.md src
git commit -m "chore: tune thresholds on real match video"
```

---

## 自己レビュー

- **設計書との対応:** 映像読み込み(Task 2)、コート検出と信頼度・確認画像・手動指定(Task 3)、選手検出・追跡・絞り込み(Task 4)、スタッツ(Task 5)、出力ファイル6種とPDF任意(Task 6)、エラー処理と終了動作(Task 7)、実映像確認としきい値決定(Task 8)。抜けなし。
- **型の一貫性:** `CourtResult.homography` はピクセルからコート座標。`apply_homography`、`court_from_corners`、`to_track_samples` で同じ向きを使う。`PlayerSample.player` は `"A"`/`"B"` で、`compute_stats`・`heatmap`・`write_outputs` が同じキーを使う。
- **設計書との差分:** 検出モデルを YOLO nano ではなく torchvision の `ssdlite320_mobilenet_v3_large` にした(AGPL 回避)。`PersonDetector` の裏にあるため差し替え可能。
