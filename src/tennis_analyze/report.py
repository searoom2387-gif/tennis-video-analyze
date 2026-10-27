import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jinja2 import Environment, PackageLoader  # noqa: E402

from tennis_analyze.court_model import COURT_LENGTH_M, COURT_WIDTH_M, court_lines  # noqa: E402
from tennis_analyze.stats import PlayerStats, heatmap  # noqa: E402
from tennis_analyze.types import PlayerSample  # noqa: E402

PLAYERS = ("A", "B")


def _write_heatmap_png(samples: list[PlayerSample], player: str, path: Path) -> None:
    hist = heatmap(samples, player)  # rows: x (width), columns: y (length)
    shown = np.ma.masked_equal(hist.T, 0)  # rows: y, columns: x; empty cells stay transparent
    fig, ax = plt.subplots(figsize=(4, 7.5), dpi=100)
    try:
        ax.imshow(
            shown, origin="lower", extent=(0, COURT_WIDTH_M, 0, COURT_LENGTH_M),
            cmap="YlOrRd", vmin=0, vmax=max(float(hist.max()), 1.0), aspect="equal", interpolation="nearest",
        )
        for (x0, y0), (x1, y1) in court_lines():
            ax.plot([x0, x1], [y0, y1], color="0.25", linewidth=1.5)
        ax.set_xlim(0, COURT_WIDTH_M)
        ax.set_ylim(0, COURT_LENGTH_M)
        ax.axis("off")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def _write_pdf(out_dir: Path, html_path: Path, warnings: list[str]) -> Path | None:
    """Best effort: PDF is optional, so any failure becomes a warning, never an exception."""
    try:
        import weasyprint
    except ImportError:
        warnings.append("weasyprint が見つからないため PDF は作成されませんでした")
        return None
    except OSError as e:  # e.g. pango/gobject system libraries missing
        warnings.append(f"weasyprint を読み込めないため PDF は作成されませんでした({e})")
        return None
    pdf_path = out_dir / "report.pdf"
    try:
        weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    except Exception as e:
        pdf_path.unlink(missing_ok=True)
        warnings.append(f"PDF の作成に失敗しました({e})")
        return None
    if not pdf_path.exists():
        warnings.append("PDF の作成に失敗しました")
        return None
    return pdf_path


def write_outputs(
    out_dir: Path,
    samples: list[PlayerSample],
    stats: dict[str, PlayerStats],
    court_check: np.ndarray,
    warnings: list[str],
    pdf: bool = False,
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings = list(warnings)
    paths: list[Path] = []

    csv_path = out_dir / "positions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["time_s", "player", "x_m", "y_m"])
        for s in samples:
            writer.writerow([s.time_s, s.player, s.x_m, s.y_m])
    paths.append(csv_path)

    stats_path = out_dir / "stats.json"
    stats_path.write_text(
        json.dumps({p: asdict(st) for p, st in stats.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths.append(stats_path)

    heatmaps: list[str] = []
    for player in PLAYERS:
        if any(s.player == player for s in samples):
            png = out_dir / f"heatmap_{player}.png"
            _write_heatmap_png(samples, player, png)
            paths.append(png)
            heatmaps.append(player)

    check_path = out_dir / "court_check.png"
    cv2.imwrite(str(check_path), court_check)
    paths.append(check_path)

    html_path = out_dir / "report.html"
    # Rendered again after the PDF attempt if that added a warning.
    env = Environment(
        loader=PackageLoader("tennis_analyze", "templates"), autoescape=True
    )
    template = env.get_template("report.html.j2")

    def render() -> None:
        html_path.write_text(
            template.render(stats=stats, heatmaps=heatmaps, warnings=warnings),
            encoding="utf-8",
        )

    render()
    if pdf:
        before = len(warnings)
        pdf_path = _write_pdf(out_dir, html_path, warnings)
        if pdf_path is not None:
            paths.append(pdf_path)
        if len(warnings) != before:
            render()
    paths.append(html_path)
    return paths
