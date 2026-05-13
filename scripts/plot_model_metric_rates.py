"""
Plot validation metric rates (precision, recall, mAP) from Ultralytics results.csv
and save PNGs under reports/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"
RUNS = REPO_ROOT / "runs"


def _read_results_csv(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows: list[dict[str, float | int]] = []
        for row in r:
            parsed: dict[str, float | int] = {}
            for k, v in row.items():
                if k is None or v is None or v == "":
                    continue
                if k == "epoch":
                    parsed[k] = int(float(v))
                else:
                    parsed[k] = float(v)
            rows.append(parsed)
    return rows


def _plot_rates(
    epochs: list[int],
    series: list[tuple[str, list[float], str]],
    title: str,
    outfile: Path,
) -> None:
    plt.figure(figsize=(9, 5.5))
    for label, values, color in series:
        plt.plot(epochs, values, label=label, color=color, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Rate (0–1)")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_run(results_csv: Path) -> list[Path]:
    rows = _read_results_csv(results_csv)
    if not rows:
        return []

    # e.g. runs/segment/train30_coco_val5k/results.csv -> train30_coco_val5k
    run_slug = results_csv.parent.name
    epochs = [int(r["epoch"]) for r in rows]

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows]

    written: list[Path] = []

    box_series = [
        ("Precision (box)", col("metrics/precision(B)"), "#1f77b4"),
        ("Recall (box)", col("metrics/recall(B)"), "#ff7f0e"),
        ("mAP50 (box)", col("metrics/mAP50(B)"), "#2ca02c"),
        ("mAP50-95 (box)", col("metrics/mAP50-95(B)"), "#d62728"),
    ]
    out_box = REPORTS / f"{run_slug}_box_metric_rates.png"
    _plot_rates(epochs, box_series, f"Validation metric rates — box ({run_slug})", out_box)
    written.append(out_box)

    mask_series = [
        ("Precision (mask)", col("metrics/precision(M)"), "#1f77b4"),
        ("Recall (mask)", col("metrics/recall(M)"), "#ff7f0e"),
        ("mAP50 (mask)", col("metrics/mAP50(M)"), "#2ca02c"),
        ("mAP50-95 (mask)", col("metrics/mAP50-95(M)"), "#d62728"),
    ]
    out_mask = REPORTS / f"{run_slug}_mask_metric_rates.png"
    _plot_rates(epochs, mask_series, f"Validation metric rates — mask ({run_slug})", out_mask)
    written.append(out_mask)

    return written


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(RUNS.rglob("results.csv"))
    if not csv_files:
        print(f"No results.csv found under {RUNS}")
        return
    all_written: list[Path] = []
    for p in csv_files:
        try:
            all_written.extend(plot_run(p))
        except KeyError as e:
            print(f"Skip {p}: missing column {e}")
    for w in all_written:
        print(w)


if __name__ == "__main__":
    main()
