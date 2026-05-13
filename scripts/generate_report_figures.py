"""
Build publication-style figures from Ultralytics training logs (results.csv).

Run from repo root:
  python scripts/generate_report_figures.py
  python scripts/generate_report_figures.py --results runs/segment/finetune_lite_v2/results.csv --tag finetune_lite_v2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_results_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    keys = [k for k in rows[0].keys() if k is not None]
    out: dict[str, list[float]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            v = row.get(k, "")
            try:
                out[k].append(float(v) if v != "" else float("nan"))
            except ValueError:
                out[k].append(float("nan"))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def apply_report_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _epochs(d: dict[str, np.ndarray]) -> np.ndarray:
    return d["epoch"].astype(int)


def figure_losses(d: dict[str, np.ndarray], title: str, out: Path) -> None:
    ep = _epochs(d)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True)
    pairs = [
        ("train/box_loss", "val/box_loss", "Box loss"),
        ("train/seg_loss", "val/seg_loss", "Segmentation loss"),
        ("train/cls_loss", "val/cls_loss", "Classification loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL loss"),
    ]
    for ax, (tk, vk, name) in zip(axes.ravel(), pairs):
        if tk in d and vk in d:
            ax.plot(ep, d[tk], color="#1f77b4", linewidth=2, label="Train")
            ax.plot(ep, d[vk], color="#ff7f0e", linewidth=2, label="Validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight="600", y=1.02)
    fig.savefig(out)
    plt.close(fig)


def figure_detection_metrics(d: dict[str, np.ndarray], title: str, out: Path) -> None:
    ep = _epochs(d)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series = [
        ("metrics/precision(B)", "Precision (box)", "#1f77b4"),
        ("metrics/recall(B)", "Recall (box)", "#ff7f0e"),
        ("metrics/mAP50(B)", "mAP@0.5 (box)", "#2ca02c"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95 (box)", "#d62728"),
    ]
    for key, label, color in series:
        if key in d:
            ax.plot(ep, d[key], label=label, color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def figure_mask_metrics(d: dict[str, np.ndarray], title: str, out: Path) -> None:
    ep = _epochs(d)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series = [
        ("metrics/precision(M)", "Precision (mask)", "#9467bd"),
        ("metrics/recall(M)", "Recall (mask)", "#8c564b"),
        ("metrics/mAP50(M)", "mAP@0.5 (mask)", "#17becf"),
        ("metrics/mAP50-95(M)", "mAP@0.5:0.95 (mask)", "#bcbd22"),
    ]
    for key, label, color in series:
        if key in d:
            ax.plot(ep, d[key], label=label, color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.savefig(out)
    plt.close(fig)


def figure_lr(d: dict[str, np.ndarray], title: str, out: Path) -> None:
    ep = _epochs(d)
    key = "lr/pg0"
    if key not in d:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ep, d[key], color="#444444", linewidth=2, label="lr (pg0)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out)
    plt.close(fig)


def figure_dashboard(d: dict[str, np.ndarray], run_label: str, out: Path) -> None:
    ep = _epochs(d)
    fig = plt.figure(figsize=(12.5, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :])
    for key, lab, c in (
        ("train/box_loss", "train box", "#1f77b4"),
        ("val/box_loss", "val box", "#aec7e8"),
        ("train/seg_loss", "train seg", "#ff7f0e"),
        ("val/seg_loss", "val seg", "#ffbb78"),
    ):
        if key in d:
            ax0.plot(ep, d[key], label=lab, color=c, linewidth=1.8)
    ax0.set_title("Training dynamics: box and segmentation losses")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.legend(ncol=4, loc="upper right", fontsize=8)

    ax1 = fig.add_subplot(gs[1, 0])
    for key, lab, c in (
        ("metrics/mAP50(B)", "mAP50 box", "#2ca02c"),
        ("metrics/mAP50-95(B)", "mAP50-95 box", "#98df8a"),
    ):
        if key in d:
            ax1.plot(ep, d[key], label=lab, color=c, linewidth=2)
    ax1.set_title("Detection (bounding box) metrics")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("mAP")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right", fontsize=9)

    ax2 = fig.add_subplot(gs[1, 1])
    for key, lab, c in (
        ("metrics/mAP50(M)", "mAP50 mask", "#17becf"),
        ("metrics/mAP50-95(M)", "mAP50-95 mask", "#9edae5"),
    ):
        if key in d:
            ax2.plot(ep, d[key], label=lab, color=c, linewidth=2)
    ax2.set_title("Instance segmentation (mask) metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right", fontsize=9)

    ax3 = fig.add_subplot(gs[2, 0])
    if "lr/pg0" in d:
        ax3.plot(ep, d["lr/pg0"], color="#444444", linewidth=2)
    ax3.set_title("Learning rate schedule")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("LR")

    ax4 = fig.add_subplot(gs[2, 1])
    if "time" in d:
        t = d["time"] / 3600.0
        ax4.plot(ep, t, color="#7f7f7f", linewidth=2)
        ax4.set_title("Cumulative training time")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Hours")

    fig.suptitle(f"YOLOv8-seg training summary — {run_label}", fontsize=15, fontweight="600", y=0.995)
    fig.savefig(out)
    plt.close(fig)


def figure_qualitative_pair(orig: Path, pred: Path, title: str, out: Path) -> None:
    if not orig.is_file() or not pred.is_file():
        return
    im0 = plt.imread(str(orig))
    im1 = plt.imread(str(pred))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].imshow(im0)
    axes[0].set_title("Input image", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(im1)
    axes[1].set_title("Detection + instance segmentation (model output)", fontsize=12)
    axes[1].axis("off")
    fig.suptitle(title, fontsize=14, fontweight="600")
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate report figures from Ultralytics results.csv.")
    parser.add_argument(
        "--results",
        type=Path,
        nargs="*",
        default=[
            ROOT / "runs" / "segment" / "train30_coco_val5k" / "results.csv",
            ROOT / "runs" / "segment" / "finetune_lite_v2" / "results.csv",
        ],
        help="One or more results.csv paths.",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports", help="Directory for PNG exports.")
    parser.add_argument(
        "--qual-input",
        type=Path,
        default=ROOT / "data" / "samples" / "bus.jpg",
        help="Image for qualitative before/after (if missing, skip).",
    )
    parser.add_argument(
        "--qual-output",
        type=Path,
        default=ROOT / "outputs" / "seg_bus.jpg",
        help="Matching model visualization for qualitative figure.",
    )
    parser.add_argument(
        "--qual-name",
        default="bus_example",
        help="Base name for qualitative PNG in reports/ (report_qualitative_<name>.png).",
    )
    parser.add_argument(
        "--only-qualitative",
        action="store_true",
        help="Only render the qualitative before/after figure (no results.csv plots).",
    )
    args = parser.parse_args()
    apply_report_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if args.only_qualitative:
        pq = args.out_dir / f"report_qualitative_{args.qual_name}.png"
        figure_qualitative_pair(
            args.qual_input,
            args.qual_output,
            "Qualitative example: object detection + segmentation",
            pq,
        )
        if pq.is_file():
            written.append(pq)
        for p in written:
            print("Wrote", p)
        if not written:
            print("Qualitative figure not written; check --qual-input and --qual-output exist.")
            return 1
        print("Done.")
        return 0

    for csv_path in args.results:
        if not csv_path.is_file():
            print("Skip (missing):", csv_path)
            continue
        tag = csv_path.parent.name
        d = read_results_csv(csv_path)
        run_title = f"{tag} (Ultralytics YOLOv8-seg)"

        p1 = args.out_dir / f"report_losses_{tag}.png"
        figure_losses(d, f"Train vs validation losses — {run_title}", p1)
        written.append(p1)

        p2 = args.out_dir / f"report_metrics_detection_{tag}.png"
        figure_detection_metrics(d, f"Bounding-box metrics vs epoch — {run_title}", p2)
        written.append(p2)

        p3 = args.out_dir / f"report_metrics_mask_{tag}.png"
        figure_mask_metrics(d, f"Mask (instance segmentation) metrics vs epoch — {run_title}", p3)
        written.append(p3)

        p4 = args.out_dir / f"report_lr_{tag}.png"
        if "lr/pg0" in d:
            figure_lr(d, f"Learning rate — {run_title}", p4)
            written.append(p4)

        p5 = args.out_dir / f"report_dashboard_{tag}.png"
        figure_dashboard(d, run_title, p5)
        written.append(p5)

    pq = args.out_dir / f"report_qualitative_{args.qual_name}.png"
    figure_qualitative_pair(
        args.qual_input,
        args.qual_output,
        "Qualitative example: object detection + segmentation",
        pq,
    )
    if pq.is_file():
        written.append(pq)

    for p in written:
        print("Wrote", p)
    if not written:
        print("No figures written. Train a model first or pass --results paths to results.csv.")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
