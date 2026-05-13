"""
Best-effort ~30 minute GPU training run on COCO val2017 (~5k images) + segment labels.

- First run downloads ~815 MB images + labels zip (extra wall time — see below).
- Training time is capped at ``--train-hours`` (default 0.5 = 30 minutes) using Ultralytics'
  ``time=`` argument (overrides epoch count once the limit is hit).

Example (from repo root):
  python scripts/train_run_30min.py
  python scripts/train_run_30min.py --model weights/yolov8s-seg.pt --batch 4 --imgsz 512
"""
from __future__ import annotations

import argparse
import time

from pathlib import Path

from ultralytics import YOLO

from prepare_coco_val5k import ensure_coco_val5k_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="~30 min capped YOLOv8-seg training on COCO val2017.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "weights" / "yolov8s-seg.pt",
        help="Starting checkpoint (pretrained). s balances quality vs speed on a laptop GPU.",
    )
    parser.add_argument(
        "--train-hours",
        type=float,
        default=0.5,
        help="Maximum training duration in hours (0.5 = 30 minutes GPU time).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=999,
        help="High ceiling; training usually stops on --train-hours first.",
    )
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (lower if CUDA OOM).")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "segment")
    parser.add_argument("--name", default="train30_coco_val5k")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 = Windows-safe).")
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Assume datasets/coco_val5k is already complete (images + labels + txts + yaml).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.skip_data_prep:
        from prepare_coco_val5k import YAML_OUT

        data_yaml = str(YAML_OUT.resolve())
        if not YAML_OUT.is_file():
            raise SystemExit(f"Missing {YAML_OUT}; run without --skip-data-prep first.")
    else:
        print("=== Preparing dataset (downloads on first run) ===")
        data_yaml = str(ensure_coco_val5k_dataset().resolve())
    prep_s = time.perf_counter() - t0
    print(f"Data prep finished in {prep_s / 60:.1f} min\n")

    print(
        f"=== Training (GPU cap {args.train_hours} h ~= {args.train_hours * 60:.0f} min) ===\n"
        f"data={data_yaml}\nmodel={args.model}\n"
    )
    model = YOLO(str(args.model.resolve()))
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        time=args.train_hours,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        workers=args.workers,
        exist_ok=True,
        patience=999,
        plots=True,
        verbose=True,
        cache=False,
    )
    print("Best weights:", model.trainer.best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
