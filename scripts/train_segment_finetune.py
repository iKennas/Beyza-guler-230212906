"""
Short fine-tune of YOLOv8-seg on coco128-seg (128 COCO images, ~7 MB).

By default this script keeps data under this project folder (datasets/coco128-seg/)
and writes datasets/coco128_bayza.yaml with an absolute path so training does not
depend on Ultralytics' global datasets_dir (which can be wrong on some Windows setups).

Weight updates apply to all 80 COCO classes present in the labels, including person and car.

Run from repo root: python scripts/train_segment_finetune.py
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
COCO_ROOT = ROOT / "datasets" / "coco128-seg"
DATA_YAML = ROOT / "datasets" / "coco128_bayza.yaml"
ZIP_PATH = ROOT / "datasets" / "coco128-seg.zip"
COCO128_ZIP = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(
        url,
        headers={"User-Agent": "BayzaTrain/1.0 (compatible; educational use)"},
    )
    with urlopen(req, timeout=300) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def ensure_coco128_seg() -> None:
    train_dir = COCO_ROOT / "images" / "train2017"
    if train_dir.is_dir() and any(train_dir.iterdir()):
        return
    print("Preparing coco128-seg under", COCO_ROOT)
    if not ZIP_PATH.is_file() or ZIP_PATH.stat().st_size < 1_000_000:
        print("Downloading", COCO128_ZIP)
        _download(COCO128_ZIP, ZIP_PATH)
    print("Extracting", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(ROOT / "datasets")
    if not train_dir.is_dir():
        raise RuntimeError(f"Extracted dataset missing: {train_dir}")


def write_data_yaml() -> Path:
    template = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / "coco128-seg.yaml"
    text = template.read_text(encoding="utf-8")
    out_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("path:"):
            out_lines.append(f"path: {COCO_ROOT.as_posix()}  # Bayza local root")
        elif line.strip().startswith("download:"):
            continue
        elif "Download script/URL" in line:
            continue
        else:
            out_lines.append(line)
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return DATA_YAML


def resolve_data_arg(data: str | None) -> str:
    if data and data.lower() not in {"local", "bayza", "project"}:
        return data
    ensure_coco128_seg()
    yaml_path = write_data_yaml()
    return str(yaml_path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Light YOLOv8-seg fine-tune (coco128-seg).")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "weights" / "yolov8n-seg.pt",
        help="Starting checkpoint (downloaded if missing). n=fastest; s/m larger.",
    )
    parser.add_argument(
        "--data",
        default="local",
        help="Dataset: 'local' (default, coco128 in ./datasets), or path / ultralytics yaml name.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="How many passes over the train set.")
    parser.add_argument("--imgsz", type=int, default=640, help="Train image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (lower if OOM).")
    parser.add_argument("--device", default="0", help="cuda device index, or 'cpu'.")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "segment")
    parser.add_argument("--name", type=str, default="finetune_lite")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers (0 is safest on Windows).",
    )
    args = parser.parse_args()

    data_yaml = resolve_data_arg(args.data)
    print("Training with data YAML:", data_yaml)

    model = YOLO(str(args.model.resolve()))
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        workers=args.workers,
        exist_ok=True,
        patience=max(args.epochs, 50),
        plots=True,
        verbose=True,
    )
    best = model.trainer.best
    print("Best weights:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
