"""
Object detection + instance segmentation: people and cars with YOLOv8-seg
(COCO classes 0=person, 2=car). Draws bounding boxes with class names and scores,
then semi-transparent masks and contours.

Weights download automatically on first run (~23 MB for yolov8s-seg.pt into weights/).

Run from repo root: python scripts/segment_humans_cars.py [--source PATH ...]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

# COCO class ids used by Ultralytics YOLO
CLASS_PERSON = 0
CLASS_CAR = 2
CLASS_NAMES = {CLASS_PERSON: "person", CLASS_CAR: "car"}
# BGR for overlay
COLORS = {CLASS_PERSON: (0, 180, 255), CLASS_CAR: (0, 255, 100)}

DEFAULT_SAMPLES = [
    (
        "bus.jpg",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg",
    ),
    (
        "zidane.jpg",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/zidane.jpg",
    ),
]


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BayzaSegmentDemo/1.0 (compatible; educational use)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        out.write(resp.read())


def download_samples(data_dir: Path) -> list[Path]:
    out: list[Path] = []
    for name, url in DEFAULT_SAMPLES:
        p = data_dir / name
        download_file(url, p)
        out.append(p)
    return out


def blend_instance_masks(
    image_bgr: np.ndarray,
    masks_hw: np.ndarray,
    classes: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """masks_hw: (N, H, W) aligned to image_bgr; classes parallel to N."""
    vis = image_bgr.astype(np.float32)
    for mask, cls in zip(masks_hw, classes):
        c = int(cls)
        if c not in COLORS:
            continue
        color = np.array(COLORS[c], dtype=np.float32)
        m = mask > 0.5
        for ch in range(3):
            plane = vis[:, :, ch]
            plane[m] = plane[m] * (1.0 - alpha) + color[ch] * alpha
            vis[:, :, ch] = plane
    return np.clip(vis, 0, 255).astype(np.uint8)


def draw_detection_boxes(
    image_bgr: np.ndarray,
    xyxy: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
) -> None:
    """Draw axis-aligned boxes and labels in place (xyxy in pixel coords, N×4)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x1, y1, x2, y2), cls_arr, conf in zip(xyxy, classes, confidences):
        c = int(cls_arr)
        if c not in COLORS:
            continue
        color = COLORS[c]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(image_bgr, p1, p2, color, 2)
        name = CLASS_NAMES.get(c, str(c))
        label = f"{name} {float(conf):.2f}"
        (tw, th), baseline = cv2.getTextSize(label, font, 0.5, 1)
        ty = max(p1[1] - 4, th + 6)
        cv2.rectangle(
            image_bgr,
            (p1[0], ty - th - 6),
            (p1[0] + tw + 4, ty + baseline - 2),
            color,
            -1,
        )
        cv2.putText(image_bgr, label, (p1[0] + 2, ty - 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def run_segmentation(
    model: YOLO,
    source: Path,
    out_dir: Path,
    conf: float,
    alpha: float,
    draw_boxes: bool,
) -> Path:
    im = cv2.imread(str(source))
    if im is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    results = model.predict(
        source=im,
        classes=[CLASS_PERSON, CLASS_CAR],
        conf=conf,
        verbose=False,
    )[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    vis = im.copy()

    if results.masks is not None and len(results.masks) > 0:
        # (N, H, W) after numpy()
        masks = results.masks.data.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy().astype(int)
        # Resize masks to original image size
        h0, w0 = im.shape[:2]
        ms = []
        for m in masks:
            ms.append(cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR))
        masks_hw = np.stack(ms, axis=0)
        vis = blend_instance_masks(im, masks_hw, cls, alpha=alpha)

        # Thin outline per instance
        for m, c in zip(masks_hw, cls):
            if int(c) not in COLORS:
                continue
            binm = (m > 0.5).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, COLORS[int(c)], 2)

    if draw_boxes and results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy().astype(int)
        conf = results.boxes.conf.cpu().numpy()
        draw_detection_boxes(vis, xyxy, cls, conf)

    out_path = out_dir / f"seg_{source.stem}.jpg"
    cv2.imwrite(str(out_path), vis)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Object detection (boxes) + instance segmentation (masks) for person and car (YOLOv8-seg)."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "weights" / "yolov8s-seg.pt",
        help="Model weights (downloaded automatically if missing).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        nargs="*",
        help="Images or folders. If omitted, uses data/samples after --download-samples.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs",
        help="Output directory for segmented images.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask fill opacity.")
    parser.add_argument(
        "--no-boxes",
        action="store_true",
        help="Do not draw bounding boxes or labels (masks and contours only).",
    )
    parser.add_argument(
        "--download-samples",
        action="store_true",
        help="Download demo images to data/samples/ (Ultralytics assets).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "samples",
        help="Where sample images are stored.",
    )
    args = parser.parse_args()

    if args.download_samples:
        download_samples(args.data_dir)

    sources: list[Path] = []
    if args.source:
        for s in args.source:
            if s.is_dir():
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
                    sources.extend(sorted(s.glob(ext)))
            else:
                sources.append(s)
    else:
        args.data_dir.mkdir(parents=True, exist_ok=True)
        if not any(args.data_dir.iterdir()):
            print("No --source given; downloading default samples to", args.data_dir)
            download_samples(args.data_dir)
        sources = sorted(
            p for p in args.data_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        )
        if not sources:
            print("No images found. Run with --download-samples or pass --source paths.", file=sys.stderr)
            return 1

    print("Loading model (first run downloads weights)...")
    model = YOLO(str(args.weights.resolve()))

    for src in sources:
        if not src.is_file():
            print("Skip missing:", src, file=sys.stderr)
            continue
        out = run_segmentation(
            model,
            src,
            args.out,
            args.conf,
            args.alpha,
            draw_boxes=not args.no_boxes,
        )
        print("Wrote", out)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
