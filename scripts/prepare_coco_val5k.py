"""
Download COCO 2017 val images (~5k, ~815 MB) + YOLO segment labels from Ultralytics,
layout under datasets/coco_val5k/ for training without the full 20 GB COCO train set.

Splits val images 90% train / 10% val (disjoint file lists; both live in images/val2017/).
train.txt / val.txt use absolute .jpg paths so Ultralytics resolves labels correctly on Windows.

Run from repo root: python scripts/prepare_coco_val5k.py
"""
from __future__ import annotations

import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import ultralytics

ROOT = Path(__file__).resolve().parents[1]
DROOT = ROOT / "datasets" / "coco_val5k"
IMAGES_DIR = DROOT / "images" / "val2017"
LABELS_DIR = DROOT / "labels" / "val2017"
TRAIN_TXT = DROOT / "train.txt"
VAL_TXT = DROOT / "val.txt"
YAML_OUT = DROOT / "coco_val5k_bayza.yaml"

VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
LABELS_ZIP_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip"
)
VAL_ZIP_LOCAL = DROOT / "_val2017.zip"
LABELS_ZIP_LOCAL = DROOT / "_coco2017labels-segments.zip"
STAGE = DROOT / "_stage_extract"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(
        url,
        headers={"User-Agent": "BayzaPrepareCOCO/1.0 (compatible; educational use)"},
    )
    print(f"Downloading\n  {url}\n  -> {dest}")
    with urlopen(req, timeout=600) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, length=16 * 1024 * 1024)
    print("Download done:", dest.stat().st_size // (1024 * 1024), "MB")


def _extract_val_labels_from_zip(zip_path: Path) -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "coco/labels/val2017/"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".txt")]
        if not names:
            raise RuntimeError("No val2017 label files found in labels zip.")
        print(f"Extracting {len(names)} label files from zip…")
        for n in names:
            target = LABELS_DIR / Path(n).name
            if not target.exists():
                with zf.open(n) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def _extract_val_images_zip(zip_path: Path) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if any(IMAGES_DIR.glob("*.jpg")):
        n = len(list(IMAGES_DIR.glob("*.jpg")))
        if n > 4000:
            print(f"Images already present ({n} jpgs), skipping image extract.")
            return
    STAGE.mkdir(parents=True, exist_ok=True)
    print("Extracting val2017 images (one-time, a few minutes on SSD)…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(STAGE)
    # Typical layout: stage/val2017/000000xxxxx.jpg
    cand = list(STAGE.glob("val2017/*.jpg"))
    if not cand:
        cand = list(STAGE.rglob("*.jpg"))
    if not cand:
        raise RuntimeError("No jpg files found after extracting val2017.zip")
    for p in cand:
        dest = IMAGES_DIR / p.name
        if not dest.exists():
            shutil.move(str(p), str(dest))
    shutil.rmtree(STAGE, ignore_errors=True)
    print("Images ready:", len(list(IMAGES_DIR.glob("*.jpg"))), "files in", IMAGES_DIR)


def _write_split_txts(seed: int = 42, val_fraction: float = 0.1) -> None:
    stems_img = {p.stem for p in IMAGES_DIR.glob("*.jpg")}
    stems_lbl = {p.stem for p in LABELS_DIR.glob("*.txt")}
    common = sorted(stems_img & stems_lbl)
    if len(common) < 100:
        raise RuntimeError(f"Too few matched image/label pairs: {len(common)}")
    rng = random.Random(seed)
    rng.shuffle(common)
    n_val = max(1, int(len(common) * val_fraction))
    val_set = common[:n_val]
    train_set = common[n_val:]

    def write_list(path: Path, stems: list[str]) -> None:
        # Absolute paths: Ultralytics resolves txt entries relative to CWD otherwise.
        lines = [(IMAGES_DIR / f"{s}.jpg").resolve().as_posix() for s in stems]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_list(VAL_TXT, val_set)
    write_list(TRAIN_TXT, train_set)
    print(f"Split: train={len(train_set)} val={len(val_set)} (written {TRAIN_TXT.name}, {VAL_TXT.name})")


def _write_data_yaml() -> Path:
    coco_yaml = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / "coco.yaml"
    text = coco_yaml.read_text(encoding="utf-8")
    cut = text.index("# Download script")
    lines_out: list[str] = []
    for line in text[:cut].splitlines():
        if line.startswith("path:"):
            lines_out.append(f"path: {DROOT.as_posix()}  # Bayza COCO val subset root")
        elif line.startswith("train:"):
            lines_out.append("train: train.txt  # 90% of val2017 image ids")
        elif line.startswith("val:"):
            lines_out.append("val: val.txt  # 10% holdout")
        elif line.startswith("test:"):
            lines_out.append("test: # unused")
        else:
            lines_out.append(line)
    YAML_OUT.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print("Wrote", YAML_OUT)
    return YAML_OUT


def ensure_coco_val5k_dataset(
    skip_image_download: bool = False,
    skip_label_download: bool = False,
) -> Path:
    """
    Ensure images + labels exist, write train/val txts and return path to data yaml.
    """
    DROOT.mkdir(parents=True, exist_ok=True)

    if not skip_label_download:
        if not LABELS_ZIP_LOCAL.is_file() or LABELS_ZIP_LOCAL.stat().st_size < 1_000_000:
            _download(LABELS_ZIP_URL, LABELS_ZIP_LOCAL)
        if not any(LABELS_DIR.glob("*.txt")):
            _extract_val_labels_from_zip(LABELS_ZIP_LOCAL)

    if not skip_image_download:
        if not VAL_ZIP_LOCAL.is_file() or VAL_ZIP_LOCAL.stat().st_size < 100_000_000:
            _download(VAL_IMAGES_URL, VAL_ZIP_LOCAL)
        _extract_val_images_zip(VAL_ZIP_LOCAL)

    _write_split_txts()
    return _write_data_yaml()


if __name__ == "__main__":
    ensure_coco_val5k_dataset()
    print("OK:", YAML_OUT)
