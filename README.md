# Bayza

**Student:** 230212044 — Beyza Güler

YOLOv8-seg utilities: object detection (bounding boxes and class labels) and instance segmentation (pixel masks) for people and cars on images, plus COCO val prep and short training scripts.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Weights live under `weights/` (`.pt` files are gitignored; Ultralytics can download missing checkpoints on first run).

## Scripts

Run from the repo root so paths resolve correctly.

| Command | Purpose |
|--------|---------|
| `python scripts/segment_humans_cars.py` | Draw person and car bounding boxes, scores, and masks on images in `data/samples/` (or pass `--source`; `--no-boxes` for masks only) |
| `python scripts/prepare_coco_val5k.py` | Download COCO 2017 val images + segment labels into `datasets/coco_val5k/` |
| `python scripts/train_run_30min.py` | Time-capped fine-tune on that subset |
| `python scripts/train_segment_finetune.py` | Quick run on `coco128-seg` |
| `python scripts/generate_report_figures.py` | Export high-DPI training curves and a qualitative figure into `reports/` |

## Layout

- `scripts/` — entry points
- `weights/` — checkpoints (local only)
- `data/samples/` — demo inputs
- `datasets/` — prepared COCO subsets and YAMLs
- `outputs/` — segmentation results
- `reports/` — exported figures for write-ups (`generate_report_figures.py`)
- `runs/` — Ultralytics training exports
