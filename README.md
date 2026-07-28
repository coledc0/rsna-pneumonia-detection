# RSNA Pneumonia Detection

An end-to-end object detection pipeline that identifies pneumonia in chest X-rays, from raw DICOM data through a deployed, publicly accessible API on AWS.

Built on the [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) dataset (~26,000 chest X-rays), using YOLOv8 for detection, with experiment tracking, containerized deployment, and CI.

---

## Live Demo

The API is deployed on AWS ECS Fargate. Since it runs on a paid AWS service, it may be paused to manage cost — if the link below doesn't respond, it means the service is currently scaled down. Reach out and I'll spin it back up in under a minute.

```
http://<current-task-ip>:8000/docs
```

*(Note: since this deployment has no load balancer, the public IP changes each time the service restarts. Check back here or ask for the current IP.)*

---

## Problem

Pneumonia detection from chest X-rays is a well-studied but genuinely difficult computer vision problem — subtle radiographic findings, class imbalance, and high stakes for false negatives. The RSNA dataset frames this as an **object detection** task: given a chest X-ray, predict bounding boxes around regions of lung opacity consistent with pneumonia.

This project treats it as binary object detection (pneumonia present / not present, with localization) using YOLOv8, and builds a full deployment pipeline around the trained model rather than stopping at a notebook.

---

## Pipeline Overview

```
Raw DICOM (26,684 patients)
        │
        ▼
   EDA & class balance analysis (notebooks/01_eda.ipynb)
        │
        ▼
   DICOM → PNG conversion + RSNA → YOLO label conversion (src/dataset.py)
        │
        ▼
   YOLOv8 training with MLflow experiment tracking (src/train.py)
        │
        ▼
   Automatic best-checkpoint selection across runs (src/model_utils.py)
        │
        ▼
   FastAPI inference service (api/main.py) ── tested (tests/) ── CI (GitHub Actions)
        │
        ▼
   Docker container
        │
        ▼
   Weights promoted to S3  ──►  Deployed on AWS ECS Fargate (public IP, no load balancer)
```

---

## Results

| Metric | Value |
|---|---|
| mAP50 | 0.374 |
| mAP50-95 | 0.147 |
| Precision | 0.413 |
| Recall | 0.442 |

**For context:** an untuned YOLOv3 baseline on this dataset scores ~0.32 mAP50; the official RSNA challenge metric (a stricter multi-threshold average) benchmarks around 0.25. Top competitive solutions using ensembles and heavier engineering reach 0.55–0.65.

### What the tuning process found

Five+ training runs were used to isolate what actually mattered:

- **Learning rate was the single biggest lever.** Dropping from `0.005` → `0.001` produced the largest single improvement across all runs. Going lower still (`0.0005`) did not help further — `0.001` sits near the optimum for this setup.
- **Bigger model ≠ better.** YOLOv8x (extra-large) overfit faster and scored *worse* than YOLOv8m (medium) at the same learning rate — more capacity without matching regularization just memorized faster.
- **Image resolution (512 vs 640) made no meaningful difference** despite bounding boxes averaging only 3–8% of image area — the expected benefit from EDA didn't materialize in practice.
- **A consistent overfitting point emerged around epoch 30–40** across nearly every configuration, suggesting it's closer to a property of the dataset/task than a hyperparameter to tune away. Warmup duration and weight decay were used to fight it directly.

All experiments are tracked in MLflow (`mlflow_tracking/`, gitignored) with full metric/parameter logging per run.

---

## Tech Stack

- **Model:** YOLOv8 (Ultralytics), pretrained on COCO, fine-tuned on RSNA
- **Training:** PyTorch, MLflow for experiment tracking
- **Data:** pydicom, OpenCV, albumentations
- **API:** FastAPI, Pydantic, Uvicorn
- **Deployment:** Docker, AWS ECS Fargate, AWS S3 (weight storage), IAM (least-privilege task roles)
- **Testing/CI:** pytest, ruff, GitHub Actions

---

## Project Structure

```
rsna-pneumonia/
├── api/                  # FastAPI application
│   ├── main.py
│   └── schemas.py
├── src/                  # Core pipeline code
│   ├── dataset.py        # DICOM → YOLO preprocessing
│   ├── train.py          # Training loop + MLflow logging
│   ├── model_utils.py    # Best-checkpoint selection with corruption handling
│   ├── predict.py        # Inference logic (shared by API and scripts)
│   └── evaluate.py       # Visual ground-truth vs. prediction comparison
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_experiments.ipynb
├── tests/
│   ├── test_dataset.py   # Unit tests for RSNA→YOLO conversion, label dedup
│   └── test_api.py       # API endpoint tests (mocked model)
├── configs/
│   └── config.yaml       # All hyperparameters and paths
├── .github/workflows/
│   └── ci.yml            # Lint + test on every push
├── Dockerfile
└── requirements.txt
```

---

## Setup

**1. Clone and create the environment:**

```bash
git clone https://github.com/coledc0/rsna-pneumonia-detection.git
cd rsna-pneumonia-detection
conda create -n rsna python=3.11
conda activate rsna
```

**2. Install PyTorch** (CPU or CUDA build depending on your hardware — see [pytorch.org](https://pytorch.org/get-started/locally/)), then the rest:

```bash
pip install -r requirements.txt
```

**3. Download the data** from the [Kaggle competition page](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) and place it under `data/raw/`.

**4. Run preprocessing:**

```bash
python src/dataset.py
```

**5. Train:**

```bash
python src/train.py
```

Training config (architecture, learning rate, epochs, augmentation) lives entirely in `configs/config.yaml`.

---

## Running the API Locally

```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI. The `/predict` endpoint accepts a PNG/JPEG chest X-ray and returns:

```json
{
  "filename": "example.png",
  "predicted": true,
  "confidence": 0.2813,
  "n_detections": 2,
  "boxes": [
    { "x1": 111, "y1": 126, "x2": 228, "y2": 383, "confidence": 0.2813 },
    { "x1": 282, "y1": 125, "x2": 423, "y2": 371, "confidence": 0.2663 }
  ]
}
```

By default the API looks for local weights (auto-selecting the best available training run). Set `S3_BUCKET_NAME` and `S3_WEIGHTS_KEY` environment variables to instead download weights from S3 at startup — this is the path used in production.

---

## Running with Docker

```bash
docker build -t rsna-api .
docker run -p 8000:8000 -v ${PWD}/runs:/app/runs rsna-api
```

---

## Testing

```bash
pytest tests/ -v
ruff check .
```

Both run automatically on every push via GitHub Actions.

---

## Notes on What's Next

- Hyperparameter search is deliberately not exhaustive — the current model represents a reasonable stopping point given diminishing returns, not a ceiling. Further gains are available (see Results section) but weren't worth the time tradeoff for this project's goals.
- The deployment currently runs without a load balancer to avoid its fixed monthly cost; a production version would add one back for a stable URL and TLS termination.