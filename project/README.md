# Insole Foot Classification

> AI-powered foot-classification system for insole recommendation.
> Multi-view CNN + cross-modal fusion + generative VAE branch, with a
> native desktop GUI, FastAPI service, PostgreSQL store, and Docker
> orchestration.

---

## What it does

Given up to **three foot images** (lateral / top / back) and an optional
set of clinical measurements, the system:

1. Classifies the foot into one of **five clinical categories** — Severe
   Flat Arch, Flat Arch, Normal Foot, High Arch, Severe High Arch.
2. Predicts the **five clinical angles** from images alone if the user
   doesn't provide them (calcaneal inclination, heel angle, arch height,
   kite angle, 1st metatarsal–talus angle).
3. Emits a **recommended insole configuration** (arch support height,
   heel cup depth, medial post strength, lateral wedge strength,
   forefoot cushioning) sampled from the generative branch's
   class-conditional manifold.
4. Persists every result to PostgreSQL with full audit metadata.

The target accuracy from the brief — **≥90% on unseen images** — is
addressed by (a) per-view ImageNet-pretrained backbones, (b) cross-modal
transformer fusion of views + measurements, (c) class-balanced sampling,
(d) Albumentations augmentation, and (e) a multi-task loss with
auxiliary measurement regression and VAE regularisation.

---

## Architecture

```
                ┌───────────────────────────────────────────┐
                │           Desktop GUI (PySide6)           │
                │  ┌─────────────────┐  ┌───────────────┐   │
                │  │ Classification  │  │   Training    │   │
                │  │      tab        │  │      tab      │   │
                │  └────────┬────────┘  └────────┬──────┘   │
                └───────────┼────────────────────┼──────────┘
                            │ HTTP (local fallback in-proc) │
                            ▼                              ▼
                ┌───────────────────────────┐   ┌──────────────────────┐
                │   FastAPI service         │   │  In-process Trainer  │
                │  /api/classify            │   │  (QThread worker)    │
                │  /api/training/runs       │   └──────────┬───────────┘
                │  /api/data/summary        │              │
                │  /api/patients            │              │
                └─────────┬─────────────────┘              │
                          │                                 │
              ┌───────────┴───────────┐         ┌──────────┴─────────┐
              ▼                       ▼         ▼                    ▼
       ┌─────────────┐        ┌─────────────┐  ┌──────────────┐  ┌────────┐
       │ Predictor   │        │ Repositories│  │ Multi-view   │  │ Data   │
       │ (PyTorch)   │        │ (SQLAlchemy)│  │ Classifier   │  │ loader │
       └──────┬──────┘        └──────┬──────┘  └──────┬───────┘  └────┬───┘
              │                      │                │               │
              ▼                      ▼                ▼               ▼
       ┌─────────────┐        ┌─────────────────────────────────────────┐
       │ Checkpoints │        │              PostgreSQL                 │
       │  (volume)   │        │  patients · classifications ·           │
       └─────────────┘        │  measurements · training_runs           │
                              └─────────────────────────────────────────┘
```

### Model topology

```
   Lateral ─┐                                       ┌─► logits (5 classes)
   Top  ────┼─► [ViewEncoder × 3]                   │
   Back ────┘        │                              │
                     ▼                              │
              MultiModalFusion ──► fused (B, 512) ──┼─► insole_config (5)
                     ▲                              │
   Measurements ─────┘                              ├─► measurements_hat (5)
   (+ mask)                                         │
                                                    └─► VAE (recon, μ, logσ²)
                                                          ↑ class-conditional
                                                            generative branch
```

- Per-view encoders are `timm` models (default: EfficientNet-B0).
- Fusion is a small transformer over 4 tokens (3 views + 1 measurement)
  with learned role embeddings.
- The VAE branch operates *in the fused-embedding space* — cheap, and
  doubles as feature-space mixup augmentation during training.

---

## Quick start

### Option A — Local Python (fastest iteration)

```bash
# 1. Create env + install deps
make install          # or: python -m venv .venv && pip install -r requirements.txt

# 2. (Optional) Start Postgres in Docker
docker compose up -d db

# 3. (Optional) Apply migrations — only if you started Postgres
make migrate

# 4. Launch the desktop GUI
make app
```

The GUI's "Classification" tab works out of the box even without a
trained model (random initialisation). To train:

```bash
# Copy or rsync your dataset into ./data/  (mirroring Drive layout)
make train EPOCHS=50 BATCH_SIZE=16
```

### Option B — Docker (production-ish)

```bash
cp .env.example .env

docker compose up -d --build db backend
# Open http://localhost:8000/docs

# Train inside the trainer container:
docker compose --profile training run --rm trainer \
   --data-dir /workspace/data --epochs 50
```

### Option C — Build a standalone `.exe`

```bash
make build-exe        # output -> dist/InsoleFootClassification/
```

On Windows this produces a `.exe` you can ship; on macOS a `.app` bundle;
on Linux a single-folder ELF binary.

---

## Project layout

```
insole-foot-classification/
├── app/                                # Frontend (PySide6 desktop GUI)
│   ├── main.py                         # Entry point
│   ├── config.py                       # GUI config + paths
│   ├── build_exe.py                    # PyInstaller packager
│   └── ui/
│       ├── main_window.py              # QMainWindow + tabs + menu
│       ├── tabs/
│       │   ├── classification_tab.py   # PRIMARY tab — matches brief layout
│       │   └── training_tab.py         # SECONDARY tab — training console
│       ├── widgets/
│       │   ├── image_dropzone.py       # Drag-and-drop image tile
│       │   ├── measurement_panel.py    # 5 angle/height inputs
│       │   ├── results_panel.py        # Class probs + insole config display
│       │   └── log_console.py
│       ├── workers/
│       │   ├── inference_worker.py     # QThread, falls back to local model
│       │   └── training_worker.py      # QThread wrapping the Trainer
│       └── theme/                      # Dark palette + stylesheet
│
├── backend/
│   ├── model/                          # The AI itself
│   │   ├── config.py                   # ModelConfig / TrainingConfig dataclasses
│   │   ├── architectures/
│   │   │   ├── view_encoder.py         # Per-view CNN (timm-backed)
│   │   │   ├── fusion_network.py       # Cross-modal transformer fusion
│   │   │   ├── generative_vae.py       # Conditional VAE + InsoleConfigHead
│   │   │   ├── measurement_predictor.py# Angle regression head
│   │   │   └── classifier.py           # MultiViewFootClassifier (top-level)
│   │   ├── data/
│   │   │   ├── dataset.py              # Walks the Drive folder structure
│   │   │   ├── transforms.py           # Albumentations train/eval pipelines
│   │   │   └── dataloader.py           # Stratified split + balanced sampler
│   │   ├── training/
│   │   │   ├── losses.py               # Multi-task loss (CE + Smooth-L1 + KL)
│   │   │   ├── metrics.py              # Acc, macro-F1, confusion matrix
│   │   │   └── trainer.py              # Mixed precision, ES, checkpoints
│   │   ├── inference/
│   │   │   └── predictor.py            # Stateful Predictor + rule-based check
│   │   ├── preprocessing/
│   │   │   └── image_processor.py      # EXIF-aware loading, letterbox pad
│   │   ├── utils/                      # Checkpoint + seeding helpers
│   │   └── checkpoints/                # Trained weights live here
│   │
│   ├── database/                       # PostgreSQL + Prisma + SQLAlchemy
│   │   ├── schema.prisma               # Source-of-truth schema
│   │   ├── models.py                   # SQLAlchemy ORM (runtime)
│   │   ├── schemas.py                  # Pydantic API I/O
│   │   ├── connection.py               # Engine + session factory
│   │   ├── repositories/
│   │   │   ├── patient_repo.py
│   │   │   ├── classification_repo.py
│   │   │   └── training_run_repo.py
│   │   └── migrations/                 # Alembic
│   │       ├── env.py
│   │       └── versions/0001_initial.py
│   │
│   └── server/                         # FastAPI HTTP layer
│       ├── main.py                     # App factory + lifespan
│       ├── middleware.py               # Request-ID + logging
│       ├── dependencies.py
│       ├── utils/file_handler.py
│       └── routes/
│           ├── health.py               # /api/health
│           ├── classification.py       # /api/classify
│           ├── training.py             # /api/training/runs
│           ├── data_router.py          # /api/data/summary
│           └── patients.py             # /api/patients
│
├── data/                               # Dataset (mirrors Drive folder)
│   ├── Heel/  Flat/  Normal/  Sheet/   #  ← per the brief
│   └── README.md
│
├── docker/
│   ├── Dockerfile.backend              # FastAPI + ML image
│   ├── Dockerfile.trainer              # CLI training image
│   └── postgres-init.sql               # uuid-ossp + pgcrypto + tz=UTC
│
├── scripts/
│   ├── train.py                        # CLI: train a model
│   ├── predict.py                      # CLI: predict on three images
│   ├── prepare_data.py                 # CLI: scan + manifest
│   ├── seed_demo_data.py               # CLI: seed Postgres with demo rows
│   └── export_onnx.py                  # CLI: torch -> ONNX
│
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_api.py
│
├── docker-compose.yml                  # db + backend + trainer + pgadmin
├── alembic.ini
├── Makefile                            # `make help`
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── .gitignore
```

---

## Common commands

```bash
make help              # show every target
make app               # launch desktop GUI
make api               # uvicorn dev server with reload
make train EPOCHS=20   # train (override hyper-params on the CLI)
make test              # run pytest
make lint              # ruff + black --check
make format            # auto-format
make build-exe         # bundle desktop app via PyInstaller
make docker-up         # docker compose up -d db backend
make docker-train      # one-shot training job in a container
```

---

## API reference (excerpt)

| Endpoint                  | Method | Description                                   |
| ------------------------- | ------ | --------------------------------------------- |
| `/api/health`             | GET    | Liveness + model + DB status                  |
| `/api/classify`           | POST   | Multipart: lateral/top/back images + measurements_json |
| `/api/training/runs`      | POST   | Kick off a training run                       |
| `/api/training/runs`      | GET    | List recent training runs                     |
| `/api/training/runs/{id}` | GET    | Status of a single run                        |
| `/api/data/summary`       | GET    | Scan `data/` and report counts                |
| `/api/patients`           | POST/GET | Patient CRUD                                |

Interactive Swagger UI at `http://localhost:8000/docs`.

---

## Database

The canonical schema is `backend/database/schema.prisma`. SQLAlchemy
models in `models.py` mirror it 1:1. Migrations are managed with
Alembic.

Tables:

- `patients` — anonymised by patient code (e.g. `P1097`)
- `classifications` — one row per inference run
- `measurements` — clinician-entered or imported measurement records
- `training_runs` — full audit trail of every training experiment

To bootstrap a fresh DB:

```bash
make docker-up        # starts postgres
make migrate          # alembic upgrade head
make seed-db          # optional: demo data
```

---

## Configuration

All runtime config goes through environment variables (see
`.env.example`). The most relevant:

| Variable                  | Default                                  | Purpose                |
| ------------------------- | ---------------------------------------- | ---------------------- |
| `DATABASE_URL`            | derived from `POSTGRES_*`                | SQLAlchemy DSN         |
| `API_PORT`                | `8000`                                   | FastAPI port           |
| `DATA_DIR`                | `./data`                                 | Dataset root           |
| `DEFAULT_CHECKPOINT_PATH` | `./backend/model/checkpoints/best.pt`    | Loaded at API startup  |
| `MAX_UPLOAD_MB`           | `25`                                     | Per-image upload limit |

---

## Training tips

- **First run**: `make prepare-data` first — it prints how many patients,
  per-class counts, and surfaces any missing measurements before you
  burn 50 epochs.
- **Class imbalance**: handled automatically via `WeightedRandomSampler`
  + class-frequency-weighted cross-entropy.
- **Multi-task signal**: even when only a fraction of patients have
  measurements, the measurement-regression head still learns from those
  rows (mask-weighted).
- **Mixed precision**: enabled by default on CUDA; ignored on CPU/MPS.
- **Early stopping**: triggered after `early_stopping_patience` epochs
  without val-acc improvement (default 10).

---

## Confidentiality

Per the project brief, the dataset, trained checkpoints, and outputs are
**confidential and remain the property of the project owner**. The
`.gitignore` excludes `data/Heel/*`, `data/Flat/*`, `data/Normal/*`,
`data/Sheet/*`, and `backend/model/checkpoints/*.pt` to prevent
accidental commits. Do not push the trained model to a public registry.

---

## License

Proprietary. See the project agreement.
