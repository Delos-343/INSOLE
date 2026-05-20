# Insole Foot Classification

> Sheet-lookup, dual-rule deterministic foot-classification system for
> insole recommendation. Native desktop GUI (PySide6), FastAPI backend,
> PostgreSQL store, Docker orchestration.

---

## What it does

Given a patient code (and optional images), the system:

1. **Looks up the patient's clinical measurements** in the consolidated
   records (`data/Sheet/measurements_consolidated.xlsx`) by patient code.
2. **Classifies into one of five clinical categories** — Severe Flat
   Arch, Flat Arch, Normal Foot, High Arch, Severe High Arch — using
   two independent deterministic rules:
   - **Arch-height rule (authoritative):** centimetre bands per the brief
     (Severe Flat `< 2.7` · Flat `2.7–3.5` · Normal `3.6–5.5` · High
     `5.6–6.4` · Severe High `> 6.4`).
   - **Heel-angle rule (corroborating):** degree bands per the revision
     spec (Normal `0–5°` · Flat `> 5°` · Severe Flat `> 10°` · High `< 0°`
     · Severe High `< -5°`).
3. **Emits a recommended insole configuration** from the generative
   branch (arch support, heel cup depth, medial post, lateral wedge,
   forefoot cushioning).
4. **Persists every result** with full provenance.

### Three result states, clearly distinguished

| State | When | Authority | Confidence |
|---|---|---|---|
| **SHEET** (green) | Patient resolved; rules agree | Authoritative | 100% |
| **BOUNDARY** (blue) | Patient resolved; rules disagree by one class | Authoritative, arch-height wins; flagged for review | 100% |
| **ESTIMATED** (amber) | Patient absent from records | Assistive, **non-authoritative** | Honest sub-100% |

> Manual measurement entry is intentionally removed. Measurements come
> from the consolidated records by patient code, not by hand.

> **Why measurement-first?** Diagnostics showed arch height is not
> visually recoverable from the supplied photo views (image-only ≈ 33%),
> while a deterministic rule on the measurement is ≈ 100%. The success
> criterion was formally revised. See `Success_Criterion_Revision.docx`.

---

## Architecture

```
                ┌────────────────────────────────────────────┐
                │           Desktop GUI (PySide6)            │
                │  ┌─────────────────┐  ┌────────────────┐   │
                │  │ Classification  │  │   Training     │   │
                │  └────────┬────────┘  └───────┬────────┘   │
                └───────────┼───────────────────┼────────────┘
                            │ HTTP (Docker backend)
                            ▼                   ▼
                ┌──────────────────────┐  ┌──────────────────────┐
                │  FastAPI service     │  │  Training (threaded) │
                │  /api/classify       │  └──────────┬───────────┘
                │  /api/training/runs  │             │
                └────────┬─────────────┘             │
                         │                            │
        ┌────────────────┼──────────────────┐ ┌──────┴──────┐
        ▼                ▼                  ▼ ▼             ▼
  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
  │ Measurement  │  │ Predictor   │  │ Repositories│  │  Multi-view │
  │   lookup     │──│ (sheet-first│──│ (SQLAlchemy)│  │   network   │
  │  + dual rule │  │  authoritat-│  └──────┬──────┘  └──────┬──────┘
  └──────┬───────┘  │   ive)      │         │                │
         │          └─────────────┘         ▼                ▼
         ▼                          ┌─────────────────────────────────┐
  ┌──────────────┐                  │           PostgreSQL            │
  │  Consolidated│                  │  patients · classifications     │
  │   sheet xlsx │                  │  measurements · training_runs   │
  └──────────────┘                  └─────────────────────────────────┘
```

### Model role (honest)

The network is **not** the classifier for the core deliverable. The
deterministic rule is. The network's value-adding roles are:

- **Image → measurement estimation** — used only as a flagged assistive
  pre-fill when a patient is absent from the consolidated records.
- **Generative insole-config head** — class-conditional insole
  recommendation.

The estimated-path classification is computed by applying the same
arch-height rule to the model's *estimated* arch height — never taken
directly from network logits, which diagnostics showed are unreliable
for this data.

---

## Quick start (Docker — the supported path)

```bash
cp .env.example .env          # set POSTGRES_PASSWORD; POSTGRES_PORT if 5432 is taken

# Build & start db + backend
docker compose up -d --build db backend
docker compose ps             # both should be (healthy)

# Confirm the model loaded a real checkpoint (NOT random weights)
curl http://localhost:8000/api/health    # expect "model_loaded": true

# Launch the desktop GUI
python -m app.main
```

> **Note:** if no trained checkpoint is present, the Predictor raises
> `NoTrainedModelError` rather than silently producing garbage. Ensure
> `backend/model/checkpoints/best.pt` exists.

### Train (only needed for the estimator / insole head)

```bash
# 1. Place the consolidated measurement workbook
#    -> data/Sheet/measurements_consolidated.xlsx
# 2. ALWAYS verify before training:
docker compose exec backend python scripts/verify_dataset.py
#    expect: 5/5 classes populated, 0 duplicate patients
# 3. Train via the GUI Training tab, or:
docker compose exec backend python scripts/train.py --epochs 50
```

Training is **not required** for the core classification — that path is
a deterministic rule. Train only to improve the assistive estimator and
the insole-config head.

### Build a standalone `.exe` (Windows)

```bash
pip install pyinstaller
python app/build_exe.py        # -> dist/InsoleFootClassification/
```

The hardened build script verifies a checkpoint exists, confirms the
binary was produced, and bundles `best.pt`.

---

## Verification & diagnostics (reproducible)

Permanent parts of the codebase. Any claim about the system can be
re-checked independently.

```bash
# Dataset soundness
docker compose exec backend python scripts/verify_dataset.py

# Model characterisation: the measured / image-only / measurement-only
# accuracy decomposition
docker compose exec backend python scripts/diagnose_model.py
```

| Probe | Accuracy |
|---|---|
| Rule on true measurements (no ML) | ≈ 100% |
| Model with measurements present | ≈ 88% |
| Model images-only | ≈ 33% |

The first row is the delivered behaviour. The third is why image-only
classification is exposed only as an assistive, flagged path.

---

## Environment requirements

```
Python      3.11 – 3.13
PySide6     >=6.10, <6.11   (pinned: 6.11.x has a QLabel regression that
                             breaks GUI image previews)
torch       >=2.5
torchvision >=0.20
Docker      Desktop (WSL2 backend on Windows)
```

See `requirements.txt` for the full list. The PySide6 upper bound is
load-bearing — without it, a fresh venv install will pull 6.11.x and
the GUI image dropzones will not render uploaded images.

---

## Known limitations & quirks

- **Image-only classification is unreliable** for this dataset because
  arch height is not visually encoded in the supplied views. It is
  exposed only as an assistive, clearly-flagged estimate.
- **Estimated measurements are approximate** and must be confirmed by
  consulting the consolidated records before use.
- **Clear-button reload (cosmetic UX):** clicking *Clear* between
  patients leaves the image drop-zones unable to render the next set
  of images. Classification is unaffected (sheet lookup is keyed on
  patient code, not images). Workaround: close and relaunch the
  application between patients instead of using Clear.
- Class boundaries follow the project's fixed bands; changing them
  requires a central configuration change in `backend/model/config.py`.

---

## API reference (excerpt)

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness + `model_loaded` + DB status |
| `/api/classify` | POST | Multipart: optional images + `patient_code`; returns class, **`classification_source`** (`sheet`/`image_estimated`), `arch_class`, `heel_class`, `rules_agree`, per-class probabilities, measurements, insole config |
| `/api/training/runs` | POST | Start a training run |
| `/api/data/summary` | GET | Scan `data/` and report counts |

Swagger UI at `http://localhost:8000/docs`.

---

## Project layout

```
project/
├── app/                              # PySide6 desktop GUI
│   ├── main.py                       # Entry point
│   ├── build_exe.py                  # Hardened PyInstaller packager
│   └── ui/
│       ├── tabs/                     # classification_tab, training_tab
│       ├── widgets/                  # dropzone, results_panel, ...
│       ├── workers/                  # inference + training QThreads
│       └── theme/
│
├── backend/
│   ├── model/
│   │   ├── config.py                 # CLASS_NAMES, ARCH_HEIGHT_BANDS
│   │   ├── architectures/
│   │   ├── data/
│   │   │   ├── dataset.py
│   │   │   ├── measurement_lookup.py # Sheet lookup + dual-rule logic
│   │   │   └── transforms.py
│   │   ├── training/
│   │   ├── inference/
│   │   │   └── predictor.py          # SHEET-LOOKUP-FIRST
│   │   └── utils/checkpoint.py
│   ├── database/                     # SQLAlchemy + Alembic + Pydantic
│   └── server/                       # FastAPI
│
├── data/
│   ├── Heel/  Flat/  Normal/         # images by cohort
│   └── Sheet/measurements_consolidated.xlsx   # AUTHORITATIVE source
│
├── docker/                           # Dockerfile.backend (+trainer)
├── scripts/
│   ├── verify_dataset.py
│   ├── diagnose_model.py
│   ├── train.py  predict.py
│   └── ...
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Database

Tables: `patients` (by patient code, e.g. `P014`), `classifications`
(one row per inference, includes provenance + rule outcomes),
`measurements`, `training_runs`. Migrations via Alembic.

```bash
docker compose up -d db
docker compose exec backend alembic upgrade head
```

---

## Confidentiality

Per the project agreement, the dataset, trained checkpoints, and outputs
are **confidential and remain the property of the project owner**.
`.gitignore` excludes the image folders, the consolidated sheet, and
trained checkpoints. Do not push data or trained models to any external
or public registry.

---

## License

Proprietary. See the project agreement.
