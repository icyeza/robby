# ML workstream

Everything that trains, evaluates and serves the cesarean readiness model and the Robson audit.

**Start with the notebooks.** They are the deliverable; everything else in this folder is the tested
code they call and the settings they read.

| Notebook | What it shows |
|---|---|
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | **Exploratory data analysis**: data inventory, data quality, Robson classification, outcome and predictor distributions, missingness, correlations, leakage audit, populations |
| [`notebooks/02_model_training.ipynb`](notebooks/02_model_training.ipynb) | **Model training**: feature sets, model architectures and training curve, evaluation design, training, **hyperparameter tuning** (Optuna trial curves), **experiment tracking** (MLflow), **per-model metrics table** (accuracy, precision, recall, specificity, F1, log loss, AUC, Brier), **confusion matrix per model**, calibration, pre-registered model selection |
| [`notebooks/03_model_evaluation.ipynb`](notebooks/03_model_evaluation.ipynb) | **Model evaluation**: interpretation (permutation importance, SHAP, partial dependence), robustness (internal vs other hospitals vs later period), onset sensitivity, deployment model, complete-case analysis, summary of findings |
| [`notebooks/04_audit_and_research_questions.ipynb`](notebooks/04_audit_and_research_questions.ipynb) | The Robson audit table, facility contributions, case-mix-adjusted facility CS rates (RQ1) and the missing-data / under-recording analysis (RQ2) |

The committed notebooks are executed on the real data, but show **aggregates only**: no record,
identifier or individual value is ever displayed, and counts of 1-4 are shown as `<5`.

## Layout

```
ML/
├── notebooks/          the four notebooks above
├── api/                FastAPI service: Robson classification + readiness probability (Swagger UI)
├── src/robson_ml/      the ML code the notebooks call (one module per step, see below)
├── packages/robson_engine/  the WHO Robson Ten-Group rule engine (its own small package)
├── configs/            settings read by the code (see below)
├── tests/              pytest suite, including a synthetic-data copy of the whole pipeline
└── scripts/            notebook runner and helpers
```

`data/`, `mlruns/`, `reports/` and `artefacts/` are created locally and never committed (the dataset
is under a data-sharing agreement).

### `src/robson_ml`: one module per pipeline step

| Step | Modules |
|---|---|
| Read and map the raw export | `ingest`, `mapping`, `schema`, `privacy` |
| Robson classification and data profile | `robson_run`, `profile`, `eda` |
| Features and leakage audit | `features`, `leakage`, `feature_sets`, `populations` |
| Models | `models/` (one file per model: baselines B0-B3, logistic, elastic net, CART, random forest, SVM, XGBoost, MLP, FT-Transformer) |
| Evaluation | `splits`, `evaluate`, `calibration`, `metrics`, `select`, `preregistration`, `tracking` |
| Interpretation and deployment | `explain`, `deploy` |
| Audit analyses (notebook 04) | `casemix`, `missingness`, `cmodel`, `references`, `audit_offline` |
| Figures, notebook set-up and command line | `figures`, `plots`, `reporting`, `notebook`, `cli` |

### `configs/`

| File | Used for |
|---|---|
| `project.yaml` | Where the commands read and write |
| `mapping_ur_cmhs.yaml` | Maps the raw export's columns onto the canonical admission fields |
| `features_v1.yaml` | The feature registry: every candidate input, whether it is known at admission, and why it is included or excluded |
| `selection_rule.yaml` | The model selection rule, committed **before** any model was run on real data (pre-registration). The code refuses to train on real data if this file is modified |
| `experiments/*.yaml` | The training runs: `baselines`, `main_models`, `additional_models`, `deployment_check`, `sensitivity_onset_coded`, `complete_case` |
| `deployment.yaml` | The model that gets deployed |
| `analysis.yaml` | Settings for the audit analyses in notebook 04 |

`features_v1.yaml` and `selection_rule.yaml` are left exactly as they were when the experiments ran:
their hashes are recorded with every run.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.11 and every dependency from
`uv.lock`). From this `ML/` folder:

```bash
uv sync                      # create .venv with all dependencies
uv run pytest                # run the test suite (synthetic data only)
```

## Running the notebooks

Without the private dataset, run them on **synthetic data** (the default); this builds a synthetic
project in a temporary folder and runs every step end to end:

```bash
uv run jupyter lab notebooks/          # or open the .ipynb in VS Code with the .venv kernel
uv run python scripts/run_notebook.py --mode synthetic --notebook notebooks/02_model_training.ipynb \n    --output reports/notebooks/02.synthetic.ipynb
```

With the dataset in `data/raw/`, the real pipeline is driven by the `robson-ml` command:

```bash
uv run robson-ml ingest          # raw export -> canonical admissions (identifiers hashed)
uv run robson-ml robson          # Robson classification of every admission
uv run robson-ml profile         # data-quality profile
uv run robson-ml leakage         # leakage screens for the feature registry
uv run robson-ml run configs/experiments/main_models.yaml   # train + evaluate (logged to MLflow)
uv run robson-ml fit-deploy configs/deployment.yaml         # fit the deployment model
ROBSON_NOTEBOOK_MODE=real uv run python scripts/run_notebook.py --mode real \
    --notebook notebooks/02_model_training.ipynb --output reports/notebooks/02.executed.ipynb
```

## Tracking experiments (MLflow)

Every training run is logged to MLflow in `mlruns/`: configuration, tuned hyperparameters per fold,
calibration choice, and metrics per held-out hospital, averaged and pooled (AUC, calibration, Brier,
log loss, accuracy, precision, recall, specificity, F1). To browse and compare the runs:

```bash
uv run mlflow ui --backend-store-uri mlruns      # then open http://127.0.0.1:5000
uv run python scripts/log_threshold_metrics.py   # once: add precision/recall/F1 to older runs
```

## Serving the model (API)

```bash
uv run uvicorn api.main:app --reload     # then open http://127.0.0.1:8000/docs
```

| Endpoint | What it does |
|---|---|
| `GET /health` | Service status, whether a model is loaded, model and rule-set versions |
| `GET /model` | Model card: algorithm, features, hyperparameters, calibration, training window |
| `POST /classify` | Robson group from the six Robson inputs: *resolved*, *partial* (candidate groups + the field that would decide) or *conflict* (the fields in tension), with the explanation trace |
| `POST /admissions/assess` | A full admission: rejects missing screening (blood pressure, proteinuria, glucose) unless a reason is given, classifies, and returns the readiness probability |

The API loads the model from `artefacts/readiness-v1.0/` (override with `ROBSON_MODEL_DIR`). That
model was fitted on the private data, so it is not committed. To try the API with a working model
without the data, build one from synthetic data (about a minute):

```bash
uv run python scripts/build_demo_model.py
ROBSON_MODEL_DIR=artefacts/demo-synthetic uv run uvicorn api.main:app     # bash
$env:ROBSON_MODEL_DIR="artefacts/demo-synthetic"; uv run uvicorn api.main:app   # PowerShell
```

Without any model the API still classifies admissions and returns `readiness: null` with the
reason, the same graceful degradation the app is designed for. Inputs are never stored.

Example (`POST /admissions/assess`, real model):

```json
{"facility_id": "Muhima", "maternal_age": 29, "gestational_age_weeks": 39, "parity": 2,
 "previous_cs_count": 1, "plurality": 1, "fetal_presentation": "cephalic",
 "onset_of_labour": "spontaneous",
 "screening": {"blood_pressure": {"systolic": 128, "diastolic": 84},
               "proteinuria": {"not_measured_reason": "equipment_unavailable"},
               "glucose": {"value_mmol_l": 5.2}}}
```

returns group **5a** (multiparous with a previous CS, single cephalic, 37 weeks or more), the rules
it checked, and `"readiness": {"probability": 0.75, "of_100_women_like_her": 75, "label":
"Operational readiness signal: probability of CS under current practice. Not a recommendation.",
"model_version": "readiness-v1.0", ...}`.
