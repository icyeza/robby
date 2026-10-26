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
├── packages/robson_serving/ serving runtime: model loader, feature rows, calibration, monitoring
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
| Interpretation and deployment | `explain`, `deploy`, `package` |
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
uv run robson-ml package artefacts/readiness-v1.0 --out artefacts/packaged/readiness-v1.0  # package it for serving
ROBSON_NOTEBOOK_MODE=real uv run python scripts/run_notebook.py --mode real \
    --notebook notebooks/02_model_training.ipynb --output reports/notebooks/02.executed.ipynb
```

## Tracking experiments (MLflow)

Every training run is logged to MLflow in `mlruns/`: configuration, tuned hyperparameters per fold,
calibration choice, and metrics per held-out hospital, averaged and pooled (AUC, calibration, Brier,
log loss, PR-AUC, accuracy, precision, recall, specificity, F1). To browse and compare the runs:

```bash
uv run mlflow ui --backend-store-uri mlruns      # then open http://127.0.0.1:5000
uv run python scripts/log_threshold_metrics.py   # once: add precision/recall/F1/PR-AUC to older runs
uv run python scripts/threshold_report.py        # threshold trade-off of the selected model
```

### Threshold, F1 and PR-AUC

Selected model (L2 logistic regression, FS2, run `ae044e24`), pooled leave-one-hospital-out
out-of-fold predictions (4,265 admissions, 27.3% CS). Aggregates only; from
`scripts/threshold_report.py`, written to `reports/threshold/<run_id>.{csv,md}`.

| threshold | TP | FP | FN | TN | precision | recall | specificity | F1 | flagged % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 875 | 1,350 | 290 | 1,750 | 0.393 | 0.751 | 0.565 | 0.516 | 52.2 |
| 0.25 | 747 | 925 | 418 | 2,175 | 0.447 | 0.641 | 0.702 | 0.527 | 39.2 |
| 0.30 | 631 | 651 | 534 | 2,449 | 0.492 | 0.542 | 0.790 | 0.516 | 30.1 |
| 0.35 | 570 | 440 | 595 | 2,660 | 0.564 | 0.489 | 0.858 | 0.524 | 23.7 |
| 0.40 | 512 | 327 | 653 | 2,773 | 0.610 | 0.439 | 0.895 | 0.511 | 19.7 |
| 0.45 | 472 | 241 | 693 | 2,859 | 0.662 | 0.405 | 0.922 | 0.503 | 16.7 |
| **0.50** | 440 | 177 | 725 | 2,923 | 0.713 | 0.378 | 0.943 | 0.494 | 14.5 |
| 0.55 | 413 | 136 | 752 | 2,964 | 0.752 | 0.355 | 0.956 | 0.482 | 12.9 |
| 0.60 | 370 | 105 | 795 | 2,995 | 0.779 | 0.318 | 0.966 | 0.451 | 11.1 |

- **Why F1 at 0.5 is 0.49.** About 27% of the prediction population has a CS, so a calibrated
  probability rarely passes 0.5: only 14.5% of admissions are flagged, precision is high (0.71)
  and recall is low (0.38).
- **The threshold is a deployment choice, not a property of the model.** The model was selected on
  AUC and calibration under the pre-registered rule (`configs/selection_rule.yaml`), never on a
  thresholded metric, and the app shows the probability itself ("N in 100"), not a yes/no. The 0.5
  in MLflow is a reporting convention only.
- **Lowering the threshold buys little F1 at a large cost.** The F1-maximising threshold is 0.25
  (searched from 0.05 to 0.95 by 0.01). It raises F1 only from 0.494 to 0.527, but more than
  quintuples the false positives, from 177 to 925 vaginal births flagged as likely CS, and flags
  39% of admissions instead of 14.5%.
- **PR-AUC (average precision) is 0.584** against a no-skill baseline equal to the prevalence,
  0.273: about twice the baseline, over all thresholds.

### Why recall is low, and why that is not a flaw in the selected model

`scripts/recall_report.py` compares the best configuration of every model (and the baselines) on
the same pooled out-of-fold predictions, at the same operating points
(`reports/recall/recall_by_model.{csv,md}`).

| Model | recall at 0.5 | recall at the prevalence cut (0.273) | recall at 80% specificity |
|---|---:|---:|---:|
| B1 (Robson-group lookup) | 0.354 | 0.695 | 0.601 |
| logreg_l2 (selected) | 0.378 | 0.590 | 0.538 |
| elasticnet | 0.376 | 0.589 | 0.540 |
| xgboost | 0.362 | 0.562 | 0.543 |
| random_forest | 0.352 | 0.553 | 0.524 |
| mlp | 0.379 | 0.623 | 0.530 |
| ft_transformer | 0.347 | 0.587 | 0.585 |

(B1 and ft_transformer output few distinct probabilities or tied scores, so their 80% specificity
cut lands at 0.70 specificity; compare them at the 0.5 and prevalence columns.)

1. **The low recall comes from the 0.5 cut, not from the algorithm.** Every model, including the
   B1 lookup table, has recall of 0.35-0.38 at 0.5. With 27% CS, a calibrated probability seldom
   passes 0.5, so most women who go on to a CS are given 0.3-0.5 and are not flagged. Moving the
   cut raises recall for every model (0.55-0.70 at the prevalence cut), at the price of precision.
2. **Tuning or a bigger model does not fix it.** At matched specificity the ten models differ by
   only a few points of recall (0.52-0.54 at 80% specificity), the same ceiling their AUCs
   (about 0.75) imply. Models such as XGBoost and the MLP, tuned over 50 trials on every hospital
   (below), do not beat the regularised logistic regression. What is available at admission
   limits how well a CS can be anticipated: for the selected model, recall 0.75 means flagging
   43% of the vaginal births (specificity 0.565 at the 0.20 cut).
3. **The model is selected and shown as a probability, not as a yes/no.** Selection (the
   pre-registered rule in `configs/selection_rule.yaml`) uses AUC and calibration. The app shows
   "N in 100", and the cut is a deployment choice set by the cost of a missed CS (an unprepared
   theatre) against a false alarm (a theatre prepared and not used). The table above gives that
   trade-off at each cut.
4. **If a higher recall is wanted, the cost is explicit.** At 0.20 the selected model reaches
   recall 0.75 but flags 52% of admissions (precision 0.39).

### Raising recall: a recall-oriented operating point

Recall can be raised without touching the model, because the ranking (AUC) is unchanged and only
the cut moves. `scripts/recall_operating_point.py` picks, for each held-out hospital, the cut that
reaches a target recall on the *other* hospitals' predictions and applies it to the held-out one,
so no hospital's own outcomes are used (`reports/recall/operating_point.{csv,md}`):

| target recall | mean cut | recall obtained | specificity | precision | flagged % |
|---:|---:|---:|---:|---:|---:|
| 0.5 (default cut 0.5: 0.38) | 0.335 | 0.501 | 0.829 | 0.524 | 26.1 |
| 0.6 | 0.269 | 0.604 | 0.733 | 0.460 | 35.9 |
| 0.7 | 0.220 | 0.701 | 0.620 | 0.410 | 46.8 |
| 0.75 | 0.200 | 0.755 | 0.561 | 0.392 | 52.5 |

The target is met in the pooled data, but not equally at every hospital: with the 0.75 target
Kibagabaga obtains 0.63 and Masaka 0.84 (Masaka also flags 67% of admissions). The hospitals differ
in case mix and in how often CS is performed, so a single national cut does not give the same recall
everywhere; this is the same case-mix issue as the local recalibration (S2) in notebook 03. A
recall-oriented cut is therefore a deployment choice per hospital (cost of a missed CS against a
false alarm), not a change to the model; the deployed model, the 0.5 reporting cut and the
pre-registered selection are unchanged.

## Hyperparameter tuning experiments

Every tuned model is searched with 50 seeded Optuna TPE trials on each held-out hospital's
training rows, scored by the inner facility-grouped CV log loss. `scripts/tuning_report.py`
replays those searches and writes every trial (`reports/tuning/trials.csv`, one row per model,
hospital and trial with its hyperparameter values and loss), a per-hospital summary (best,
median, worst loss; whether the best trial reproduces the hyperparameters logged in MLflow), a
sensitivity table (rank correlation of each hyperparameter with the loss) and one figure per
model plotting the loss of every trial against each hyperparameter. Notebook 02 section 5 shows
these.

```bash
uv run python scripts/tuning_report.py [--models logreg_l2 xgboost ...]
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
