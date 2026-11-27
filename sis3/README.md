# Breast Cancer ML System — FastAPI + Docker + MLflow + Streamlit

**Machine Learning (CSCI3234), Practical Task 6 + SIS-3** — Kamazhay Kospan

Practical Task 6 trained a model, served it with FastAPI and packaged it in Docker.
SIS-3 turns that into a small end-to-end ML system:

```
            ┌──────────────┐  log params / metrics / artifacts   ┌──────────────────┐
 train.py ─►│  6 sklearn   │ ──────────────────────────────────► │  MLflow server   │
            │  models      │  register versions + "champion"     │  tracking +      │
            └──────────────┘                                     │  model registry  │
                                                                 └────────┬─────────┘
                                                                          │ models:/breast-cancer-classifier/<v>
                                                                 ┌────────▼─────────┐
   browser ──► Streamlit UI (:8501) ──── HTTP/JSON ────────────► │  FastAPI (:8000) │
                                                                 └──────────────────┘
```

## Task

**Dataset:** Breast Cancer Wisconsin (Diagnostic), which is built into scikit-learn. It has 569 patients and 30 numeric features describing cell nuclei.
The target is flipped so that **1 = malignant** (the positive class). That way, *recall = share of real cancers the model catches*.
A missed cancer (false negative) is the costliest error, so the UI shows recall and FN next to F1.

Split: 80/20 stratified, `random_state=42`. The split lives in `app/data.py` and is shared by training and the API.

## Models (all tracked and registered)

Each model is a `Pipeline(StandardScaler → classifier)`, tuned with `GridSearchCV` (5-fold stratified CV, scoring = F1):

| Model | Test F1 | Recall | ROC-AUC | Registry version |
|---|---|---|---|---|
| **Random Forest** ⭐ champion | 0.963 | 0.929 | 0.995 | v3 |
| SVM (RBF) | 0.963 | 0.929 | 0.995 | v2 |
| Logistic Regression | 0.951 | 0.929 | 0.996 | v1 |
| Gradient Boosting | 0.950 | 0.905 | 0.995 | v4 |
| KNN | 0.911 | 0.857 | 0.983 | v5 |
| Decision Tree | 0.872 | 0.810 | 0.919 | v6 |

The champion is picked by best test F1, with ties broken by ROC-AUC. RF and SVM tie on F1, and RF wins on AUC.

## MLflow integration (SIS-3 requirements)

MLflow is used **as a Python library** inside `app/train.py`. You call `mlflow.log_*` and it gets logged.

**Experiment tracking.** The experiment is named `breast-cancer-classification`. Each training session creates one parent run, with one nested run per model.
- **Parameters:** `mlflow.log_params(...)` logs the best hyperparameters, the model type, and the CV setup.
- **Metrics:** `mlflow.log_metrics(...)` logs accuracy, precision, recall, F1, ROC-AUC, CV F1 mean/std, and training time.
- **Artifacts:**
  - the fitted model (`mlflow.sklearn.log_model`)
  - confusion matrix and ROC plots (`mlflow.log_figure`)
  - `classification_report.txt`
  - on the parent run: `leaderboard.csv`, the comparison chart, and `model.joblib`
- **Dataset lineage:** `mlflow.log_input(...)` records which training data was used.

**Model registry.**
- Every model is registered as a **new version** of `breast-cancer-classifier` (`registered_model_name=...` in `log_model`).
- Each version gets tags (`model_type`, `f1`) and a description.
- The best version gets the alias **`champion`** (`client.set_registered_model_alias`).

**Serving from the registry.** On startup, the API loads `models:/breast-cancer-classifier/<version>` for the latest version of each model type.
It serves the `champion` by default. If you retrain and press **Reload** in the UI (or call `POST /reload`), the API switches to the new champion without restarting.

## Run it

### Option A — Docker Compose (whole system, one command)
```bash
docker compose up --build
```
| Service | URL | What it does |
|---|---|---|
| Streamlit UI | http://localhost:8501 | prediction + model comparison |
| FastAPI docs | http://localhost:8000/docs | Swagger |
| MLflow UI | http://localhost:5000 | runs, metrics, artifacts, registry |

Startup order: `mlflow` (waits for its healthcheck) → `trainer` (trains, then exits) → `api` → `ui`.
The trainer skips training if models are already registered. To force a new training session (this adds new versions):
```bash
docker compose run --rm trainer python -m app.train
```
Then press **Reload models from registry** in the UI.

### Option B — locally, without Docker
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m app.train                                    # trains + logs to ./mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db      # http://localhost:5000  (terminal 2)
uvicorn app.main:app --reload                          # http://localhost:8000/docs (terminal 3)
streamlit run ui/app.py                                # http://localhost:8501  (terminal 4)
```

### Option C — Practical Task 6 mode (just the API container)
```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```
With no MLflow server reachable, the API falls back to the bundled `model.joblib` (the champion), so `GET /` and `POST /predict` still work.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | `{"message": "ML API is running", ...}` + champion + source |
| GET | `/features` | the 30 feature names with min / max / median |
| GET | `/models` | loaded models: version, aliases, params, metrics, run_id |
| GET | `/models/evaluation` | ROC / PR curves + confusion matrix per model (test set) |
| GET | `/sample?label=malignant` | random real patient from the test set |
| POST | `/predict` | one model (default `champion`; or `"svm"`, or a version like `"2"`) |
| POST | `/predict/all` | same input scored by every model |
| POST | `/reload` | re-read the registry |

Example:
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"features": {"mean radius": 20.0, "worst concave points": 0.25}, "model": "champion"}'
```
Any feature you leave out is filled with its training median, and the response lists which ones were filled.

## Streamlit UI

- **Predict:** load a random malignant or benign patient (or move the sliders), then run the chosen model.
  You get a verdict, a probability gauge, and a bar chart of *every* model on the same patient (shows where models disagree).
- **Compare models:**
  - leaderboard with the best score per metric highlighted
  - grouped metric bars
  - ROC and Precision–Recall curves overlaid
  - confusion matrices with TN/FP/FN/TP labels
  - CV-vs-test F1 chart as an overfitting check
- **MLflow registry:** versions, aliases, params, run IDs, a link to the MLflow UI, and a reload button.

The UI never loads models itself; it only calls the API. That separation (frontend ↔ serving ↔ registry) is the point of the system design.

## Project structure
```
ml-mlflow-system/
├── app/
│   ├── data.py         # dataset + shared train/test split
│   ├── train.py        # trains 6 models, MLflow tracking + registry
│   └── main.py         # FastAPI, loads models from MLflow registry
├── ui/
│   └── app.py          # Streamlit frontend
├── model.joblib        # champion as plain file (PT6 artifact / fallback)
├── requirements.txt
├── Dockerfile          # one image; default CMD = the API
├── docker-compose.yml  # mlflow + trainer + api + ui
└── screenshots/
```

## Notes
- MLflow 3 saves sklearn models in the safe *skops* format. Tree models need `skops_trusted_types=["sklearn.tree._tree.Tree"]`, which is safe here because we trained them ourselves.
- `mlflow server` is started with `--allowed-hosts` because MLflow 3 rejects unknown Host headers (DNS-rebinding protection). Inside compose the API reaches it as `mlflow:5000`.
