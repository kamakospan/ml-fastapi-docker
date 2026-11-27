"""FastAPI service that serves models straight from the MLflow Model Registry.

On startup it asks MLflow for the registered model "breast-cancer-classifier",
takes the latest version of every model type and loads them all into memory.
The version with alias "champion" is used by default for /predict.
If MLflow is unreachable it falls back to the local model.joblib (Practical Task 6 mode).

Endpoints
  GET  /                    health message (Practical Task 6 requirement)
  GET  /features            feature names + min/max/median (for the UI)
  GET  /models              loaded models with their MLflow version, aliases, params, metrics
  GET  /models/evaluation   test-set curves + confusion matrices for every model
  GET  /sample              random patient from the test set (for demo)
  POST /predict             prediction from one model (champion by default)
  POST /predict/all         the same input scored by every model
  POST /reload              re-read the registry (e.g. after retraining)
"""
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from mlflow import MlflowClient
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from app.data import CLASS_NAMES, feature_stats, get_split

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL", "breast-cancer-classifier")
CHAMPION_ALIAS = "champion"
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "model.joblib")

X_train, X_test, y_train, y_test = get_split()
FEATURES = list(X_train.columns)
STATS = feature_stats(X_train)

# key = model_type (e.g. "random_forest") -> dict(model=..., version=..., ...)
STATE = {"models": {}, "champion": None, "source": None, "evaluation": None}


def load_from_registry():
    if TRACKING_URI.startswith("http"):
        # fail fast: the MLflow client itself retries for minutes on a dead server
        requests.get(f"{TRACKING_URI.rstrip('/')}/health", timeout=3).raise_for_status()
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL}'")
    if not versions:
        raise RuntimeError("no registered versions yet - run `python -m app.train` first")

    champion_version = client.get_model_version_by_alias(REGISTERED_MODEL, CHAMPION_ALIAS).version

    # latest version of each model type, plus always the champion
    latest = {}
    for mv in versions:
        mtype = mv.tags.get("model_type", f"v{mv.version}")
        if mtype not in latest or int(mv.version) > int(latest[mtype].version):
            latest[mtype] = mv
    for mv in versions:
        if mv.version == champion_version:
            latest[mv.tags.get("model_type", f"v{mv.version}")] = mv

    models, champion = {}, None
    for mtype, mv in latest.items():
        mv = client.get_model_version(REGISTERED_MODEL, mv.version)  # search() omits aliases
        run = client.get_run(mv.run_id)
        models[mtype] = {
            "model": mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}/{mv.version}"),
            "model_type": mtype,
            "version": int(mv.version),
            "aliases": list(mv.aliases),
            "run_id": mv.run_id,
            "params": run.data.params,
            "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
        }
        if mv.version == champion_version:
            champion = mtype
    return models, champion


def load_local():
    model = joblib.load(LOCAL_MODEL_PATH)
    return {"local_joblib": {"model": model, "model_type": "local_joblib", "version": None,
                             "aliases": [CHAMPION_ALIAS], "run_id": None,
                             "params": {}, "metrics": {}}}, "local_joblib"


def load_models(retries: int = 10, wait: float = 3.0):
    last_err = None
    for _ in range(retries):
        try:
            models, champion = load_from_registry()
            STATE.update(models=models, champion=champion, source="mlflow-registry", evaluation=None)
            return
        except Exception as e:  # MLflow server may still be starting in docker
            last_err = e
            time.sleep(wait)
    if os.path.exists(LOCAL_MODEL_PATH):
        models, champion = load_local()
        STATE.update(models=models, champion=champion, source=f"local file ({last_err})", evaluation=None)
    else:
        raise RuntimeError(f"Could not load any model: {last_err}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models(retries=int(os.getenv("MLFLOW_RETRIES", "10")))
    yield


app = FastAPI(
    title="Breast Cancer Classifier API",
    description="Serves every model registered in MLflow. SIS-3 / Practical Task 6.",
    version="2.0",
    lifespan=lifespan,
)


# ----------------------------------------------------------------- schemas
EXAMPLE = {f: round(STATS[f]["median"], 4) for f in FEATURES[:10]}


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature name -> value. Any feature you leave out is filled "
                    "with its training median.",
        examples=[EXAMPLE],
    )
    model: str = Field(
        "champion",
        description="'champion', a model type (e.g. 'svm') or an MLflow version number (e.g. '2')",
    )


class Prediction(BaseModel):
    model_type: str
    version: Optional[int]
    is_champion: bool
    prediction: int
    label: str
    probability_malignant: float


class PredictResponse(Prediction):
    filled_with_median: list[str]


# ----------------------------------------------------------------- helpers
def build_row(features: dict) -> tuple[pd.DataFrame, list[str]]:
    unknown = [k for k in features if k not in STATS]
    if unknown:
        raise HTTPException(422, f"Unknown features: {unknown}. See GET /features")
    missing = [f for f in FEATURES if f not in features]
    row = {f: features.get(f, STATS[f]["median"]) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURES), missing


def resolve_model(key: str) -> dict:
    models = STATE["models"]
    if key == "champion":
        return models[STATE["champion"]]
    if key in models:
        return models[key]
    for m in models.values():
        if str(m["version"]) == key:
            return m
    raise HTTPException(404, f"Model '{key}' not loaded. Options: champion, {list(models)}")


def score(entry: dict, X: pd.DataFrame) -> dict:
    proba = float(entry["model"].predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "model_type": entry["model_type"],
        "version": entry["version"],
        "is_champion": entry["model_type"] == STATE["champion"],
        "prediction": pred,
        "label": CLASS_NAMES[pred],
        "probability_malignant": round(proba, 4),
    }


# ----------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {
        "message": "ML API is running",
        "models_loaded": len(STATE["models"]),
        "champion": STATE["champion"],
        "source": STATE["source"],
        "mlflow_tracking_uri": TRACKING_URI,
    }


@app.get("/features")
def features():
    return {"features": FEATURES, "stats": STATS, "classes": CLASS_NAMES}


@app.get("/models")
def list_models():
    return [
        {k: v for k, v in m.items() if k != "model"} | {"is_champion": m["model_type"] == STATE["champion"]}
        for m in sorted(STATE["models"].values(), key=lambda m: -m["metrics"].get("f1", 0))
    ]


@app.get("/models/evaluation")
def evaluation():
    """Recomputes metrics on the held-out test set for every loaded model (cached)."""
    if STATE["evaluation"] is None:
        out = []
        for m in STATE["models"].values():
            proba = m["model"].predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_test, proba)
            prec, rec, _ = precision_recall_curve(y_test, proba)
            out.append({
                "model_type": m["model_type"],
                "version": m["version"],
                "is_champion": m["model_type"] == STATE["champion"],
                "metrics": {
                    "accuracy": round(accuracy_score(y_test, pred), 4),
                    "precision": round(precision_score(y_test, pred), 4),
                    "recall": round(recall_score(y_test, pred), 4),
                    "f1": round(f1_score(y_test, pred), 4),
                    "roc_auc": round(roc_auc_score(y_test, proba), 4),
                },
                "cv_f1_mean": m["metrics"].get("cv_f1_mean"),
                "cv_f1_std": m["metrics"].get("cv_f1_std"),
                "train_time_sec": m["metrics"].get("train_time_sec"),
                "confusion_matrix": confusion_matrix(y_test, pred).tolist(),  # [[TN, FP], [FN, TP]]
                "roc": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
                "pr": {"precision": np.round(prec, 4).tolist(), "recall": np.round(rec, 4).tolist()},
            })
        STATE["evaluation"] = sorted(out, key=lambda e: -e["metrics"]["f1"])
    return {"n_test": len(y_test), "positive_class": "malignant", "models": STATE["evaluation"]}


@app.get("/sample")
def sample(label: str = Query("any", pattern="^(any|benign|malignant)$")):
    idx = list(range(len(X_test)))
    if label != "any":
        target = 1 if label == "malignant" else 0
        idx = [i for i in idx if int(y_test.iloc[i]) == target]
    i = random.choice(idx)
    return {
        "features": {k: float(v) for k, v in X_test.iloc[i].items()},
        "true_label": CLASS_NAMES[int(y_test.iloc[i])],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X, missing = build_row(req.features)
    return score(resolve_model(req.model), X) | {"filled_with_median": missing}


@app.post("/predict/all", response_model=list[Prediction])
def predict_all(req: PredictRequest):
    X, _ = build_row(req.features)
    return sorted((score(m, X) for m in STATE["models"].values()),
                  key=lambda p: -p["probability_malignant"])


@app.post("/reload")
def reload():
    load_models(retries=1)
    return root()
