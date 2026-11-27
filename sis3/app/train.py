"""Train several classifiers, track everything in MLflow, register every model.

What gets logged for each model (one MLflow run per model, nested under one
"training session" parent run):
  * params     - best hyperparameters from GridSearchCV + model type
  * metrics    - accuracy, precision, recall, f1, roc_auc on the test set,
                 cross-validation f1 (mean/std), training time
  * artifacts  - the fitted sklearn pipeline (MLflow model format),
                 confusion matrix + ROC curve plots, classification report
  * registry   - every model becomes a new version of ONE registered model
                 ("breast-cancer-classifier"); the best one (by F1) gets the
                 alias "champion", which the API serves by default.

Run:  python -m app.train
"""
import os
import time
import warnings

# hide sklearn's SVC(probability=True) deprecation noise (also in joblib workers)
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.data import RANDOM_STATE, get_split

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "breast-cancer-classification")
REGISTERED_MODEL = os.getenv("REGISTERED_MODEL", "breast-cancer-classifier")
CHAMPION_ALIAS = "champion"

# model name -> (estimator, small hyperparameter grid)
# every estimator is wrapped in a Pipeline with StandardScaler, so scaling is
# part of the saved model and the API can send raw feature values.
MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        {"clf__C": [0.1, 1, 10]},
    ),
    "svm": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {"clf__C": [0.5, 1, 5], "clf__kernel": ["rbf", "linear"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 6]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]},
    ),
    "knn": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 9], "clf__weights": ["uniform", "distance"]},
    ),
    "decision_tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 5]},
    ),
}


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def plot_confusion(model, X_test, y_test, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["benign", "malignant"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion matrix - {title}")
    fig.tight_layout()
    return fig


def plot_roc(model, X_test, y_test, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=title)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax.set_title(f"ROC curve - {title}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # docker compose runs this on every `up`; skip if models are already registered
    # (force a retrain with: docker compose run --rm trainer python -m app.train)
    if os.getenv("TRAIN_ONLY_IF_EMPTY") == "1":
        if client.search_model_versions(f"name='{REGISTERED_MODEL}'"):
            print(f"'{REGISTERED_MODEL}' already has versions - skipping training.")
            return

    X_train, X_test, y_train, y_test = get_split()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"Tracking URI: {TRACKING_URI}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    leaderboard = []
    session_name = time.strftime("training-session-%Y%m%d-%H%M%S")

    with mlflow.start_run(run_name=session_name) as parent:
        mlflow.log_params({
            "dataset": "sklearn breast_cancer",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "cv_folds": 5,
            "selection_metric": "f1",
        })
        train_df = X_train.assign(malignant=y_train.values)
        mlflow.log_input(
            mlflow.data.from_pandas(train_df, name="breast_cancer_train", targets="malignant"),
            context="training",
        )

        for name, (estimator, grid) in MODELS.items():
            with mlflow.start_run(run_name=name, nested=True) as run:
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
                search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)

                t0 = time.time()
                search.fit(X_train, y_train)
                train_time = time.time() - t0
                best = search.best_estimator_

                # ---- params
                best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
                mlflow.log_params(best_params)
                mlflow.log_param("model_type", name)
                mlflow.set_tag("model_type", name)

                # ---- metrics
                metrics = evaluate(best, X_test, y_test)
                cv_idx = search.best_index_
                metrics["cv_f1_mean"] = float(search.cv_results_["mean_test_score"][cv_idx])
                metrics["cv_f1_std"] = float(search.cv_results_["std_test_score"][cv_idx])
                metrics["train_time_sec"] = train_time
                mlflow.log_metrics(metrics)

                # ---- artifacts (plots + text report)
                mlflow.log_figure(plot_confusion(best, X_test, y_test, name), "plots/confusion_matrix.png")
                mlflow.log_figure(plot_roc(best, X_test, y_test, name), "plots/roc_curve.png")
                plt.close("all")
                report = classification_report(
                    y_test, best.predict(X_test), target_names=["benign", "malignant"]
                )
                mlflow.log_text(report, "classification_report.txt")

                # ---- model artifact + registration (creates a new version)
                signature = infer_signature(X_train, best.predict_proba(X_train))
                info = mlflow.sklearn.log_model(
                    sk_model=best,
                    name="model",
                    signature=signature,
                    input_example=X_train.head(3),
                    registered_model_name=REGISTERED_MODEL,
                    # MLflow saves sklearn models with skops (safe format) and asks us
                    # to explicitly trust tree internals - fine, we trained them ourselves
                    skops_trusted_types=["sklearn.tree._tree.Tree"],
                )
                version = str(info.registered_model_version)
                client.set_model_version_tag(REGISTERED_MODEL, version, "model_type", name)
                client.set_model_version_tag(REGISTERED_MODEL, version, "f1", f"{metrics['f1']:.4f}")
                client.update_model_version(
                    REGISTERED_MODEL, version,
                    description=f"{name} | params={best_params} | test F1={metrics['f1']:.4f}",
                )

                leaderboard.append({"model": name, "version": version, "run_id": run.info.run_id,
                                    **{k: round(v, 4) for k, v in metrics.items()}})
                print(f"{name:<20} v{version:<3} f1={metrics['f1']:.4f} "
                      f"recall={metrics['recall']:.4f} auc={metrics['roc_auc']:.4f} "
                      f"params={best_params}")

        # ---- pick champion: best F1, ties broken by ROC-AUC
        lb = pd.DataFrame(leaderboard).sort_values(["f1", "roc_auc"], ascending=False)
        champ = lb.iloc[0]
        client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, champ["version"])
        client.update_registered_model(
            REGISTERED_MODEL,
            description="Breast cancer (malignant vs benign) classifiers. "
                        "Alias 'champion' = best test F1 from the latest training session.",
        )

        mlflow.log_text(lb.to_csv(index=False), "leaderboard.csv")
        mlflow.log_param("champion_model", champ["model"])
        mlflow.log_param("champion_version", champ["version"])
        mlflow.log_metric("champion_f1", float(champ["f1"]))

        # leaderboard chart on the parent run
        fig, ax = plt.subplots(figsize=(7, 4))
        lb.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot.bar(ax=ax)
        ax.set_ylim(0.85, 1.0)
        ax.set_title("Test metrics by model")
        ax.set_ylabel("score")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        mlflow.log_figure(fig, "plots/leaderboard.png")
        plt.close(fig)

        # also keep a plain model.joblib (the Practical Task 6 artifact / API fallback)
        champ_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}/{champ['version']}")
        joblib.dump(champ_model, "model.joblib")
        mlflow.log_artifact("model.joblib")

    print("\nLeaderboard:\n", lb[["model", "version", "accuracy", "recall", "f1", "roc_auc"]].to_string(index=False))
    print(f"\nChampion: {champ['model']} (version {champ['version']}) -> alias '{CHAMPION_ALIAS}'")
    print(f"Parent run: {parent.info.run_id}")


if __name__ == "__main__":
    main()
