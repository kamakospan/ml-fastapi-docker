"""Shared data loading so train.py and the API use the exact same split.

Dataset: Breast Cancer Wisconsin (Diagnostic), built into scikit-learn.
569 patients, 30 numeric features computed from a digitized image of a cell nucleus.

Target is flipped so that 1 = malignant (the "positive" class we care about),
0 = benign. That way recall = "share of real cancers the model catches".
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
CLASS_NAMES = {0: "benign", 1: "malignant"}


def load_data():
    ds = load_breast_cancer(as_frame=True)
    X = ds.data
    y = (ds.target == 0).astype(int)  # sklearn: 0 = malignant -> we make it 1
    y.name = "malignant"
    return X, y


def get_split():
    X, y = load_data()
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def feature_stats(X: pd.DataFrame) -> dict:
    """min / max / median per feature, used for UI sliders and to fill missing inputs."""
    return {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "median": float(X[col].median()),
        }
        for col in X.columns
    }
