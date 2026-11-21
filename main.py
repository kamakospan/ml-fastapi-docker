"""FastAPI service that serves predictions from the trained Iris model."""

from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"

# Load the model artifact once, when the app starts
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
target_names = artifact["target_names"]

app = FastAPI(
    title="Iris ML API",
    description="Predicts the Iris flower species from 4 measurements (cm).",
    version="1.0.0",
)


class IrisFeatures(BaseModel):
    """Input features. Example values are shown in Swagger UI."""

    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


class Prediction(BaseModel):
    predicted_class: int
    predicted_species: str
    probabilities: dict[str, float]


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures):
    x = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    pred = int(model.predict(x)[0])
    proba = model.predict_proba(x)[0]
    return {
        "predicted_class": pred,
        "predicted_species": target_names[pred],
        "probabilities": {
            name: round(float(p), 3) for name, p in zip(target_names, proba)
        },
    }
