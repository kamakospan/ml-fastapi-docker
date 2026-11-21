# ML Model Deployment with FastAPI and Docker

Practical Task 6 — Machine Learning (Fall 2026)
Author: Kamazhay Kospan

A Random Forest classifier trained on the **Iris** dataset, served as a REST API with **FastAPI** and packaged in a **Docker** container. Requests and responses are in **JSON**, and you can test everything in the browser through **Swagger UI** at `/docs`.

## Project structure

```
ml-fastapi-docker/
├── main.py            # FastAPI app (GET /, POST /predict)
├── train.py           # trains the model and saves model.joblib
├── model.joblib       # saved model artifact (model + class names)
├── requirements.txt   # Python dependencies (pinned versions)
├── Dockerfile         # container instructions
├── .dockerignore
└── README.md
```

## Model

| | |
|---|---|
| Dataset | Iris (150 samples, 4 features, 3 classes), from `sklearn.datasets` |
| Algorithm | `RandomForestClassifier(n_estimators=100, random_state=42)` |
| Split | 80 / 20, stratified |
| Test accuracy | 0.90 |

Features (all in cm): `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
Classes: `0 = setosa`, `1 = versicolor`, `2 = virginica`.

## API endpoints

### `GET /`
Checks that the API is running.

```json
{"message": "ML API is running"}
```

### `POST /predict`
Takes the 4 features as JSON and returns the predicted class.

**Request body**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response**
```json
{
  "predicted_class": 0,
  "predicted_species": "setosa",
  "probabilities": {"setosa": 1.0, "versicolor": 0.0, "virginica": 0.0}
}
```

The API validates input: a missing field, a non-number or a value ≤ 0 returns `422 Unprocessable Entity` with an explanation.

### `GET /docs`
Swagger UI, the interactive documentation that FastAPI generates automatically.

---

## Option A: Run locally (Python + Uvicorn)

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates model.joblib)
python train.py

# 4. Start the API
uvicorn main:app --reload
```

Open http://127.0.0.1:8000/docs

## Option B: Run with Docker

### Dockerfile explained

| Instruction | What it does |
|---|---|
| `FROM python:3.11-slim` | Python base image |
| `WORKDIR /app` | Sets the working directory inside the container |
| `COPY requirements.txt .` + `RUN pip install ...` | Installs dependencies (in their own cached layer) |
| `COPY . .` | Copies the project files into the container |
| `RUN python train.py` | Trains the model inside the image, so the model matches the installed scikit-learn version |
| `EXPOSE 8000` | Documents the port the app listens on |
| `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]` | Starts FastAPI with Uvicorn. `0.0.0.0` makes it reachable from outside the container |

### 1. Build the image

```bash
docker build -t ml-fastapi-docker .
```

The build should end with `naming to docker.io/library/ml-fastapi-docker` and no errors. Check it:

```bash
docker images
```

### 2. Run the container

```bash
docker run -d -p 8000:8000 --name ml-api ml-fastapi-docker
```

`-p 8000:8000` maps port 8000 on your machine to port 8000 in the container.

### 3. Test

- Browser: http://localhost:8000/ and http://localhost:8000/docs
- Terminal:

```bash
curl http://localhost:8000/

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}'
# -> {"predicted_class":2,"predicted_species":"virginica",...}
```

### 4. Stop and remove

```bash
docker stop ml-api
docker rm ml-api
```

## Testing in Swagger UI

1. Open http://localhost:8000/docs
2. **GET /** → *Try it out* → *Execute*. You should get `{"message": "ML API is running"}`.
3. **POST /predict** → *Try it out*. The request body is pre-filled with an example → *Execute*.
4. Change the values to try other flowers:

| Input (sl, sw, pl, pw) | Expected |
|---|---|
| 5.1, 3.5, 1.4, 0.2 | setosa |
| 6.0, 2.7, 4.2, 1.3 | versicolor |
| 6.7, 3.0, 5.2, 2.3 | virginica |

## What this demonstrates

- **The model is an artifact.** `train.py` produces `model.joblib`, and `main.py` only loads it, once at startup. Training and serving are separate.
- **Model serving.** A client sends JSON, the API validates it with Pydantic, runs `model.predict`, and returns JSON.
- **Reproducible deployment.** The Docker image contains the code, the pinned dependencies and the runtime config, so the service runs the same way on any machine.
