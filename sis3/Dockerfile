# One image for every service (mlflow server, trainer, api, ui) - only the command differs.
# Run alone it behaves like Practical Task 6: a FastAPI container serving model.joblib.
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_RETRIES=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
