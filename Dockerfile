# 1. Python base image
FROM python:3.11-slim

# 2. Working directory inside the container
WORKDIR /app

# 3. Install dependencies first (cached layer if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy project files (code + model.joblib)
COPY . .

# 5. Train the model inside the image so its version always matches the installed scikit-learn
RUN python train.py

# 6. Expose the API port
EXPOSE 8000

# 7. Run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
