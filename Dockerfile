FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts
COPY data ./data
COPY mlflow.db ./mlflow.db
COPY mlruns ./mlruns

EXPOSE 8000
ENV MODEL_STAGE=Production

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
