
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and exported champion model.
COPY src/ ./src/
COPY docker_model/ ./docker_model/

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/docker_model

EXPOSE 8000

CMD ["uvicorn", "mp081.app:app", "--host", "0.0.0.0", "--port", "8000"]
