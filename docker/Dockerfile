# =========================
# Dockerfile
# =========================
# Контейнер для Flask API сервиса прогнозирования дефолта
# =========================

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/

RUN mkdir -p logs

EXPOSE 5000

CMD ["python", "-m", "app.api"]