FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/weights outputs runs/detect

# 🎯 Скачиваем базовую модель YOLO26n
RUN python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"

# 📸 Генерируем синтетический датасет молотков (32 train + 8 val + 5 test)
RUN python scripts/generate_dataset.py

# 🧠 Обучаем модель на датасете (25 эпох)
RUN python scripts/train.py

# ✅ После обучения файл data/weights/best.pt будет внутри контейнера

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
