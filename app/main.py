# -*- coding: utf-8 -*-
"""
YOLO26n Hammer Detection - FastAPI Web Service
Student #16 | Class: hammer | LR2
"""

import sys
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(
    title="YOLO26n Hammer Detection",
    description="Минималистичный сервис детекции молотков",
    version="1.0.0"
)

# Статические файлы и шаблоны
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")),
    name="static"
)
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница с минималистичным UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/health")
async def health_check():
    """Health check endpoint для Render.com"""
    return {
        "status": "ok",
        "model": "yolo26n",
        "class": "hammer",
        "student": "#16"
    }


@app.get("/api/info")
async def model_info():
    """Информация о модели."""
    return {
        "model": "YOLO26n",
        "class": "hammer",
        "nc": 1,
        "device": "cpu",
        "imgsz": 416,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45
    }


@app.post("/api/detect")
async def detect_hammer(file: UploadFile = File(...)):
    """Детекция молотка на загруженном изображении."""
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO

        # Читаем файл
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Неверный формат изображения")

        # Загружаем модель
        weights_path = PROJECT_ROOT / "data" / "weights" / "best.pt"
        if not weights_path.exists():
            weights_path = "yolo26n.pt"

        model = YOLO(str(weights_path))

        # Инференс
        results = model.predict(
            source=img,
            conf=0.25,
            iou=0.45,
            imgsz=416,
            device="cpu",
            verbose=False
        )

        # Собираем детекции
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    class_name = result.names[cls_id]

                    if class_name == "hammer":
                        xyxy = box.xyxy[0].tolist()
                        detections.append({
                            "class": class_name,
                            "confidence": round(conf, 4),
                            "bbox": [round(x, 2) for x in xyxy]
                        })

        return {
            "status": "success",
            "detections": detections,
            "count": len(detections)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
