# YOLO26n Hammer Detection
**Student #16** | **Class: hammer** | **LR2**

## Quick Start
pip install -r requirements.txt
python scripts/download_weights.py
python scripts/generate_dataset.py
uvicorn app.main:app --reload

## Deploy Render
1. Push to GitHub
2. Connect to Render
3. Blueprint auto-detected

## Commands
python scripts/generate_dataset.py
python scripts/train.py
python -m src.detect --source data/samples --weights data/weights/best.pt
python -m src.detect --source data/samples --baseline

**Teacher:** MaxxxVS | **Tag:** v1.0
