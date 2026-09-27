import json
from pathlib import Path
def save_detections_json(d, p):
    r={"model":"yolo26n-finetuned","model_version":"v1.0","class_id":"hammer","conf_threshold":0.25,"iou_threshold":0.45,"images":d}
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(r,f,indent=2,ensure_ascii=False)
    print(f"Saved: {p}")
