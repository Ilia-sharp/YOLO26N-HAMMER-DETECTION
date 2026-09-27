import sys
from pathlib import Path
P = Path(__file__).parent.parent
sys.path.insert(0, str(P))
import yaml
from ultralytics import YOLO
with open(P/"configs/training_config.yaml") as f: c=yaml.safe_load(f)
model = YOLO(c.get('model','yolo26n.pt'))
model.train(data=str(P/'data/dataset/data.yaml'),epochs=c.get('epochs',25),batch=c.get('batch',8),imgsz=c.get('imgsz',416),device=c.get('device','cpu'),workers=c.get('workers',0),project=str(P/'runs/detect'),name='train',exist_ok=False,pretrained=True)
w = P/'runs/detect/train/weights/best.pt'
if w.exists():
    import shutil; shutil.copy2(w, P/'data/weights/best.pt')
    print(f"Best: {P/'data/weights/best.pt'}")
