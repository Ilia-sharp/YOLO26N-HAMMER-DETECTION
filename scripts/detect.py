import argparse, sys
from pathlib import Path
P = Path(__file__).parent.parent
sys.path.insert(0, str(P))
from ultralytics import YOLO
from src.utils import save_detections_json
def run(model, source, out, conf=0.25, iou=0.45):
    Path(out).mkdir(parents=True,exist_ok=True)
    results = model.predict(source=source,conf=conf,iou=iou,imgsz=416,device="cpu",save=True,project=str(out),name='predictions',exist_ok=True)
    dl=[]
    for r in results:
        ip=Path(r.path); h,w=r.orig_shape; ids=[]
        if r.boxes is not None:
            for b in r.boxes:
                cn=r.names[int(b.cls[0].item())]
                if cn=='hammer': ids.append({"class":cn,"confidence":round(float(b.conf[0].item()),4),"xyxy":[round(x,2) for x in b.xyxy[0].tolist()]})
        dl.append({"file":ip.name,"width":w,"height":h,"detections":ids})
    return dl
if __name__=='__main__':
    pa=argparse.ArgumentParser(); pa.add_argument('--source',default='data/samples'); pa.add_argument('--weights',default='data/weights/best.pt'); pa.add_argument('--out',default='outputs'); pa.add_argument('--baseline',action='store_true')
    a=pa.parse_args()
    if a.baseline:
        m=YOLO('yolo26n.pt'); d=run(m,a.source,str(Path(a.out)/'baseline')); save_detections_json(d,str(Path(a.out)/'baseline/detections.json'))
        wp=P/'data/weights/best.pt'
        if wp.exists():
            m2=YOLO(str(wp)); d2=run(m2,a.source,str(Path(a.out)/'finetuned')); save_detections_json(d2,str(Path(a.out)/'finetuned/detections.json'))
    else:
        m=YOLO(a.weights); d=run(m,a.source,a.out); save_detections_json(d,str(Path(a.out)/'detections.json'))
    print("Done!")
