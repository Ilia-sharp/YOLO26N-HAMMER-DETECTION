import os, random, cv2, numpy as np
from pathlib import Path
P = Path(__file__).parent.parent
random.seed(42); np.random.seed(42)
def draw_hammer(img,x,y,w,h):
    hc=tuple(map(int,np.random.randint(80,140,3))); hc2=tuple(map(int,np.random.randint(120,200,3)))
    cv2.rectangle(img,(x+w//2-int(w*.15)//2,y+h//2-int(h*.7)//2),(x+w//2+int(w*.15)//2,y+h//2+int(h*.7)//2),hc,-1)
    cv2.rectangle(img,(x+w//2-int(w*.6)//2,y+h//6),(x+w//2+int(w*.6)//2,y+h//6+int(h*.3)),hc2,-1)
    return img
def gen():
    bg=tuple(map(int,np.random.randint(200,255,3))); img=np.full((640,640,3),bg,dtype=np.uint8)
    w,h=random.randint(100,300),random.randint(150,400); x,y=random.randint(50,640-w-50),random.randint(50,640-h-50)
    img=draw_hammer(img,x,y,w,h)
    return img,f"0 {(x+w/2)/640:.6f} {(y+h/2)/640:.6f} {w/640:.6f} {h/640:.6f}"
for d in ["data/dataset/images/train","data/dataset/images/val","data/dataset/labels/train","data/dataset/labels/val","data/samples"]:
    (P/d).mkdir(parents=True,exist_ok=True)
for i in range(32):
    img,l=gen(); cv2.imwrite(str(P/f"data/dataset/images/train/hammer_{i:04d}.jpg"),img)
    (P/f"data/dataset/labels/train/hammer_{i:04d}.txt").write_text(l)
for i in range(8):
    img,l=gen(); cv2.imwrite(str(P/f"data/dataset/images/val/hammer_{i:04d}.jpg"),img)
    (P/f"data/dataset/labels/val/hammer_{i:04d}.txt").write_text(l)
for i in range(5):
    img,_=gen(); cv2.imwrite(str(P/f"data/samples/test_{i:02d}.jpg"),img)
(P/"data/dataset/data.yaml").write_text(f"path: {P/'data/dataset'}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['hammer']\n")
print("Done: 32 train + 8 val + 5 test")
