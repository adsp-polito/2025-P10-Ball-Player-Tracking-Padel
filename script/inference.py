from ultralytics import YOLO
import time

# selezione del modello
model = YOLO("yolo11n.pt")

start = time.time()
metrics = model.val(
    data="data/padel_tracker_100/padel_data/data.yaml",         # file yaml da dove prende immagini e label
    split="test",           # split che consideriamo del file yaml
    imgsz=640,              # imgsz che vogliamo usare
    name="del",             # nome della cartella che verrà salvata in runs/
)
end = time.time()

print("imgsz:640")
print(f"tempo: {end-start:.4f} secondi\n")

names = ['person', 'sports ball']
P = metrics.box.p
R = metrics.box.r
mAP = metrics.box.ap50

per_class_metrics = []
for i in range(len(P)):
    per_class_metrics.append({
        "class_name": names[i],
        "precision": float(P[i]),
        "recall": float(R[i]),
        "mAP50": float(mAP[i]),
    })

print(per_class_metrics)
