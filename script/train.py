from ultralytics import YOLO
import time


## train
model = YOLO("yolo11n.pt")                  # modello che vogliamo trainare
model_train = model.train(
    data="data/padel_tracker_100/padel_data/data.yaml",         # percorso del file yaml 
    epochs=20,                      # per quante epoche
    imgsz=960,                      # imgsz che vogliamo usare
    name="train11_960"              # nome cartella che verrà salvata
)

## test
model = YOLO("runs/detect/train11_960/weights/best.pt")         # modello che abbiamo appena trainato, prendiamo la migliore

start = time.time()
metrics = model.val(
    data="data/padel_tracker_100/padel_data/data.yaml",
    split="test",
    imgsz=960,
    name="test10_960_40"
)
end = time.time()

print("imgsz:960")
print(f"tempo: {end-start:.4f}")

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