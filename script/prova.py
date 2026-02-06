from ultralytics import YOLO
import time

model = YOLO("runs/detect/train10_960/weights/best.pt")

start = time.time()
metrics = model.val(
    data="data/padel_tracker_100/padel_data/data.yaml",
    split="test",
    imgsz=960,
    name="test10_960"
)
end = time.time()
print(f"tempo {end-start:.4f}")