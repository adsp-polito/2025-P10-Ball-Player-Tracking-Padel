from ultralytics import YOLO
import json
import os
from glob import glob

# selection of the best configuration
model = YOLO("runs/detect/train10_1280/weights/best.pt") 

# where the images are
base = "data/padel_tracker_100/padel_data/images"
img_dir = ["test/", "val/", "train/"]

all_preds = []

for dir in img_dir:
    source = os.path.join(base, dir, "*")
    images = sorted([f for f in glob(source) if f.lower().endswith(('.png'))])
    
    # consideriamo solo la parte femminile
    for img_path in images:
        if os.path.basename(img_path).split("_")[0] == "F":
            results = model(img_path, retina_masks=True)[0]         # dove yolo genera le bbox
            annotated_frame = results.plot() 

            img_annotations = {
                "image_id": int(os.path.basename(img_path).split("_")[2].split(".")[0]),
                "predictions": []
            }

            # conversione nel formato che ci interessa
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()    
                w = x2 - x1
                h = y2 - y1

                cord = [x1,y1,w,h]
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                img_annotations["predictions"].append({
                    "bbox": cord,
                    "conf": conf,
                    "category_id": cls,
                    "class_name": results.names[cls]
                })  

            all_preds.append(img_annotations)

# salviamo il file nella cartella predictions/
output_json = "predictionsF.json"
with open(output_json, "w") as f:
    json.dump(all_preds, f, indent=4)

# check finale
unique_ids = len(set(item["image_id"] for item in all_preds))
print(f"Numero di image_id univoci: {unique_ids}")

print(f"File salvato: {output_json}")