import os
from ultralytics.data.converter import convert_coco
from os.path import exists
from glob import glob
import shutil
import yaml


#### converto le annotazioni da pixel a percentuali
# temporary directory
source = "data/padel_tracker_100/labels"
temp_json_dir = "data/padel_tracker_100/temporary"
output = "data/padel_tracker_100/formatted_labels"

json_files = [f for f in os.listdir(source) if f.endswith('.json')]

# Move the JSON file into the temporary directory
for file in json_files:
    source_path =os.path.join(source, file)
    dest_path = os.path.join(temp_json_dir, file)

    os.makedirs(temp_json_dir, exist_ok=True)
    os.rename(source_path, dest_path)

    convert_coco(
        labels_dir=temp_json_dir,
        save_dir=os.path.join(output, file.replace('.json', '')),
        #use_keypoints=True,  i keypoints non ci servono
    )

    # Move the file back to its original location
    os.rename(dest_path, source_path)
    os.rmdir(temp_json_dir)


##### unione delle labels nello stesso file
# per la parte maschile
ball_label = "data/padel_tracker_100/formatted_labels/2022_BCN_FinalM_1_ball/labels/2022_BCN_FinalM_1_ball"
pose_label = "data/padel_tracker_100/formatted_labels/2022_BCN_FinalM_1_pose/labels/2022_BCN_FinalM_1_pose"
output_dir = "data/padel_tracker_100/labels_yolo/m"
os.makedirs(output_dir, exist_ok=True)

ball_files_paths = sorted(glob(os.path.join(ball_label, "*.txt")))
pose_files_paths = sorted(glob(os.path.join(pose_label, "*.txt")))

#data\padel_tracker_100\formatted_labels\2022_BCN_FinalF_1_ball\labels\2022_BCN_FinalF_1_ball/frame_000000.txt
max = int(pose_files_paths[len(pose_files_paths)-1].split("/")[6].split(".")[0].split("_")[1])

for i in range(max+1):
    frame_name = f"frame_{i:06d}.txt"
    combined = []

    current_pose_filepath = os.path.join(pose_label, frame_name)
    if os.path.exists(current_pose_filepath):
        with open(current_pose_filepath, 'r') as f_pose:
            file_content = f_pose.read().strip()
            combined.append(file_content)

    current_ball_filepath = os.path.join(ball_label, frame_name)
    if os.path.exists(current_ball_filepath):
        with open(current_ball_filepath, 'r') as f_ball:
            for line in f_ball:
                parts = line.strip().split()
                parts[0] = '32'
                combined.append(' '.join(parts))

    outname = f"M_{frame_name}"
    output_filepath = os.path.join(output_dir, outname)
    content_to_write = '\n'.join(combined)
    with open(output_filepath, 'w') as outfile:
        outfile.write(content_to_write)

# per la parte femminile
ball_label = "data/padel_tracker_100/formatted_labels/2022_BCN_FinalF_1_ball/labels/2022_BCN_FinalF_1_ball"
pose_label = "data/padel_tracker_100/formatted_labels/2022_BCN_FinalF_1_pose/labels/2022_BCN_FinalF_1_pose"
output_dir = "data/padel_tracker_100/labels_yolo/f"
os.makedirs(output_dir, exist_ok=True)

ball_files_paths = sorted(glob(os.path.join(ball_label, "*.txt")))
pose_files_paths = sorted(glob(os.path.join(pose_label, "*.txt")))

max = int(pose_files_paths[len(pose_files_paths)-1].split("/")[6].split(".")[0].split("_")[1])

for i in range(max+1):
    frame_name = f"frame_{i:06d}.txt"
    combined = []

    current_pose_filepath = os.path.join(pose_label, frame_name)
    if os.path.exists(current_pose_filepath):
        with open(current_pose_filepath, 'r') as f_pose:
            file_content = f_pose.read().strip()
            combined.append(file_content)

    current_ball_filepath = os.path.join(ball_label, frame_name)
    if os.path.exists(current_ball_filepath):
        with open(current_ball_filepath, 'r') as f_ball:
            for line in f_ball:
                parts = line.strip().split()
                parts[0] = '32'
                combined.append(' '.join(parts))

    outname = f"F_{frame_name}"
    output_filepath = os.path.join(output_dir, outname)
    content_to_write = '\n'.join(combined)
    with open(output_filepath, 'w') as outfile:
        outfile.write(content_to_write)

#### divisione in train test val
from glob import glob

framesM = sorted(glob("data/padel_tracker_100/frames/2022_BCN_FinalM_1/*.png"))
framesF = sorted(glob("data/padel_tracker_100/frames/2022_BCN_FinalF_1/*.png"))

train = []
test = []
val = []

for f in framesM:
    filename = os.path.basename(f)
    num_frame = int(filename.split("_")[1].split(".")[0])
    if num_frame >= 16298: train.append(f)
    if num_frame >= 22 and num_frame <= 10752: test.append(f)
    if num_frame >= 10753 and num_frame <= 16297: val.append(f)

for f in framesF:
    filename = os.path.basename(f)
    num_frame = int(filename.split("_")[1].split(".")[0])
    if num_frame >= 13882: train.append(f)
    if num_frame <= 9102: test.append(f)
    if num_frame >= 9103 and num_frame <= 13881: val.append(f)

print(len(train), len(test), len(val))

#### definizione dello yaml
# spostiamo le immagini
os.makedirs(os.path.join("data/padel_tracker_100/padel_data", 'images'), exist_ok=True)
os.makedirs(os.path.join("data/padel_tracker_100/padel_data", 'labels'), exist_ok=True)

train_root = "data/padel_tracker_100/padel_data/images/train"
test_root = "data/padel_tracker_100/padel_data/images/test"
val_root = "data/padel_tracker_100/padel_data/images/val"

for split, root in zip([train, test, val], [train_root, test_root, val_root]):
    print(root)
    for frame in split:
        filename = os.path.basename(frame)

        if "2022_BCN_FinalM" in frame:
            name = f"M_{filename}"
        elif "2022_BCN_FinalF" in frame:
            name = f"F_{filename}"

        dst_path = os.path.join(root, name)

        os.makedirs(root, exist_ok=True)

        shutil.move(frame, dst_path)

# spostiamo le labels
train_root = "data/padel_tracker_100/padel_data/labels/train"
test_root = "data/padel_tracker_100/padel_data/labels/test"
val_root = "data/padel_tracker_100/padel_data/labels/val"

for split, root in zip([train, test, val], [train_root, test_root, val_root]):
    print(root)
    for frame in split:
        filename = os.path.basename(frame).split(".")[0]

        if "2022_BCN_FinalM" in frame:
            name = f"M_{filename}.txt"
            gender = 0
        elif "2022_BCN_FinalF" in frame:
            name = f"F_{filename}.txt"
            gender = 1

        dst_path = os.path.join(root, name)

        os.makedirs(root, exist_ok=True)
        if gender: shutil.copy2(f"data/padel_tracker_100/labels_yolo/f/F_{filename}.txt", dst_path)
        else: shutil.copy2(f"data/padel_tracker_100/labels_yolo/m/M_{filename}.txt", dst_path)


#### creazione effettiva dello yaml
dataset_path = 'data/padel_tracker_100/padel_data'

nc_classi = 2
nomi_classi = ['player', 'sports ball']

yaml_filename = 'data.yaml'

yaml_full_path = os.path.join(dataset_path, yaml_filename)

# Struttura
data = {
    'path': dataset_path,

    'train': 'images/train',
    'val': 'images/val',
    'test': 'images/test',

    'nc': nc_classi,
    'names': nomi_classi
}

try:
    with open(yaml_full_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

except Exception as e:
    print("errore")