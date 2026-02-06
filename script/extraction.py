import cv2
import os
from pathlib import Path

DATA_DIR = Path("data/padel_tracker_100")
LABELS_DIR = Path("data/padel_tracker_100/labels")

# info per dataset (name, video and annotation for ball player and shot)
DATASETS = [
    {
        "name": "2022_BCN_FinalF_1_sample",
        "video": DATA_DIR.joinpath("2022_BCN_FinalF_1_sample.mp4"),
        "annotations": {
            "ball": {
                "path": LABELS_DIR.joinpath("2022_BCN_FinalF_1_ball.json"),
                "format": "coco_bbox",
            },
            "player": {
                "path": LABELS_DIR.joinpath("2022_BCN_FinalF_1_pose.json"),
                "format": "coco_keypoints",
            },
            "shot": {"path": LABELS_DIR.joinpath("2022_BCN_FinalF_1_shots.csv"), "format": "custom"},
        },
    },
    {
        "name": "2022_BCN_FinalF_1",
        "video": DATA_DIR.joinpath("2022_BCN_FinalF_1.mp4"),
        "annotations": {
            "ball": {
                "path": LABELS_DIR.joinpath("2022_BCN_FinalF_1_ball.json"),
                "format": "coco_bbox",
            },
            "player": {
                "path": LABELS_DIR.joinpath("2022_BCN_FinalF_1_pose.json"),
                "format": "coco_keypoints",
            },
            "shot": {"path": LABELS_DIR.joinpath("2022_BCN_FinalF_1_shots.csv"), "format": "custom"},
        },
    },
    {
        "name": "2022_BCN_FinalM_1",
        "video": DATA_DIR.joinpath("2022_BCN_FinalM_1.mp4"),
        "annotations": {
            "ball": {
                "path": LABELS_DIR.joinpath("2022_BCN_FinalM_1_ball.json"),
                "format": "coco_bbox",
            },
            "player": {
                "path": LABELS_DIR.joinpath("2022_BCN_FinalM_1_pose.json"),
                "format": "coco_keypoints",
            },
            "shot": {"path": LABELS_DIR.joinpath("2022_BCN_FinalM_1_shots.csv"), "format": "custom"},
        },

    },
]

# extraction
for i in range(len(DATASETS)):
    video_path = DATASETS[i]["video"]
    video_name = DATASETS[i]["name"]

    output_dir = DATA_DIR.joinpath("frames")
    output_dir = output_dir.joinpath(video_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # salva ogni frame
        frame_filename = output_dir / f"frame_{frame_count:06d}.png"
        cv2.imwrite(str(frame_filename), frame)
        frame_count += 1

    cap.release()
    print(f"{frame_count} frames")