import os
import shutil
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def create_coco_annotations(img_files, labels, img_dir, output_file):
    # Initialize COCO-style annotation structure
    annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "drone",
                "keypoints": [
                    "front_top_left", "front_top_right", "front_bottom_left", "front_bottom_right",
                    "rear_top_left", "rear_top_right", "rear_bottom_left", "rear_bottom_right"
                ],
                "skeleton": []  # Optional: Define keypoint connections
            }
        ]
    }

    annotation_id = 1  # Unique ID for each annotation

    for idx, (img_file, label) in enumerate(zip(img_files, labels)):
        # Image metadata
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        height, width, _ = img.shape

        # Add image entry to COCO JSON
        annotations["images"].append({
            "file_name": img_file,
            "id": idx + 1,
            "width": width,
            "height": height
        })

        # Keypoints and bounding box
        keypoints = []
        for i in range(0, len(label), 2):
            x, y = label[i], label[i + 1]
            visibility = 2 if (x >= 0 and y >= 0) else 0  # Visibility: 2 (visible), 0 (not labeled)
            keypoints.extend([x, y, visibility])

        bbox_x = min([kp for kp in label[::2] if kp >= 0])
        bbox_y = min([kp for kp in label[1::2] if kp >= 0])
        bbox_w = max([kp for kp in label[::2] if kp >= 0]) - bbox_x
        bbox_h = max([kp for kp in label[1::2] if kp >= 0]) - bbox_y

        # Add annotation entry to COCO JSON
        annotations["annotations"].append({
            "image_id": idx + 1,
            "id": annotation_id,
            "keypoints": keypoints,
            "num_keypoints": 8,
            "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
            "category_id": 1,
            "iscrowd": 0
        })

        annotation_id += 1

    # Save JSON file
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)

def split_and_create_dataset(img_dir, label_file, dataset_dir, split_ratio=0.8):
    # Prepare directories
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Load image file names and labels
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
    labels = np.loadtxt(label_file, delimiter=',', skiprows=1)

    # Ensure labels are numpy array and reshape
    labels = labels.reshape(-1, 16)  # 8 keypoints * (x, y) = 16 values per row

    # Split into training and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        img_files, labels, test_size=(1 - split_ratio), random_state=42
    )

    # Copy training files
    for file in train_files:
        shutil.copy(os.path.join(img_dir, file), os.path.join(train_dir, file))

    # Copy validation files
    for file in val_files:
        shutil.copy(os.path.join(img_dir, file), os.path.join(val_dir, file))

    # Create COCO-style annotations
    create_coco_annotations(train_files, train_labels, train_dir, os.path.join(dataset_dir, "train_annotations.json"))
    create_coco_annotations(val_files, val_labels, val_dir, os.path.join(dataset_dir, "val_annotations.json"))

# Example Usage
original_img_dir = "/home/yousa/anafi_simulation/data/keypoints/img"  # Original image directory
label_file = "/home/yousa/anafi_simulation/data/keypoints/point/key_points.csv"  # Original label file
dataset_dir = "/home/yousa/anafi_simulation/data/keypoints/dataset"  # New dataset directory for splits

# Split and create dataset
split_and_create_dataset(original_img_dir, label_file, dataset_dir, split_ratio=0.8)
