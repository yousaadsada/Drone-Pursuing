import os
import json
import cv2
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances

# Function to create COCO-style annotations
def create_coco_annotations(img_files, labels, img_dir, output_file):
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

    annotation_id = 1
    for idx, (img_file, label) in enumerate(zip(img_files, labels)):
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")
        height, width, _ = img.shape

        annotations["images"].append({
            "file_name": img_file,
            "id": idx + 1,
            "width": width,
            "height": height
        })

        keypoints = []
        for i in range(0, len(label), 2):
            x, y = label[i], label[i + 1]
            visibility = 2 if (x >= 0 and y >= 0) else 0
            keypoints.extend([x, y, visibility])

        bbox_x = min([kp for kp in label[::2] if kp >= 0])
        bbox_y = min([kp for kp in label[1::2] if kp >= 0])
        bbox_w = max([kp for kp in label[::2] if kp >= 0]) - bbox_x
        bbox_h = max([kp for kp in label[1::2] if kp >= 0]) - bbox_y

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

    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=4)

# Split dataset and create train/val annotations
def split_and_create_dataset(img_dir, label_file, dataset_dir, split_ratio=0.8):
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')])
    labels = np.loadtxt(label_file, delimiter=',', skiprows=1)
    labels = labels.reshape(-1, 16)

    train_files, val_files, train_labels, val_labels = train_test_split(
        img_files, labels, test_size=(1 - split_ratio), random_state=42
    )

    for file in train_files:
        shutil.copy(os.path.join(img_dir, file), os.path.join(train_dir, file))
    for file in val_files:
        shutil.copy(os.path.join(img_dir, file), os.path.join(val_dir, file))

    create_coco_annotations(train_files, train_labels, train_dir, os.path.join(dataset_dir, "train_annotations.json"))
    create_coco_annotations(val_files, val_labels, val_dir, os.path.join(dataset_dir, "val_annotations.json"))

# Register datasets with Detectron2
# Register datasets with Detectron2
def register_datasets(dataset_dir):
    # Register train dataset
    register_coco_instances(
        "drone_train", 
        {}, 
        os.path.join(dataset_dir, "train_annotations.json"), 
        os.path.join(dataset_dir, "train")
    )
    MetadataCatalog.get("drone_train").keypoint_names = [
        "front_top_left", "front_top_right", "front_bottom_left", "front_bottom_right",
        "rear_top_left", "rear_top_right", "rear_bottom_left", "rear_bottom_right"
    ]
    MetadataCatalog.get("drone_train").keypoint_flip_map = [
        ("front_top_left", "front_top_right"), 
        ("front_bottom_left", "front_bottom_right"),
        ("rear_top_left", "rear_top_right"),
        ("rear_bottom_left", "rear_bottom_right")
    ]

    # Register validation dataset
    register_coco_instances(
        "drone_val", 
        {}, 
        os.path.join(dataset_dir, "val_annotations.json"), 
        os.path.join(dataset_dir, "val")
    )
    MetadataCatalog.get("drone_val").keypoint_names = [
        "front_top_left", "front_top_right", "front_bottom_left", "front_bottom_right",
        "rear_top_left", "rear_top_right", "rear_bottom_left", "rear_bottom_right"
    ]
    MetadataCatalog.get("drone_val").keypoint_flip_map = [
        ("front_top_left", "front_top_right"), 
        ("front_bottom_left", "front_bottom_right"),
        ("rear_top_left", "rear_top_right"),
        ("rear_bottom_left", "rear_bottom_right")
    ]

# Train Detectron2 model
def train_model(dataset_dir, output_dir):
    cfg = get_cfg()
    cfg.merge_from_file("/home/yousa/anafi_simulation/src/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("drone_train",)
    cfg.DATASETS.TEST = ("drone_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = ""
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = output_dir

    os.makedirs(output_dir, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return trainer

# Evaluate model
def evaluate_model(cfg, trainer):
    evaluator = COCOEvaluator("drone_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "drone_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))

# Run inference
def run_inference(cfg, img_path, metadata):
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(img_path)
    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Drone Keypoints", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)

# Main Script
if __name__ == "__main__":
    img_dir = "/home/yousa/anafi_simulation/data/keypoints/img"
    label_file = "/home/yousa/anafi_simulation/data/keypoints/point/key_points.csv"
    dataset_dir = "/home/yousa/anafi_simulation/data/keypoints/dataset"
    output_dir = "/home/yousa/anafi_simulation/output/drone_keypoints"

    split_and_create_dataset(img_dir, label_file, dataset_dir, split_ratio=0.8)
    register_datasets(dataset_dir)
    trainer = train_model(dataset_dir, output_dir)

    cfg = get_cfg()
    cfg.merge_from_file("/home/yousa/anafi_simulation/src/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TEST = ("drone_val",)
    cfg.MODEL.WEIGHTS = ""
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8
    cfg.OUTPUT_DIR = output_dir

    drone_metadata = MetadataCatalog.get("drone_train")
    evaluate_model(cfg, trainer)
    run_inference(cfg, "/path/to/test_image.jpg", drone_metadata)