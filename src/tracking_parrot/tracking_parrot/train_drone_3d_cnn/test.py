import cv2
import torch
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


def draw_3d_box(image, keypoints):
    """
    Draw a 3D bounding box using 8 keypoints.
    Args:
        image (numpy.ndarray): The image to draw on.
        keypoints (numpy.ndarray): Array of 8 keypoints in the format [[x, y, v], ...].
                                   `v` indicates visibility: 2 = visible, 1 = occluded, 0 = not labeled.
    Returns:
        numpy.ndarray: The image with the 3D bounding box drawn.
    """
    # Define connections for a cube (3D bounding box)
    connections = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Front face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Rear face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connections between front and rear
    ]

    for start, end in connections:
        # Ensure the keypoints are visible before drawing
        if keypoints[start, 2] > 0 and keypoints[end, 2] > 0:
            start_point = tuple(map(int, keypoints[start, :2]))
            end_point = tuple(map(int, keypoints[end, :2]))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

    return image


def main():
    # Path to the trained model and test image
    model_weights = "/home/yousa/anafi_simulation/output/drone_keypoints/model_final.pth"
    test_image_path = "/home/yousa/anafi_simulation/data/keypoints/dataset/train/8.jpg"

    # Load the Detectron2 configuration
    cfg = get_cfg()
    cfg.merge_from_file("/home/yousa/anafi_simulation/src/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8
    cfg.MODEL.DEVICE = "cuda"  # Use GPU for inference

    # Set metadata for visualization
    drone_metadata = MetadataCatalog.get("drone_train")  # Use the same dataset metadata as in training

    # Create a predictor
    predictor = DefaultPredictor(cfg)

    # Load the test image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Cannot load image from {test_image_path}")
        return

    # Perform inference
    outputs = predictor(image)
    print(outputs)
    instances = outputs["instances"].to("cpu")
    keypoints = instances.pred_keypoints.numpy()  # Extract keypoints


    # Draw 3D bounding box for each detected instance
    for kp in keypoints:
        image = draw_3d_box(image, kp)

    # Visualize the output with Detectron2's Visualizer (optional)
    visualizer = Visualizer(image[:, :, ::-1], metadata=drone_metadata, scale=0.8)
    vis_image = visualizer.draw_instance_predictions(instances).get_image()[:, :, ::-1]

    # Display the output
    cv2.imshow("Drone Detection with 3D Box", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the output image
    output_image_path = "/home/yousa/anafi_simulation/output/test_image_with_3d_box.jpg"
    cv2.imwrite(output_image_path, image)
    print(f"Saved output image to {output_image_path}")


if __name__ == "__main__":
    main()
