from ultralytics import YOLO
import os
import torch


def prepare_and_train_yolo():
    """
    Prepares YOLOv8 for training using the labeled drone dataset and trains the model.
    """
    # Define paths
    data_yaml_path = "/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo/data.yaml"  # Path to your dataset YAML file
    model_type = "yolov8n.pt"  # Use 'yolov8n.pt' (nano), 'yolov8s.pt' (small), etc.

    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"{data_yaml_path} not found. Please create a YAML file for your dataset.")

    # Load a pre-trained YOLOv8 model
    print("Loading pre-trained YOLO model...")
    model = YOLO(model_type)

    # Train the model
    print("Starting training...")
    model.train(
        data=data_yaml_path,  # Path to dataset configuration file
        epochs=200,  # Number of epochs
        imgsz=640,  # Image size
        batch=16,  # Batch size
        workers=4,  # Number of dataloader workers
        device=device,  # Specify the device ('cuda' for GPU, 'cpu' for CPU)
        project="/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo/runs/detect",  # Directory to save results
        name="yolo_detection",  # Experiment name
    )

    print("Training complete. Best weights saved in runs/detect/drone_detection/weights/best.pt")


def evaluate_model():
    """
    Evaluates the trained YOLO model on the validation dataset.
    """
    model_path = "/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo/runs/detect/drone_detection/weights/best.pt"  # Path to the best-trained model weights
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")

    print(f"Using device: {device}")
    print("Loading trained model for evaluation...")
    model = YOLO(model_path)

    print("Starting evaluation...")
    results = model.val(device=device)
    print("Evaluation complete. Metrics:")
    print(results)


def test_inference():
    """
    Tests inference using the trained YOLO model on new images.
    """
    model_path = "/home/yousa/anafi_simulation/runs/detect/drone_detection2/weights/best.pt"  # Path to the best-trained model weights
    test_image_path = "/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/fig/drone.png"  # Replace with the path to your test image
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")

    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"{test_image_path} not found. Provide a test image.")

    print(f"Using device: {device}")
    print("Loading trained model for inference...")
    model = YOLO(model_path)

    print(f"Running inference on {test_image_path}...")
    results = model(test_image_path, device=device)

    print("Inference complete. Showing results...")
    results.show()  # Visualize results
    print(results)


if __name__ == "__main__":
    # Uncomment one of the following functions based on your workflow:
    
    # 1. Train the YOLO model
    prepare_and_train_yolo()

    # 2. Evaluate the YOLO model
    evaluate_model()

    # 3. Run inference with the YOLO model
    #test_inference()
