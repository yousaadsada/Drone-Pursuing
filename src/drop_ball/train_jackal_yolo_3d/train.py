from ultralytics import YOLO
import os
import torch


def train_yolo_keypoint_model(data_yaml, output_dir, epochs=100, imgsz=640, device=0):
    """
    Trains a YOLOv8 model for keypoint detection and saves it to a specified path.

    Args:
        data_yaml (str): Path to the dataset YAML file.
        model_path (str): Path to the YOLOv8 model weights (e.g., yolov8n.pt).
        output_dir (str): Directory where the trained model and logs will be saved.
        epochs (int): Number of training epochs (default is 100).
        imgsz (int): Input image size (default is 640).
        device (int): Device to use: 0 for GPU, -1 for CPU.
    """
    # Load the YOLO model
    model_type = "yolov8n-pose.pt"
    model = YOLO(model_type)
    device = 0 
    # device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")  # Load pretrained YOLO model (e.g., yolov8n.pt)
    
    model.train(               # Specify the 'pose' task for keypoint detection
        data=data_yaml,               # Path to the data.yaml file
        epochs=epochs,                # Number of training epochs
        imgsz=imgsz,  
        batch=16,  # Batch size
        workers=4,  # Number of dataloader workers                # Image size for training
        device=device,                # GPU or CPU
        project=output_dir,           # Custom output directory
        name="pose_training_run"      # Subdirectory for this specific run
    )
    print(f"Training complete! Model saved in: {os.path.join(output_dir, 'keypoint_training_run')}")


if __name__ == "__main__":
    # Paths to the dataset and model
    data_yaml = "/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo_3d/data.yaml"  # Path to the data.yaml file


    # Specify the output directory
    output_dir = "/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo_3d"  # Custom output directory for saving results

    # Training configuration
    epochs = 500      # Number of epochs
    imgsz = 640       # Image size
    device = 0        # Set to 0 for GPU, -1 for CPU

    # Train the model
    print("Starting YOLOv8 training for keypoints...")
    train_yolo_keypoint_model(data_yaml, output_dir, epochs=epochs, imgsz=imgsz, device=device)