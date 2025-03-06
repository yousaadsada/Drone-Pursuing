from ultralytics import YOLO
import os
import torch


def prepare_and_train_yolo():
    """
    Prepares YOLOv8 for training using the labeled drone dataset and trains the model.
    """
    # Define paths
    data_yaml_path = "/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/train_drone_yolo/data.yaml"  # Path to your dataset YAML file
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
        project="runs/detect",  # Directory to save results
        name="drone_detection",  # Experiment name
    )

    print("Training complete. Best weights saved in runs/detect/drone_detection/weights/best.pt")


def evaluate_model():
    """
    Evaluates the trained YOLO model on the validation dataset.
    """
    model_path = "runs/detect/drone_detection/weights/best.pt"  # Path to the best-trained model weights
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


def test_inference(input_dir, output_dir):
    """
    Tests inference using the trained YOLO model on new images and calculates the average confidence score.
    """
    import os
    import torch
    from ultralytics import YOLO

    model_path = "/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/train_drone_yolo_2d/runs/detect/drone_detection/weights/best.pt"  # Path to the best-trained model weights
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_confidence = 0
    total_detections = 0

    for filename in os.listdir(input_dir):
        # Check if the file is an image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Using device: {device}")
            print("Loading trained model for inference...")
            model = YOLO(model_path)

            print(f"Running inference on {input_path}...")
            results = model(input_path, device=device)

            print("Inference complete. Showing results...")
            for result in results:
                result.save(filename=output_path)  # Save output to the specified path

                # Extract confidence scores safely
                if result.boxes is not None and len(result.boxes) > 0:
                    confidences = result.boxes.conf.tolist()  # Use .conf to access confidence directly
                    total_confidence += sum(confidences)
                    total_detections += len(confidences)

            print(f"Saved result to {output_path}")

    # Calculate average confidence score
    if total_detections > 0:
        average_confidence = total_confidence / total_detections
        print(f"\nAverage Confidence Score: {average_confidence:.4f}")
    else:
        print("\nNo detections found.")



if __name__ == "__main__":
    # Uncomment one of the following functions based on your workflow:
    
    # # 1. Train the YOLO model
    # prepare_and_train_yolo()

    # # 2. Evaluate the YOLO model
    # evaluate_model()

    # 3. Run inference with the YOLO model

    input_dir = '/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/fig/parrot_fig_test'  # Path to input directory
    output_dir = '/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/fig/output_results_2d'  # Path to output directory
    
    test_inference(input_dir, output_dir)
