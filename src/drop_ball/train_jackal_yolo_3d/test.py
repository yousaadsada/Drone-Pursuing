import cv2
import os
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/train_drone_yolo_3d/pose_training_run4/weights/best.pt')  # Replace with the path to your best.pt file

def draw_cube(image, keypoints, color=(0, 255, 0), thickness=2):
    """
    Draws an 8-keypoint cube on the image.
    Args:
        image: The image on which to draw.
        keypoints: List of keypoints (8 keypoints in [x, y] format).
        color: Color of the cube lines (default: green).
        thickness: Thickness of the lines.
    """
    # Define the edges of the cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom square
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top square
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical lines
    ]

    for start, end in edges:
        start_point = tuple(map(int, keypoints[start][:2]))  # (x, y)
        end_point = tuple(map(int, keypoints[end][:2]))      # (x, y)
        cv2.line(image, start_point, end_point, color, thickness)

def process_images(input_dir, output_dir):
    """
    Process all images in the input directory, detect keypoints, and save the results in the output directory.
    Args:
        input_dir: Path to the directory containing input images.
        output_dir: Path to the directory where processed images will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file is an image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not load image {input_path}. Skipping.")
                continue

            # Perform inference
            results = model(image)
            if not results or not hasattr(results[0], 'keypoints'):
                print(f"Error: Unable to detect keypoints in {input_path}. Skipping.")
                continue

            # Extract keypoints as a tensor
            keypoints_tensor = results[0].keypoints.xy  # Extract xy coordinates
            keypoints = keypoints_tensor.cpu().numpy()[0]  # Convert to NumPy array

            if keypoints.shape[0] != 8:
                print(f"Error: Expected 8 keypoints in {input_path}, but got {keypoints.shape[0]}. Skipping.")
                continue

            # Draw the cube using the extracted keypoints
            draw_cube(image, keypoints)

            # Save the processed image
            cv2.imwrite(output_path, image)
            print(f"Processed and saved: {output_path}")

def main():
    # Input and output directories
    input_dir = '/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/parrot_fig_test'  # Path to input directory
    output_dir = '/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/output_results'  # Path to output directory

    # Process all images
    process_images(input_dir, output_dir)

if __name__ == '__main__':
    main()
