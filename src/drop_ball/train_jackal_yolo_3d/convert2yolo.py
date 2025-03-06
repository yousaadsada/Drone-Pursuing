import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import cv2


def normalize(value, max_value):
    """Normalize a value to the range [0, 1]."""
    return value / max_value


def split_csv_and_convert_to_yolo(csv_file, img_dir, output_dir, train_ratio=0.8):
    """
    Splits the CSV and images into train and val datasets, converts them to YOLO format.

    Args:
        csv_file (str): Path to the CSV file.
        img_dir (str): Path to the directory containing images.
        output_dir (str): Path to save the YOLO dataset.
        train_ratio (float): Proportion of data for training (default is 0.8).
    """
    # Output directories
    train_img_dir = os.path.join(output_dir, "images/train")
    val_img_dir = os.path.join(output_dir, "images/val")
    train_label_dir = os.path.join(output_dir, "labels/train")
    val_label_dir = os.path.join(output_dir, "labels/val")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Read CSV and skip the header
    df = pd.read_csv(csv_file, header=0)

    # Assign column names
    num_keypoints = (len(df.columns) - 4) // 2
    column_names = (
        ["x_center", "y_center", "width", "height"] +
        [f"kpt_{i}_{coord}" for i in range(num_keypoints) for coord in ["x", "y"]]
    )
    df.columns = column_names

    # Add an image_id column explicitly as sequential numbers from 0 to 999
    df["image_id"] = range(len(df))

    # Split into train and val sets
    train_df, val_df = train_test_split(df, test_size=1-train_ratio, random_state=42)

    # Function to process subsets
    def process_subset(subset_df, img_output_dir, label_output_dir):
        for _, row in subset_df.iterrows():
            # Extract image ID
            image_id = int(row["image_id"])
            img_path = os.path.join(img_dir, f"{image_id}.jpg")

            # Check if the image exists
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found. Skipping...")
                continue

            # Bounding box
            x_center, y_center, width, height = row.iloc[0:4].astype(float)

            # Keypoints
            keypoints = row.iloc[4:-1].astype(float).values.reshape(-1, 2)

            # Image dimensions
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            # Normalize bounding box
            x_center = normalize(x_center, img_width)
            y_center = normalize(y_center, img_height)
            width = normalize(width, img_width)
            height = normalize(height, img_height)

            # Normalize keypoints
            normalized_keypoints = [(normalize(x, img_width), normalize(y, img_height)) for x, y in keypoints]

            # Flatten the keypoints and add visibility set to 2 for each keypoint
            flattened_keypoints_with_visibility = [coord for kp in normalized_keypoints for coord in (*kp, 2)]


            # Write the YOLO label file
            label_path = os.path.join(label_output_dir, f"{image_id}.txt")
            with open(label_path, "w") as label_file:
                label_line = f"0 {x_center} {y_center} {width} {height} " + " ".join(map(str, flattened_keypoints_with_visibility))
                label_file.write(label_line + "\n")

            # Copy the image to the respective directory
            shutil.copy(img_path, os.path.join(img_output_dir, f"{image_id}.jpg"))

    # Process train and val subsets
    process_subset(train_df, train_img_dir, train_label_dir)
    process_subset(val_df, val_img_dir, val_label_dir)

    print("Dataset conversion and splitting complete!")


if __name__ == "__main__":
    # Paths
    csv_file = "/home/yousa/anafi_simulation/src/drop_ball/jackal_fig/key_point/key_points.csv"  # Path to your CSV file
    img_dir = "/home/yousa/anafi_simulation/src/drop_ball/jackal_fig/image"  # Directory containing images
    output_dir = "/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo_3d/dataset"  # Output directory for YOLO dataset

    # Convert CSV and split dataset/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo_3d
    split_csv_and_convert_to_yolo(csv_file, img_dir, output_dir)
