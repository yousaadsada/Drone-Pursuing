import os

def rename_images_and_labels(directory, prefix="image_"):
    """
    Renames all images and associated YOLO label files in a directory to a sequential format.

    Args:
        directory (str): Path to the directory containing images and labels.
        prefix (str): Prefix for the renamed files (default: 'image_').
    """
    # Get all image files and YOLO label files
    images = sorted([f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))])
    # labels = sorted([f for f in os.listdir(directory) if f.endswith('.txt')])

    for i, image in enumerate(images):
        # New name for the image
        new_image_name = f"{prefix}{i}.jpg"

        # Rename image
        old_image_path = os.path.join(directory, image)
        new_image_path = os.path.join(directory, new_image_name)
        os.rename(old_image_path, new_image_path)

        # Check if a corresponding label file exists and rename it
        # label_name = os.path.splitext(image)[0] + ".txt"
        # if label_name in labels:
        #     old_label_path = os.path.join(directory, label_name)
        #     new_label_name = f"{prefix}{i}.txt"
        #     new_label_path = os.path.join(directory, new_label_name)
        #     os.rename(old_label_path, new_label_path)

    print(f"Renamed all files in directory: {directory}")


if __name__ == "__main__":
    # Define paths
    train_dir = "/home/yousa/anafi_simulation/src/tracking_parrot/dataset/images/train"
    val_dir = "/home/yousa/anafi_simulation/src/tracking_parrot/dataset/images/val"

    # Rename images and labels in train and val directories
    rename_images_and_labels(train_dir, prefix="train_")
    rename_images_and_labels(val_dir, prefix="val_")
