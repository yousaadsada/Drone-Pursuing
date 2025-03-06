import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(image_dir, train_dir, val_dir, split_ratio=0.8):
    """
    Splits images in the given directory into train and validation folders.

    Args:
        image_dir (str): Path to the directory containing the images.
        train_dir (str): Path to the directory where training images will be stored.
        val_dir (str): Path to the directory where validation images will be stored.
        split_ratio (float): Ratio of training data (default: 0.8).
    """
    # Create train and validation directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get a list of all image files in the directory
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Split the images into train and validation sets
    train_images, val_images = train_test_split(images, test_size=1 - split_ratio, random_state=42)

    # Move the images to their respective directories
    for img in train_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(train_dir, img))
    for img in val_images:
        shutil.move(os.path.join(image_dir, img), os.path.join(val_dir, img))

    print(f"Dataset split completed!")
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")


if __name__ == "__main__":
    # Define paths
    image_dir = "/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/parrot_fig"
    train_dir = "/home/yousa/anafi_simulation/src/tracking_parrot/dataset/image/train"
    val_dir = "/home/yousa/anafi_simulation/src/tracking_parrot/dataset/image/val"

    # Split dataset
    split_dataset(image_dir, train_dir, val_dir)
