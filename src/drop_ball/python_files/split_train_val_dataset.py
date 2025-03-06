import os
import shutil

def rename_and_split_dataset(base_dir, output_dir, split_ratio=0.8):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')

    # Check if images and labels directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError("Images or labels directory not found.")

    # List and sort all files
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt') and f != 'classes.txt'])

    # Ensure the number of images and labels match
    if len(image_files) != len(label_files):
        raise ValueError("Number of images and labels do not match!")

    # Create output directories for train and val
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    val_images_dir = os.path.join(output_dir, 'images', 'val')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')

    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Rename files to train_0, train_1, ..., val_0, val_1...
    split_index = int(len(image_files) * split_ratio)

    train_images = image_files[:split_index]
    train_labels = label_files[:split_index]
    val_images = image_files[split_index:]
    val_labels = label_files[split_index:]

    # Move and rename train files
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        new_image_name = f"train_{i}.jpg"
        new_label_name = f"train_{i}.txt"

        shutil.move(os.path.join(images_dir, image), os.path.join(train_images_dir, new_image_name))
        shutil.move(os.path.join(labels_dir, label), os.path.join(train_labels_dir, new_label_name))

    # Move and rename val files
    for i, (image, label) in enumerate(zip(val_images, val_labels)):
        new_image_name = f"val_{i}.jpg"
        new_label_name = f"val_{i}.txt"

        shutil.move(os.path.join(images_dir, image), os.path.join(val_images_dir, new_image_name))
        shutil.move(os.path.join(labels_dir, label), os.path.join(val_labels_dir, new_label_name))

    # Copy classes.txt to both train and val directories
    classes_file = os.path.join(labels_dir, 'classes.txt')
    if os.path.exists(classes_file):
        shutil.copy(classes_file, train_labels_dir)
        shutil.copy(classes_file, val_labels_dir)

    print("Dataset renamed, split, and moved successfully with the correct structure!")

# Example usage
base_dataset_dir = '/home/yousa/anafi_simulation/src/drop_ball/jackal_fig/dataset'
output_dataset_dir = '/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo/dataset'
rename_and_split_dataset(base_dataset_dir, output_dataset_dir)
