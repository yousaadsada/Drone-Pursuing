from ultralytics import YOLO
import cv2
import os

# Define paths
model_path = "/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo/runs/detect/yolo_detection/weights/best.pt"
fig_path = "/home/yousa/anafi_simulation/src/drop_ball/jackal_fig/yolo_2dbbox_test"
save_dir = "/home/yousa/anafi_simulation/src/drop_ball/jackal_fig/yolo_2dbbox_results"

# Load the YOLO model
model = YOLO(model_path)

# Ensure the output directory exists
os.makedirs(save_dir, exist_ok=True)

index = 0
# Iterate through all images in the figure directory
for filename in os.listdir(fig_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
        image_path = os.path.join(fig_path, filename)

        # Read the image
        img = cv2.imread(image_path)


        # Run inference
    results = model(img)

    # Draw bounding boxes and refine results
    for result in results:
        if result.boxes:
            for box in result.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract class and confidence
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[cls]

                # Refine results: focus only on 'drone' class if available
                if label.lower() == 'jackal' and confidence >= 0.5:  # Replace with the actual drone class name
                    # Draw the bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"{label} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    image_save_path = os.path.join(save_dir, f'{index}.jpg')
                    cv2.imwrite(image_save_path, img)
                    print(f"Drone detected: {label} with confidence {confidence:.2f}")
                    index += 1
                else:
                    print(f"Other object detected: {label} with confidence {confidence:.2f}")


print("Testing complete. Results saved in:", save_dir)
