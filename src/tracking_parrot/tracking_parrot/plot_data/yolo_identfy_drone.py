from ultralytics import YOLO
import cv2
import os
import cv2
import torch
from torchvision.transforms import functional as F
from cv_bridge import CvBridge
import numpy as np


save_dir = os.path.join(os.path.expanduser("~"), f'/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/parrot_fig_test_result')
os.makedirs(save_dir, exist_ok=True)
save_dir_3d = os.path.join(os.path.expanduser("~"), f'/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/parrot_fig_test_result_3d')
os.makedirs(save_dir, exist_ok=True)

def detect_drone(image_path, index, save_dir, model_path="/home/yousa/anafi_simulation/runs/detect/drone_detection2/weights/best.pt"):
    # Load the pre-trained YOLO model
    model = YOLO(model_path)

    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Perform inference
    results = model(image)

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
                if label.lower() == 'drone' and confidence >= 0.5:  # Replace with the actual drone class name
                    # Draw the bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        f"{label} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    image_save_path = os.path.join(save_dir, f'{index}.jpg')
                    cv2.imwrite(image_save_path, image)
                    print(f"Drone detected: {label} with confidence {confidence:.2f}")
                else:
                    print(f"Other object detected: {label} with confidence {confidence:.2f}")

    # Display the output
    # cv2.imshow("Drone Detection", image)
    # print("Press any key to close the window...")
    # cv2.waitKey(0)  # Wait indefinitely until a key is pressed
    # cv2.destroyAllWindows()  # Close all OpenCV windows

def detect_drone_3d(image_path, index, save_dir_3d):
    bridge = CvBridge()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load('/home/yousa/anafi_main_ros2/anafi_ros2/src/anafi_ros2/anafi_ros2/Data/frame_kp_vertices.pt')
    model.to(device)

    frame = cv2.imread(image_path)  # Read the image into a frame
    msg = bridge.cv2_to_imgmsg(frame, "bgr8")
    target_frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    img = F.to_tensor(target_frame).to(device)

    model.eval()
    output = model([img])


    def kp_enhancement(kp):
        vertices = kp
        vertices = vertices.reshape(8, 2)

        u0 = vertices[0, 0]
        v0 = vertices[0, 1]
        u1 = vertices[1, 0]
        v3 = vertices[3, 1]
        u4 = vertices[4, 0]
        v4 = vertices[4, 1]
        u5 = vertices[5, 0]
        v7 = vertices[7, 1]
        h1 = v3-v0
        w1 = u1-u0
        h2 = v7-v4
        w2 = u5-u4

        vertices[1, :] = [u0+w1, v0]
        vertices[2, :] = [u0+w1, v0+h1]
        vertices[3, :] = [u0, v0+h1]
        vertices[5, :] = [u4+w2, v4]
        vertices[6, :] = [u4+w2, v4+h2]
        vertices[7, :] = [u4, v4+h2]

        return vertices.reshape(-1)

    target_kp = []
    scores = output[0]['scores'].detach().cpu().numpy()
    filtered_scores = np.where(scores > 0.3)[0]
    if len(filtered_scores)>0:
        max_score = filtered_scores[np.argmax(scores[filtered_scores])]
        kp = output[0]['keypoints'][max_score].detach().cpu().numpy().astype(np.int32)
        kp = kp[:,0:2]
        target_kp = kp_enhancement(kp).reshape(8,2)

    # print(target_kp)


        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Back face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connect front and back faces
        ]

        # Draw the edges on the image
        for edge in edges:
            start_point = tuple(target_kp[edge[0]])
            end_point = tuple(target_kp[edge[1]])
            cv2.line(frame, start_point, end_point, (0, 255, 0), thickness=2)

        # Draw keypoints as circles
        for point in target_kp:
            cv2.circle(frame, tuple(point), radius=5, color=(0, 0, 255), thickness=-1)

        # Convert the image back to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Save the output image
        image_save_path = os.path.join(save_dir_3d, f'{index}.jpg')
        cv2.imwrite(image_save_path, frame_bgr)


if __name__ == "__main__":
    save_dir = os.path.join(os.path.expanduser("~"), f'/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/parrot_fig_test_result')
    os.makedirs(save_dir, exist_ok=True)
    save_dir_3d = os.path.join(os.path.expanduser("~"), f'/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/parrot_fig_test_result_3d')
    os.makedirs(save_dir_3d, exist_ok=True)
    for i in range (0, 44):
        index = i
        detect_drone(f'/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/parrot_fig_test/{i}.jpg', index, save_dir)
        detect_drone_3d(f'/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/parrot_fig_test/{i}.jpg', index, save_dir_3d)