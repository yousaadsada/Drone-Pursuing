import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from anafi_msg.msg import KpYolo, FrameYolo
import cv2
import torch
import time
import numpy as np
import tkinter as tk
from torchvision.transforms import functional as F
from cv_bridge import CvBridge
from tkinter import ttk
import cv2
import os
import numpy as np
from ultralytics import YOLO
from std_msgs.msg import Float64MultiArray
import threading


Display = True

#NODE CLASS
class BBox(Node):
    def __init__(self):
        super().__init__('af_3D_bbox')
        self.running = True
        self.bridge = CvBridge()
        self.image_with_cube = None

        #KP RCNN MODEL
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = YOLO('/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo_3d/pose_training_run3/weights/best.pt')
        self.model.to(self.device)
        #self.get_logger().info(f'Current device: {self.device}')

        #PUBLISHERS / SUBSCRIBERS
        
        self.frame_2d_sub = self.create_subscription(FrameYolo,'/jackal_frame', self.jackal_frame_callback, 1)
        self.publish_keypoints = self.create_publisher(KpYolo, '/keypoints',1)

        self.jackal_frame = FrameYolo()
        self.key_points = KpYolo()

        self.show_img_thread = threading.Thread(target = self.show_img_thread_callback)
        self.show_img_thread.daemon = True
        self.show_img_thread.start()


    def draw_cube(self, image, keypoints):

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

        # Draw edges on the image
        for start, end in edges:
            pt1 = tuple(map(int, keypoints[start]))
            pt2 = tuple(map(int, keypoints[end]))
            cv2.line(image, pt1, pt2, color=(0, 255, 0), thickness=2)  # Green edges

        # Draw keypoints as circles
        for idx, (x, y) in enumerate(keypoints):
            cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)  # Red dots
            cv2.putText(image, str(idx + 1), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return image


    def jackal_frame_callback(self, msg):
        self.jackal_frame.target = msg.target
        self.jackal_frame.image = msg.image
        self.jackal_frame.x1 = msg.x1
        self.jackal_frame.y1 = msg.y1
        self.jackal_frame.x2 = msg.x2
        self.jackal_frame.y2 = msg.y2

        self.key_points.target = False

        try:
                # Convert ROS2 Image message to OpenCV image (NumPy array)
            image = self.bridge.imgmsg_to_cv2(self.jackal_frame.image, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        if self.jackal_frame.target == True:
            # try:
            #     # Convert ROS2 Image message to OpenCV image (NumPy array)
            #     image = self.bridge.imgmsg_to_cv2(self.jackal_frame.image, desired_encoding='bgr8')
            # except Exception as e:
            #     self.get_logger().error(f"Failed to convert image: {e}")
            #     return
            
            if image is None:
                print(f"Error: No image input.")
                return
            
            height, width, _ = image.shape
            
            # x1 = max(0, int(self.jackal_frame.x1 - 100))
            # y1 = max(0, int(self.jackal_frame.y1 - 100))
            # x2 = min(width, int(self.jackal_frame.x2 + 100))
            # y2 = min(height, int(self.jackal_frame.y2 + 100))
            
            x1 = 0
            y1 = 0
            x2 = width
            y2 = height


            roi_image = image[y1:y2, x1:x2]
        
            results = self.model(roi_image)

            if not results or not hasattr(results[0], 'keypoints'):
                print(f"Error: Unable to detect keypoints.")
                return

            keypoints_tensor = results[0].keypoints.xy  # Extract xy coordinates
            keypoints = keypoints_tensor.cpu().numpy()[0]  # Convert to NumPy array

            if keypoints.shape[0] != 8:
                self.image_with_cube = image
                print(f"Error: Expected 8 keypoints, but got {keypoints.shape[0]}. Skipping.")
                return
        
            keypoints[:, 0] += x1
            keypoints[:, 1] += y1
        
            self.image_with_cube = self.draw_cube(image, keypoints)

            keypoints_flat = keypoints.flatten().tolist()
            self.key_points.target = True
            self.key_points.keypoints.data = keypoints_flat
            # self.publish_keypoints.publish(self.key_points)
        
        else:
            self.image_with_cube = image
        
        self.publish_keypoints.publish(self.key_points)


    def show_img_thread_callback(self):
        while self.running:
            if self.image_with_cube is not None:
                cv2.imshow("Image with keypoints cube", self.image_with_cube)
                cv2.waitKey(1)
            time.sleep(0.075)

        


    #STOP FUNCTION
    def Stop(self):
        
        time.sleep(0.2)
        self.destroy_node()
        rclpy.shutdown()


#MAIN BOOL
def main():
    rclpy.init()
    af_3D_bbox = BBox()

    try:
        while rclpy.ok():
            rclpy.spin_once(af_3D_bbox)
    except:
        af_3D_bbox.Stop()

if __name__ == "__main__":
    main()