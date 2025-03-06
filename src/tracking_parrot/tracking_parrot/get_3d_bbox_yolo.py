import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from anafi_msg.msg import KpYolo
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
import pyautogui

Display = True

#NODE CLASS
class BBox(Node):
    def __init__(self):
        super().__init__('af_3D_bbox')
        self.running = True
        self.bridge = CvBridge()
        self.image_with_cube = None
        self.screen_width, self.screen_height = pyautogui.size()

        #KP RCNN MODEL
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = YOLO('/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/train_drone_yolo_3d/pose_training_run4/weights/best.pt')
        self.model.to(self.device)
        #self.get_logger().info(f'Current device: {self.device}')

        #PUBLISHERS / SUBSCRIBERS
        self.frame_sub = self.create_subscription(Image,'anafi/frames', self.frame_callback, 1)
        self.publish_keypoints = self.create_publisher(KpYolo, '/keypoints',1)

        self.key_points = KpYolo()

        self.show_img_thread = threading.Thread(target = self.show_img_thread_callback)
        self.show_img_thread.daemon = True
        self.show_img_thread.start()


    def draw_cube(self, image, keypoints):

        edges = [
            (0, 1), (1, 3), (2, 3), (2, 0),  # Bottom face
            (4, 5), (5, 7), (6, 7), (6, 4),  # Top face
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


    def frame_callback(self, msg):
        self.key_points.target = False
        try:
            # Convert ROS2 Image message to OpenCV image (NumPy array)
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        if image is None:
            print(f"Error: No image input.")
            return
        
        results = self.model(image)
        if not results or not hasattr(results[0], 'keypoints'):
            print(f"Error: Unable to detect keypoints.")
            return

        keypoints_tensor = results[0].keypoints.xy  # Extract xy coordinates
        keypoints = keypoints_tensor.cpu().numpy()[0]  # Convert to NumPy array

        if keypoints.shape[0] != 8:
            print(f"Error: Expected 8 keypoints, but got {keypoints.shape[0]}. Skipping.")
            return
        
        self.image_with_cube = self.draw_cube(image, keypoints)


        keypoints_flat = keypoints.flatten().tolist()
        self.key_points.target = True
        self.key_points.keypoints.data = keypoints_flat
        self.publish_keypoints.publish(self.key_points)
        

    def show_img_thread_callback(self):
        while self.running:
            if self.image_with_cube is not None:
                new_width = self.screen_width // 2
                new_height = self.screen_height // 2
                
                cv2.namedWindow("Image with keypoints cube", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Image with keypoints cube", new_width, new_height)
                
                cv2.imshow("Image with keypoints cube", self.image_with_cube)
                cv2.waitKey(1)
            time.sleep(0.1)

        


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