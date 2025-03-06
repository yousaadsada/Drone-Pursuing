import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from anafi_msg.msg import FrameYolo
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
class Frame(Node):
    def __init__(self):
        super().__init__('af_2D_bbox')
        self.running = True
        self.bridge = CvBridge()
        self.image_with_cube = None


        #KP RCNN MODEL
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = YOLO('/home/yousa/anafi_simulation/src/drop_ball/train_jackal_yolo/runs/detect/yolo_detection/weights/best.pt')
        self.model.to(self.device)
        #self.get_logger().info(f'Current device: {self.device}')

        #PUBLISHERS / SUBSCRIBERS
        self.frame_sub = self.create_subscription(Image,'anafi/frames', self.frame_callback, 1)
        self.publish_frame = self.create_publisher(FrameYolo, '/jackal_frame',1)

        self.jackal_frame = FrameYolo()


    def frame_callback(self, msg):
        
        self.jackal_frame.target = False
        try:
            # Convert ROS2 Image message to OpenCV image (NumPy array)
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        if image is None:
            self.get_logger().error(f"Failed to convert image: {e}")
            return
        
        best_bbox = None     # Variable to store the best bounding box
        
        img_height, img_width = image.shape[:2]

        results = self.model(image)

        for result in results:
            if result.boxes:
                max_confidence = -1  # Initialize max confidence to a very low value
                
                for box in result.boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Extract class and confidence
                    cls = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = self.model.names[cls]

                    # Focus only on the 'jackal' class if available
                    if label.lower() == 'jackal' and confidence >= 0.7:
                        if 20 <= x1 < img_width - 20 and 20 <= y1 < img_height - 20 and 20 <= x2 <= img_width - 20 and 20 <= y2 <= img_height - 20:
                            # Update the best bounding box if this one has higher confidence
                            if confidence > max_confidence:
                                max_confidence = confidence
                                best_bbox = (x1, y1, x2, y2)

                # If a valid bounding box was found, store it
        if best_bbox is not None:
            self.jackal_frame.x1, self.jackal_frame.y1, self.jackal_frame.x2, self.jackal_frame.y2 = best_bbox
            self.get_logger().info(f"Best bounding box: x1={best_bbox[0]}, y1={best_bbox[1]}, x2={best_bbox[2]}, y2={best_bbox[3]}")
            self.jackal_frame.target = True
            self.jackal_frame.image = msg
        
        else:
            self.jackal_frame.target = False
            self.jackal_frame.image = msg
            bbox = (-1, -1, -1, -1)
            self.jackal_frame.x1, self.jackal_frame.y1, self.jackal_frame.x2, self.jackal_frame.y2 = bbox


        self.publish_frame.publish(self.jackal_frame)
        



    #STOP FUNCTION
    def Stop(self):
        
        time.sleep(0.2)
        self.destroy_node()
        rclpy.shutdown()


#MAIN BOOL
def main():
    rclpy.init()
    af_2D_bbox = Frame()

    try:
        rclpy.spin(af_2D_bbox)  # Keeps the node spinning and listening for messages
    except Exception as e:
        af_2D_bbox.get_logger().error(f"Exception occurred: {e}")
    finally:
        af_2D_bbox.Stop()

if __name__ == "__main__":
    main()