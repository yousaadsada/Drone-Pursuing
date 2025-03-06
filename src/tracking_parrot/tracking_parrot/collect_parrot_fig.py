import rclpy
from rclpy.node import Node
import olympe
from std_msgs.msg import String, Bool
from anafi_msg.msg import Output, Position, Speed, CurrentState
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import transforms3d
from anafi_msg.msg import DroneSize, KeyPoints, ImageKeyPoints
import numpy as np
from geometry_msgs.msg import Point32
from sensor_msgs.msg import Image
import cv2, queue
from cv_bridge import CvBridge
import os
import threading
from pynput.keyboard import Listener, Key
import time
import logging
import json
import math
from olympe.messages.camera import alignment_offsets
from olympe.messages.camera import set_alignment_offsets
from geometry_msgs.msg import Vector3

logging.getLogger("olympe").setLevel(logging.WARNING)


class GetKeyPoints(Node):
    def __init__(self):
        super().__init__('get_keypoints')

        self.save_dir_image = "/home/yousa/anafi_simulation/src/tracking_parrot/tracking_parrot/parrot_fig_test"
        self.save_dir_keypoints = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'keypoints', 'point')
        os.makedirs(self.save_dir_image, exist_ok=True)

        # Set image_id based on existing files in the directory
        self.image_id = self.get_next_image_id(self.save_dir_image)

        self.running = True
        self.connected = False
        self.data_collection = False

        self.image = None
        
        self.bridge = CvBridge()
        self.frame_queue = queue.LifoQueue()
        self.processing_image_thread = threading.Thread(target=self.yuv_frame_processing)
        self.cv2_cvt_color_flag = {
            olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
        }
        
        self.start_user_input_thread()

        self.collecting_data_thread = threading.Thread(target=self.collect_show_data_callback)
        self.collecting_data_thread.daemon = True
        self.collecting_data_thread.start()

    def get_next_image_id(self, directory):
        """
        Get the next image ID by checking the files in the directory.
        """
        # List all files in the directory
        files = os.listdir(directory)
        
        # Extract numeric parts of filenames (e.g., 0.jpg -> 0)
        image_numbers = [
            int(os.path.splitext(f)[0]) for f in files if f.endswith('.jpg') and os.path.splitext(f)[0].isdigit()
        ]
        
        # Return the next available ID
        return max(image_numbers, default=-1) + 1
    


    def start_user_input_thread(self):
        input_thread = threading.Thread(target = self.handle_user_input)
        input_thread.daemon = True
        input_thread.start()

    def handle_user_input(self):
        while rclpy.ok():
            try:
                user_input = input('Press 1 for start collecting, 2 for stop collecting.')
                if user_input == '1':
                    self.data_collection = True
                elif user_input == '2':
                    self.data_collection = False
                else:
                    print("Invalid input. Please press 1 for start collecting or 2 for stop collecting.")
            except ValueError:
                print("Invalid input. Please press 1 for start collecting and 2 for stop collecting.")

    


    def collect_show_data_callback(self):
 
        while self.running:  # Keep the thread running while self.running is True
            if self.data_collection:
                if self.image is None:
                    self.get_logger().info("No image received yet.")
                else:
                    current_image = self.image.copy()  # Use a copy to avoid race conditions
                    image_save_path = os.path.join(self.save_dir_image, f'{self.image_id}.jpg')
                    cv2.imwrite(image_save_path, current_image)
                    self.image_id += 1

                    cv2.imshow("Image with keypoints cube", current_image)
                    cv2.waitKey(1)
            
            # Sleep to prevent busy-waiting
            time.sleep(0.5)
         


    def yuv_frame_cb(self, yuv_frame):
        try:
            yuv_frame.ref()
            self.frame_queue.put_nowait(yuv_frame)

        except Exception as e:
            self.get_logger().info(f"Error handling media removal: {e}")


    def yuv_frame_processing(self):
        while self.running:
            try:
                yuv_frame = self.frame_queue.get(timeout=0.1)
                
                if yuv_frame is not None:
                    x = yuv_frame.as_ndarray()
                    cv2frame = cv2.cvtColor(x, self.cv2_cvt_color_flag[yuv_frame.format()])
                    self.image = cv2frame
                    yuv_frame.unref()
    
            except queue.Empty:
                pass
            except Exception as e:
                self.get_logger().info(f"Error processing frame: {e}")


    def flush_cb(self, stream):
        if stream["vdef_format"] != olympe.VDEF_I420:
            return True
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait().unref()
        return True


    def Connect(self):
        self.get_logger().info('Connecting to Anafi drone...')
        #self.DRONE_IP = os.getenv("DRONE_IP", "10.202.0.1")
        self.DRONE_IP = os.getenv("DRONE_IP", "192.168.42.1")
        self.DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")
        self.drone = olympe.Drone(self.DRONE_IP)

        for i in range(5):
            if self.running:
                connection = self.drone.connect(retry=1)
                if connection:
                    self.connected = True
                    self.get_logger().info('Connected to Anafi drone!')

                    if self.DRONE_RTSP_PORT is not None:
                        self.drone.streaming.server_addr = f"{self.DRONE_IP}:{self.DRONE_RTSP_PORT}"
                    
                    self.drone.streaming.set_callbacks(
                        raw_cb=self.yuv_frame_cb,
                        flush_raw_cb=self.flush_cb,)
                
                    self.drone.streaming.start()
                    self.processing_image_thread.start()   
                   
                    break
                
                else:
                    self.get_logger().info(f'Trying to connect (%d)' % (i + 1))
                    time.sleep(2)

        if not self.connected:
            self.get_logger().info("Failed to connect.")


    def Stop(self):
        self.running = False
        if self.Connected:
            self.processing_image_thread.join(timeout=1.0)
            self.drone.streaming.stop()
            self.drone.disconnect()
        
        self.get_logger().info('Shutting down...\n')
        self.Connected = False
        time.sleep(0.2)
        self.destroy_node()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = GetKeyPoints()
    node.Connect()
    rclpy.spin(node)

if __name__ == '__main__':
    main()