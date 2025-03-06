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
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from olympe.messages.gimbal import set_target, set_max_speed
import tkinter as tk
from tkinter import simpledialog

logging.getLogger("olympe").setLevel(logging.WARNING)


class GetKeyPoints(Node):
    def __init__(self):
        super().__init__('get_keypoints')

        self.running = True
        self.connected = False
                    
        self.bridge = CvBridge()
        self.frame_queue = queue.LifoQueue()
        self.processing_image_thread = threading.Thread(target=self.yuv_frame_processing)
        self.cv2_cvt_color_flag = {
            olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
        }
            

        self.Connect()

        if self.connected:

            self.save_dir_image = "/home/yousa/anafi_simulation/src/drop_ball/jackal_fig/yolo_2dbbox_test"
            os.makedirs(self.save_dir_image, exist_ok=True)


            self.camera_pitch = -20.0

            self.set_max_speed(180.0)

            # Set image_id based on existing files in the directory
            self.image_id = self.get_next_image_id(self.save_dir_image)

            self.data_collection = False
            self.publish_camera_angle = True

            self.x_manual = 0
            self.y_manual = 0
            self.z_manual = 0
            self.yaw_manual = 0

            self.image = None

            self.start_user_input_thread()

            self.start_tkinter_thread()

            self.collecting_data_thread = threading.Thread(target=self.collect_show_data_callback)
            self.collecting_data_thread.daemon = True
            self.collecting_data_thread.start()

            self.publish_camera_angle_thread = threading.Thread(target=self.publish_camera_angle_thread_callback)
            self.publish_camera_angle_thread.daemon = True
            self.publish_camera_angle_thread.start()

            self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
            self.publish_pcmd_thread.daemon = True
            self.publish_pcmd_thread.start()
    

            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()

    
    def on_press(self, key):
        if hasattr(key, 'char') and (key.char == 'w' or key.char == 's' or key.char == 'a' or key.char == 'd' or key.char == 'c' or key.char == 'x' or key.char == 'f' or key.char == 'r'): 
            self.mpc_or_manual = 'manual'

        if key == Key.left:
            #self.get_logger().info("Landing command detected (left key press).")
            time.sleep(0.1)
            self.mpc_or_manual = 'manual'
            try:
                self.drone(Landing())
            except Exception as e:
                self.get_logger().info("Failed to Land.")
            time.sleep(0.5)

        elif key == Key.right:
            #self.get_logger().info("Takeoff command detected (right key press).")
            time.sleep(0.1)
            self.mpc_or_manual = 'manual'
            try:
                self.drone(TakeOff())
            except Exception as e:
                self.get_logger().info("Failed to Take Off.")
            time.sleep(0.5)

        elif hasattr(key, 'char') and key.char:
            if key.char == 'w':
                self.x_manual = 20
            elif key.char == 's':
                self.x_manual = -20
            elif key.char == 'a':
                self.y_manual = 20
            elif key.char == 'd':
                self.y_manual = -20
            elif key.char == 'r':
                self.z_manual = 10
            elif key.char == 'f':
                self.z_manual = -10            
            elif key.char == 'c':
                self.yaw_manual = 100
            elif key.char == 'x':
                self.yaw_manual = -100

    def on_release(self, key):
        if hasattr(key, 'char') and key.char in ['w', 's']:
            self.x_manual = 0
        if hasattr(key, 'char') and key.char in ['a', 'd']:
            self.y_manual = 0
        if hasattr(key, 'char') and key.char in ['r', 'f']:
            self.z_manual = 0
        if hasattr(key, 'char') and key.char in ['x', 'c']:
            self.yaw_manual = 0

    def start_tkinter_thread(self):
        # Start Tkinter in a separate thread
        input_thread = threading.Thread(target=self.create_input_window)
        input_thread.daemon = True
        input_thread.start()

    def create_input_window(self):
    # Create a Tkinter window for float input
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        while self.running:
            try:
                # Open a dialog box to input yaw angle
                input_value = simpledialog.askfloat(
                    "Camera Yaw Angle Input",
                    "Enter camera yaw angle (float):",
                )
                if input_value is not None:
                    self.camera_yaw = input_value
                    self.publish_camera_angle = True
            except Exception as e:
                self.get_logger().info(f"Error in input window: {e}")
                


    

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




    def publish_pcmd_thread_callback(self):

        while self.running:
           
            self.drone(PCMD(1,
                            -self.y_manual,
                            self.x_manual,
                            -self.yaw_manual,
                            self.z_manual,
                            timestampAndSeqNum=0,))
            

    
    def publish_reference(self, pitch, roll):
        self.camera_pitch = pitch
        self.camera_roll = roll

    def set_max_speed(self, speed_pitch):
        response = self.drone(set_max_speed(
            gimbal_id=0,
            yaw=0.0,  # Ignored
            pitch=speed_pitch,  # Maximum pitch speed in degrees/sec
            roll=0.0  # Ignored
        )).wait()
        if response.success():
            print(f"Gimbal max speed set to {speed_pitch}°/s")
            self.max_speed_set = True
        else:
            print("Failed to set max speed.")


    def publish_camera_angle_thread_callback(self):

        while self.running:
           
            if self.publish_camera_angle is True:
                self.drone(set_target(gimbal_id=0,
                                      control_mode="position",
                                      yaw_frame_of_reference="none",
                                      yaw=0.0,  # Float
                                      pitch_frame_of_reference="absolute",
                                      pitch=self.camera_pitch,  # Float
                                      roll_frame_of_reference="none",
                                      roll=0.0  # Float
                                      ))


            else:
                pass




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
    rclpy.spin(node)

if __name__ == '__main__':
    main()