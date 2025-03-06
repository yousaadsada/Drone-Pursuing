import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from anafi_msg.msg import Output, Position, Speed, CurrentState
from geometry_msgs.msg import TransformStamped
import threading
import numpy as np
from geometry_msgs.msg import Point32
from anafi_msg.msg import DroneSize, KeyPoints, ImageKeyPoints
from rclpy.executors import MultiThreadedExecutor
import time
import os
import cv2, queue
import olympe
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import logging
from olympe.messages.gimbal import attitude
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from pynput.keyboard import Listener, Key
from geometry_msgs.msg import Vector3
from olympe.messages.gimbal import set_target, set_max_speed
from tkinter import Tk, Label, Entry, Button

logging.getLogger("olympe").setLevel(logging.WARNING)

class GetVedioData(Node):
    def __init__(self):
        super().__init__('get_vedio_data')
        self.running = True
        self.publish_camera_angle = True
        self.frameid = 0
        self.bridge = CvBridge()
        self.frame_queue = queue.LifoQueue()
        self.processing_image_thread = threading.Thread(target = self.yuv_frame_processing)
        self.cv2_cvt_color_flag = {
                    olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
                    olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
                }
  
        self.Connect()

        self.x_manual = 0
        self.y_manual = 0
        self.z_manual = 0
        self.yaw_manual = 0

        self.camera_pitch = 0.0

        self.image_pub = self.create_publisher(Image, 'anafi/vedio', 1)   
        self.angles_pub = self.create_publisher(Vector3, '/gimbal/angles', 1)
        
        self.set_max_speed(180.0)
        # self.create_gui()

        # self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
        # self.publish_pcmd_thread.daemon = True
        # self.publish_pcmd_thread.start()

        self.sub_camera_angle_thread = threading.Thread(target=self.sub_camera_angle_thread_callback)
        self.sub_camera_angle_thread.daemon = True
        self.sub_camera_angle_thread.start()

        self.pub_camera_angle_thread = threading.Thread(target=self.publish_camera_angle_thread_callback)
        self.pub_camera_angle_thread.daemon = True
        self.pub_camera_angle_thread.start()

        # self.set_camera_angle_thread = threading.Thread(target=self.set_camera_angle_thread_callback)
        # self.set_camera_angle_thread.daemon = True
        # self.set_camera_angle_thread.start()

        # self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        # self.listener.start()


    def on_press(self, key):

        if key == Key.left:
            #self.get_logger().info("Landing command detected (left key press).")
            time.sleep(0.1)
      
            try:
                self.drone(Landing())
            except Exception as e:
                self.get_logger().info("Failed to Land.")
            time.sleep(0.5)

        elif key == Key.right:
            #self.get_logger().info("Takeoff command detected (right key press).")
            time.sleep(0.1)

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


    def publish_pcmd_thread_callback(self):

        while self.running:
        
            self.drone(PCMD(1,
                            -self.y_manual,
                            self.x_manual,
                            -self.yaw_manual,
                            self.z_manual,
                            timestampAndSeqNum=0,))
            time.sleep(0.04)


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

    def create_gui(self):
        # Create a Tkinter window
        self.window = Tk()
        self.window.title("Set Camera Pitch")

        # Label for instructions
        Label(self.window, text="Enter Camera Pitch (degrees):").grid(row=0, column=0)

        # Entry box for pitch input
        self.pitch_input = Entry(self.window)
        self.pitch_input.grid(row=0, column=1)

        # Button to set camera pitch
        Button(self.window, text="Set Pitch", command=self.set_camera_pitch).grid(row=1, column=0, columnspan=2)

        # Start the Tkinter main loop in a separate thread
        self.gui_thread = threading.Thread(target=self.window.mainloop)
        self.gui_thread.daemon = True
        self.gui_thread.start()

    def set_camera_angle_thread_callback(self):
        while self.running:
            try:
                # Get the float value from the input box
                self.publish_camera_angle = True

                new_pitch = float(self.pitch_input.get())
                self.camera_pitch = new_pitch
                self.get_logger().info(f"Camera pitch set to: {self.camera_pitch} degrees")
            except ValueError:
                self.get_logger().info("Invalid input. Please enter a valid float value.")
            time.sleep(0.01)


    def publish_camera_angle_thread_callback(self):

        while self.running:
           
            if self.publish_camera_angle is True:
                self.drone(set_target(gimbal_id=0,
                                      control_mode="position",
                                      yaw_frame_of_reference="none",
                                      yaw= 0.0,  # Float
                                      pitch_frame_of_reference="absolute",
                                      pitch=self.camera_pitch,  # Float
                                      roll_frame_of_reference="none",
                                      roll=0.0  # Float
                                      ))

            else:
                pass

            time.sleep(0.1)

                

    def sub_camera_angle_thread_callback(self):

        while self.running:
            # Fetch gimbal attitude state
            gimbal_state = self.drone.get_state(attitude)

            yaw = float()
            pitch = float()
            roll = float()

            gimbal_data = gimbal_state[0]
            # Extract yaw, pitch, and roll angles
            yaw = gimbal_data['yaw_absolute']    # Yaw angle in degrees
            pitch = gimbal_data['pitch_absolute']  # Pitch angle in degrees
            roll = gimbal_data['roll_absolute']   # Roll angle in degrees

            angles_msg = Vector3()
            angles_msg.x = yaw    # Yaw corresponds to x
            angles_msg.y = pitch  # Pitch corresponds to y
            angles_msg.z = roll   # Roll corresponds to z

            self.angles_pub.publish(angles_msg)

            time.sleep(0.2)


    
    


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

                    msg = self.bridge.cv2_to_imgmsg(cv2frame, "bgr8")
                    msg.header.frame_id = str(self.frameid)
                    self.image_pub.publish(msg)
                    self.frameid += 1
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
                    time.sleep(2.0)

        if not self.connected:
            self.get_logger().info("Failed to connect.")

    def Stop(self):
        self.running = False
        if self.connected:
            self.processing_image_thread.join(timeout=1.0)
            self.drone.streaming.stop()
            self.drone.disconnect()
        
        self.get_logger().info('Shutting down...\n')
        self.Connected = False
        time.sleep(0.2)
        self.destroy_node()
        rclpy.shutdown()




def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)

    # Create the GetVedioData node
    node = GetVedioData()

    # Create a MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)

    # Create Tkinter GUI in the main thread
    root = Tk()
    root.title("Set Camera Pitch")

    # Add input elements to the Tkinter GUI
    Label(root, text="Enter Camera Pitch (degrees):").grid(row=0, column=0)
    pitch_input = Entry(root)
    pitch_input.grid(row=0, column=1)

    # Function to set camera pitch
    def set_camera_pitch():
        try:
            # Get the float value from the input box
            new_pitch = float(pitch_input.get())
            node.camera_pitch = new_pitch
            node.get_logger().info(f"Camera pitch set to: {node.camera_pitch} degrees")
        except ValueError:
            node.get_logger().info("Invalid input. Please enter a valid float value.")

    # Add a button to update pitch
    Button(root, text="Set Pitch", command=set_camera_pitch).grid(row=1, column=0, columnspan=2)

    # Start the ROS executor in a background thread
    def ros_spin():
        try:
            node.get_logger().info("GetVedioData node is running. Press Ctrl+C to exit.")
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down gracefully...")
        finally:
            executor.shutdown()
            node.Stop()

    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()

    # Run the Tkinter main loop in the main thread
    try:
        root.mainloop()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down gracefully...")
    finally:
        # Shutdown ROS2 and clean up
        rclpy.shutdown()
        node.get_logger().info("Node and ROS2 have been shut down.")

if __name__ == '__main__':
    main()