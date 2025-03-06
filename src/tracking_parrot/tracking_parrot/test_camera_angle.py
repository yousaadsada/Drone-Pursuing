import rclpy
from rclpy.node import Node
from anafi_msg.msg import CurrentState, Output, Position
import numpy as np
import casadi as ca
import os
import threading
import cvxpy as cp
from std_msgs.msg import Bool
import ecos
import rclpy
import olympe
from rclpy.node import Node
from std_msgs.msg import String, Bool
import pysphinx
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from olympe.messages.ardrone3.PilotingState import SpeedChanged
from pynput.keyboard import Listener, Key
from geometry_msgs.msg import Vector3
import time
import os
from anafi_msg.msg import Output, Position, Speed, CurrentState
import math
import numpy as np
import logging
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import transforms3d
import csv
from olympe.messages.gimbal import set_target, set_max_speed
from olympe.messages.gimbal import attitude

logging.getLogger("olympe").setLevel(logging.WARNING)

class Camera_Control(Node):
    def __init__(self):
        super().__init__('camera_test')
        self.running = True
        self.connected = False
        self.publish_camera_angle = False

        self.DRONE_IP = os.getenv("DRONE_IP", "192.168.42.1")
        self.drone = olympe.Drone(self.DRONE_IP)
        self.target_frame = 'anafi'

        self.x_manual = 0
        self.y_manual = 0
        self.z_manual = 0
        self.yaw_manual = 0
        

        self.camera_pitch = -90.0
        self.camera_roll = 0.0
    


        self.Connect()
        self.angles_pub = self.create_publisher(Vector3, '/gimbal/angles', 1)


        self.set_max_speed(180.0)

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        
        self.handle_input_thread = threading.Thread(target=self.handle_input_thread_callback)
        self.handle_input_thread.daemon = True
        self.handle_input_thread.start()

        self.publish_camera_angle_thread = threading.Thread(target=self.publish_camera_angle_thread_callback)
        self.publish_camera_angle_thread.daemon = True
        self.publish_camera_angle_thread.start()

        self.camera_angle_thread = threading.Thread(target=self.camera_angle_thread_callback)
        self.camera_angle_thread.daemon = True
        self.camera_angle_thread.start()

        self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
        self.publish_pcmd_thread.daemon = True
        self.publish_pcmd_thread.start()




    def Connect(self):
        self.get_logger().info('Connecting to Anafi drone...')

        self.DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
        self.DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")
        self.drone = olympe.Drone(self.DRONE_IP)

        for i in range(5):
            if self.running:
                connected= self.drone.connect(retry=1)
                if connected:
                    self.connected = True
                    self.get_logger().info('Conected to Anafi drone!')
                    break
                else:
                    self.get_logger().info(f'Trying to connect ({i+1})')
                    time.sleep(2)

            else:
                self.get_logger().info("Failed to connect.")


    



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




    def handle_input_thread_callback(self):
        while rclpy.ok():
            try:
                user_input = input('Enter [pitch roll]: ')
                data = [float(value) for value in user_input.split()]
                if len(data) == 2:
                    self.publish_reference(*data)
                    self.publish_camera_angle = True
    
                else:
                    print("Invalid input. Please enter 2 values.")
            except ValueError:
                print("Invalid input. Please enter numeric values.")
    
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
                                      yaw= 0.0,  # Float
                                      pitch_frame_of_reference="absolute",
                                      pitch=self.camera_pitch,  # Float
                                      roll_frame_of_reference="none",
                                      roll=self.camera_roll  # Float
                                      ))


            else:
                pass


    def camera_angle_thread_callback(self):

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
   
   
    def publish_pcmd_thread_callback(self):

        while self.running:
        
            self.drone(PCMD(1,
                            -self.y_manual,
                            self.x_manual,
                            -self.yaw_manual,
                            self.z_manual,
                            timestampAndSeqNum=0,))
            time.sleep(0.04)

    




def run_control_loop(node):
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in control loop: {e}")
    finally:
        print("Disconnected from Anafi drone.")




def main(args=None):
    rclpy.init(args=args)
    node = Camera_Control()
    run_control_loop(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()