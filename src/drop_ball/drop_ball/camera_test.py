




import rclpy
from rclpy.node import Node
import warnings
from sensor_msgs.msg import Image
import cv2, os, olympe, psutil, time, threading, queue
from cv_bridge import CvBridge
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged, SpeedChanged, AttitudeChanged
from anafi_msg.msg import PnPData, PnPDataYolo
from pynput.keyboard import Listener, Key
import csv
import os
import casadi as ca
import numpy as np
from anafi_msg.msg import CurrentState, Output
import math
import tkinter as tk
from tkinter import messagebox
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import transforms3d
from rclpy.executors import MultiThreadedExecutor
from olympe.messages.gimbal import set_target, set_max_speed
from geometry_msgs.msg import Vector3
from olympe.messages.gimbal import attitude
from olympe.messages.common import Common

class CameraTest(Node):
    def __init__(self):
        super().__init__('af_pursuer')

        self.running = True
        self.Connect()

        self.flag = 0

        self.camera_pitch = 0.0
        self.set_max_speed(180.0)
        self.prev_pitch = None

        self.angles_pub_1 = self.create_publisher(Vector3, '/gimbal/angles_1', 1)
        self.angles_pub_2 = self.create_publisher(Vector3, '/gimbal/angles_2', 1)

        self.pub_camera_angle_thread = threading.Thread(target=self.publish_camera_angle_thread_callback)
        self.pub_camera_angle_thread.daemon = True
        self.pub_camera_angle_thread.start()

        self.sub_camera_angle_thread = threading.Thread(target=self.sub_camera_angle_thread_callback)
        self.sub_camera_angle_thread.daemon = True
        self.sub_camera_angle_thread.start()




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

            if self.flag == 0:
           
                if self.camera_pitch < 90:
                    self.camera_pitch = self.camera_pitch + 1.0
            
                else:
                    self.flag = 1
            
            elif self.flag == 1:
                if self.camera_pitch > 0:
                    self.camera_pitch = self.camera_pitch - 1.0
            
                else:
                    self.flag = 0


            self.drone(set_target(gimbal_id=0,
                                control_mode="position",
                                yaw_frame_of_reference="none",
                                yaw= 0.0,  # Float
                                pitch_frame_of_reference="absolute",
                                pitch=-self.camera_pitch,  # Float
                                roll_frame_of_reference="none",
                                roll=0.0  # Float
                                ))
            
            angles_msg = Vector3()
            angles_msg.x = 0.0    # Yaw corresponds to x
            angles_msg.y = -self.camera_pitch # Pitch corresponds to y
            angles_msg.z = 0.0   # Roll corresponds to z
            
            self.angles_pub_2.publish(angles_msg)
     
            time.sleep(0.04)

    
    def sub_camera_angle_thread_callback(self):

        while self.running:
            # Fetch gimbal attitude state

            def interpolate_gimbal_angles(pitch, yaw, roll):
                i = 1
                angles_msg = Vector3()
                angles_msg.x = yaw    # Yaw corresponds to x
                angles_msg.y = pitch  # Pitch corresponds to y
                angles_msg.z = roll   # Roll corresponds to z
                self.angles_pub_1.publish(angles_msg)

                while i < 5:
                    time.sleep(0.04)
                    interpolate_pitch = pitch + (pitch - self.prev_pitch) * i / 5
                    i = i + 1

                    angles_msg = Vector3()
                    angles_msg.x = yaw    # Yaw corresponds to x
                    angles_msg.y = interpolate_pitch  # Pitch corresponds to y
                    angles_msg.z = roll   # Roll corresponds to z
                    self.angles_pub_1.publish(angles_msg)
 


            gimbal_state = self.drone.get_state(attitude)

            yaw = float()
            pitch = float()
            roll = float()

            gimbal_data = gimbal_state[0]
            # Extract yaw, pitch, and roll angles
            yaw = gimbal_data['yaw_absolute']    # Yaw angle in degrees
            pitch = gimbal_data['pitch_absolute']  # Pitch angle in degrees
            roll = gimbal_data['roll_absolute']   # Roll angle in degrees

            if self.prev_pitch != None:
                if pitch != self.prev_pitch:
                    interpolate_gimbal_angles(pitch, yaw, roll)
                    self.prev_pitch = pitch
                else:
                    pass
            
            else:
                self.prev_pitch = pitch

            # angles_msg = Vector3()
            # angles_msg.x = yaw    # Yaw corresponds to x
            # angles_msg.y = pitch  # Pitch corresponds to y
            # angles_msg.z = roll   # Roll corresponds to z

            # self.angles_pub_1.publish(angles_msg)

            time.sleep(0.01)


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

                    self.drone(Common.AllStates()).wait()

                    self.get_logger().info('Connected to Anafi drone!')
                   
                    break
                
                else:
                    self.get_logger().info(f'Trying to connect (%d)' % (i + 1))
                    time.sleep(2)

        if not self.connected:
            self.get_logger().info("Failed to connect.")

    def Stop(self):
        self.running = False

        self.destroy_node()
        rclpy.shutdown()


def main():
    rclpy.init()
    af_pursuer = CameraTest()

    # Use a MultiThreadedExecutor for parallel callback handling
    executor = MultiThreadedExecutor()
    executor.add_node(af_pursuer)

    try:
        executor.spin()  # Spin with multiple threads
    except KeyboardInterrupt:
        af_pursuer.Stop()
    finally:
        executor.shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()