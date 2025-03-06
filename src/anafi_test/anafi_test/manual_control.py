import rclpy
import olympe
from rclpy.node import Node
from std_msgs.msg import String, Bool
import pysphinx
import logging
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from pynput.keyboard import Listener, Key
from geometry_msgs.msg import Vector3
import time
import os
from anafi_msg.msg import Output, Position, Speed, CurrentState, Matrices
import math
import numpy as np
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import transforms3d

logging.getLogger("olympe").setLevel(logging.WARNING)

class ManualControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')

        self.running = True
        self.connected = False
        self.is_manual_on = True

        self.freq_publish_pcmd = 40

        self.takeoff = False
        self.land = False

        self.x_manual = 0
        self.y_manual = 0
        self.z_manual = 0
        self.yaw_manual = 0
        
        self.pcmd_pub = self.create_timer(timer_period_sec= 1/self.freq_publish_pcmd, callback=self.publish_pcmd)

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if hasattr(key, 'char') and (key.char == 'w' or key.char == 's' or key.char == 'a' or key.char == 'd' or key.char == 'c' or key.char == 'x' or key.char == 'f' or key.char == 'r'): 
           self.is_manual_on = True

        if key == Key.left:
            self.get_logger().info("Landing command detected (left key press).")
            time.sleep(0.1)
            self.drone(Landing())
            time.sleep(0.5)

        elif key == Key.right:
            self.get_logger().info("Takeoff command detected (right key press).")
            time.sleep(0.1)
            self.drone(TakeOff())
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
                self.yaw_manual = 50
            elif key.char == 'x':
                self.yaw_manual = -50

            self.get_logger().info(f"Manual control: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")

    def on_release(self, key):
        if hasattr(key, 'char') and key.char in ['w', 's']:
            self.x_manual = 0
        if hasattr(key, 'char') and key.char in ['a', 'd']:
            self.y_manual = 0
        if hasattr(key, 'char') and key.char in ['r', 'f']:
            self.z_manual = 0
        if hasattr(key, 'char') and key.char in ['x', 'c']:
            self.yaw_manual = 0

        self.get_logger().info(f"Manual control released: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")

   
    def publish_pcmd(self):

        if self.is_manual_on is True:
            self.drone(PCMD(1,
                            -self.y_manual,
                            self.x_manual,
                            -self.yaw_manual,
                            -self.z_manual,
                            timestampAndSeqNum=0,))
            


    def Connect(self):
        self.get_logger().info('Connecting to Anafi drone...')
        self.DRONE_IP = os.getenv("DRONE_IP", "192.168.42.1")
        self.drone = olympe.Drone(self.DRONE_IP)

        for i in range(5):
            if self.running:
                connection = self.drone.connect(retry=1)
                if connection:
                    self.connected = True
                    self.get_logger().info('Connected to Anafi drone!')
                    break
                else:
                    self.get_logger().info(f'Trying to connect (%d)' % (i + 1))
                    time.sleep(2)

        if not self.connected:
            self.get_logger().info("Failed to connect.")

    def Stop(self):
        self.running = False

def main(args=None):
    rclpy.init(args=args)
    node = ManualControlNode()
    node.Connect()
    rclpy.spin(node)
    
if __name__ == '__main__':
    main()