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
from my_custom_msgs.msg import Output, Position, Speed, CurrentState, Matrices
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
        self.is_control_on = False
        self.is_manual_on = True
        self.out_bound = False
        
        self.target_frame = 'anafi'
        self.drone_state = CurrentState()

        self.freq_publish_pcmd = 25
        self.freq_get_current_state = 40
        self.freq = 25

        self.bound_x_max = 100.0
        self.bound_x_min = -100.0
        self.bound_y_max = 100.0
        self.bound_y_min = -100.0
        self.bound_z_max = 100.0
        self.bound_z_min = 0.001

        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.gaz = 0
        self.takeoff = False
        self.land = False

        # MANUAL MODE
        self.x_manual = 0
        self.y_manual = 0
        self.z_manual = 0
        self.yaw_manual = 0

        self.x_control = 0
        self.y_control = 0
        self.z_control = 0
        self.yaw_control = 0

        self.x_error = 0
        self.y_error = 0
        self.z_error = 0
        self.yaw_error = 0

        self.previous_time = None

        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None

        self.pcmd = Output()

        # data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone')
        # self.A = np.loadtxt(os.path.join(data_dir, 'A_matrix.csv'), delimiter=',')
        # self.B = np.loadtxt(os.path.join(data_dir, 'B_matrix.csv'), delimiter=',')

        self.timer_pubPCMD = self.create_timer(callback=self.publish_pcmd, timer_period_sec=1/self.freq_publish_pcmd)
        self.calculate_speed = self.create_timer(callback=self.calculate_speed_callback, timer_period_sec=1/self.freq)

        self.subscribe_onAndoff = self.create_subscription(Bool, 'control_on_off', self.control_on_off_callback, 10)
        self.subscribe_drone_tf = self.create_subscription(TFMessage, 'tf',self.drone_tf_callback, 10)
        self.get_control_command = self.create_subscription(Output, '/control_command', self.receive_control_command, 10)
        self.shutdown_control_command = self.create_publisher(Bool, '/shutdown_control_command', 10)
        self.publish_drone_state = self.create_publisher(CurrentState, '/drone_current_state',10)
        self.publish_control_value = self.create_publisher(Output, 'control_command', 10)

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if hasattr(key, 'char') and (key.char == 'w' or key.char == 's' or key.char == 'a' or key.char == 'd' or key.char == 'c' or key.char == 'x' or key.char == 'f' or key.char == 'r'): 
           self.is_control_on = False
           self.is_manual_on = True
           self.shutdown_control_command.publish(Bool(data=self.is_control_on))

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

            #self.get_logger().info(f"Manual control: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")

    def on_release(self, key):
        if hasattr(key, 'char') and key.char in ['w', 's']:
            self.x_manual = 0
        if hasattr(key, 'char') and key.char in ['a', 'd']:
            self.y_manual = 0
        if hasattr(key, 'char') and key.char in ['r', 'f']:
            self.z_manual = 0
        if hasattr(key, 'char') and key.char in ['x', 'c']:
            self.yaw_manual = 0

        #self.get_logger().info(f"Manual control released: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")

    def control_on_off_callback(self, msg):
        self.is_control_on =  msg.data
        if self.is_control_on == True:
            self.is_manual_on = False
        else:
            self.is_manual_on = True

    def receive_control_command(self, msg):

        self.x_control = msg.control_x
        self.y_control = msg.control_y
        self.z_control = msg.control_z
        self.yaw_control = msg.control_yaw
    
    def publish_pcmd(self):
        # print(self.is_manual_on, self.is_control_on, self.out_bound)
        # if self.out_bound is True:
        #     self.drone(PCMD(1,
        #                     -self.y_error,
        #                     self.x_error,
        #                     0,
        #                     self.z_error,
        #                     timestampAndSeqNum=0,))

        if self.out_bound is False and self.is_control_on is False and self.is_manual_on is True:
            self.get_logger().info(f"Publishing manual PCMD: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")
            self.drone(PCMD(1,
                            -self.y_manual,
                            self.x_manual,
                            -self.yaw_manual,
                            self.z_manual,
                            timestampAndSeqNum=0,))
            
        elif self.out_bound is False and self.is_control_on is True and self.is_manual_on is False:
            self.get_logger().info(f"Publishing MPC PCMD: x={self.x_control}, y={self.y_control}, z={self.z_control}, yaw={self.yaw_control}")
            self.drone(PCMD(1,
                            -self.y_control,
                            self.x_control,
                            -self.yaw_control,
                            self.z_control,
                            timestampAndSeqNum=0,))
        
        else:
            self.get_logger().warn("Failed to publish PCMD from Sphinx.")



    def drone_tf_callback(self, msg):    
        for transform in msg.transforms:
            if transform.child_frame_id == self.target_frame:
                self.process_transform(transform)
    
    def process_transform(self, transform: TransformStamped):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        
        x = translation.x
        y = translation.y
        z = translation.z

        quaternion = (
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w
        )

        euler = transforms3d.euler.quat2euler(quaternion)
        yaw, pitch, roll = euler[0], euler[1], euler[2]
        
        self.drone_state.position.x = x
        self.drone_state.position.y = y
        self.drone_state.position.z = z 
        self.drone_state.position.yaw = yaw
        self.drone_state.position.pitch = pitch
        self.drone_state.position.roll = roll 



        # if self.drone_state.position.x > self.bound_x_max or \
        # self.drone_state.position.x < -self.bound_x_min or \
        # self.drone_state.position.y > self.bound_y_max or \
        # self.drone_state.position.y < -self.bound_y_min or \
        # self.drone_state.position.z > self.bound_z_max or \
        # self.drone_state.position.z < self.bound_z_min:
            
        #     self.out_bound = True
        
        #     e_u = np.zeros(3)
        #     if self.drone_state.position.x > self.bound_x_max:
        #         e_u[0] = -100
        #     if self.drone_state.position.x < self.bound_x_min:
        #         e_u[0] = 100
        #     if self.drone_state.position.y > self.bound_y_max:
        #         e_u[1] = -100
        #     if self.drone_state.position.y < self.bound_y_min:
        #         e_u[1] = 100
        #     if self.drone_state.position.z > self.bound_z_max:
        #         e_u[2] = -100
        #     if self.drone_state.position.z < self.bound_z_min:
        #         e_u[2] = 100
            
        #     e_R = self.get_R(self.drone_state.position.yaw)
        #     e_R_inv = np.linalg.inv(e_R)
        #     e_u_world_xy = np.array([e_u[0], e_u[1]])
        #     e_u_body_xy = e_R_inv @ e_u_world_xy
        #     self.x_error = int(e_u_body_xy[0])
        #     self.y_error = int(e_u_body_xy[1])
        #     self.z_error = int(e_u[2])

        # else:
        #     self.out_bound = False
    
    # def get_R(self, yaw):
    #     R = np.array([
    #         [np.cos(yaw), -np.sin(yaw)],
    #         [np.sin(yaw), np.cos(yaw)]
    #     ])
    #     return R

    def calculate_speed_callback(self):
        current_time = self.get_clock().now().nanoseconds / 1e9

        speeds = {"x":0.0, "y":0.0, "z":0.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0}
        positions = {"x":self.drone_state.position.x, "y":self.drone_state.position.y, "z":self.drone_state.position.z,
                     "yaw": self.drone_state.position.yaw, "roll":self.drone_state.position.roll, "pitch":self.drone_state.position.pitch}
        previous_positions = {"x": self.previous_x, "y": self.previous_y, "z": self.previous_z,
                              "yaw": self.previous_yaw, "roll":self.previous_roll, "pitch":self.previous_pitch}
        
        if self.previous_time is not None:
            for index, axis in enumerate(speeds.keys()):
                current_position = positions[axis]
                previous_position = previous_positions[axis]
                
                if previous_position is not None:
                    if index >= 3:
                        if current_position < 0 and previous_position > 0 and current_position < -0.9 * math.pi and previous_position > 0.9 * math.pi:
                            delta_position = 2 * math.pi + current_position - previous_position
                        elif current_position > 0 and previous_position < 0 and current_position > 0.9 * math.pi and previous_position < -0.9 * math.pi:
                            delta_position = -2 * math.pi + current_position - previous_position
                        else:
                            delta_position = current_position - previous_position
                    else:
                        delta_position = current_position - previous_position
 
                    delta_time = current_time - self.previous_time
                    speeds[axis] = delta_position / delta_time
        
        self.drone_state.speed.x_speed = speeds["x"]
        self.drone_state.speed.y_speed = speeds["y"]
        self.drone_state.speed.z_speed = speeds["z"]
        self.drone_state.speed.yaw_speed = speeds["yaw"]
        self.drone_state.speed.roll_speed = speeds["roll"]
        self.drone_state.speed.pitch_speed = speeds["pitch"]

        self.previous_x = self.drone_state.position.x
        self.previous_y = self.drone_state.position.y
        self.previous_z = self.drone_state.position.z
        self.previous_yaw = self.drone_state.position.yaw
        self.previous_roll = self.drone_state.position.roll
        self.previous_pitch = self.drone_state.position.pitch

        self.previous_time = current_time

        self.publish_drone_state.publish(self.drone_state)



    def Connect(self):
        self.get_logger().info('Connecting to Anafi drone...')
        self.DRONE_IP = os.getenv("DRONE_IP", "192.168.42.1")
        #self.DRONE_IP = os.getenv("DRONE_IP", "10.202.0.1")
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