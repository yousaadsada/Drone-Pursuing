import rclpy
import olympe
from rclpy.node import Node
from std_msgs.msg import String, Bool
import pysphinx
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from pynput.keyboard import Listener, Key
from geometry_msgs.msg import Vector3
import time
import os
from anafi_msg.msg import Output, Position, Speed, CurrentState, Matrices
import math
import numpy as np

class ManualControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')

        self.running = True
        self.connected = False
        self.is_mpc_on = False
        self.is_manual_on = True
        self.out_bound = False

        self.freq_publish_pcmd = 40
        self.freq_get_current_state = 100
        self.freq_publish_ab = 100

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

        self.x_mpc = 0
        self.y_mpc = 0
        self.z_mpc = 0
        self.yaw_mpc = 0

        self.x_error = 0
        self.y_error = 0
        self.z_error = 0
        self.yaw_error = 0

        self.previous_time = None

        self.position = Position()
        self.speed = Speed()
        self.current_state = CurrentState()

        self.previous_x = None
        self.previous_y = None
        self.previous_z = None

        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None

        self.sphinx = pysphinx.Sphinx(ip="127.0.0.1", port=8383)
        self.get_logger().info("Drone position publisher node has been started.")
        self.name = self.sphinx.get_default_machine_name()
        
        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data')
        self.A = np.loadtxt(os.path.join(data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(data_dir, 'B_matrix.csv'), delimiter=',')

        self.timer_pubPCMD = self.create_timer(callback=self.publish_pcmd, timer_period_sec=1/self.freq_publish_pcmd)
        self.timer_getPosition = self.create_timer(callback=self.get_current_state, timer_period_sec=1/self.freq_get_current_state)


        self.get_mpc_command = self.create_subscription(Output, '/mpc_control', self.receive_mpc_command, 10)
        self.shutdown_mpc = self.create_publisher(Bool, '/shutdown_mpc', 10)
        self.publisher_current_state = self.create_publisher(CurrentState, '/simulation_current_state',10)
        
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if hasattr(key, 'char') and (key.char == 'w' or key.char == 's' or key.char == 'a' or key.char == 'd' or key.char == 'c' or key.char == 'x' or key.char == 'f' or key.char == 'r'): 
            self.is_mpc_on = False
            self.is_manual_on = True
            self.shutdown_mpc.publish(Bool(data=self.is_mpc_on))

        if key == Key.left:
            self.get_logger().info("Landing command detected (left key press).")
            time.sleep(0.1)
            try:
                self.drone(Landing())
            except Exception as e:
                self.get_logger().info("Failed to Land.")
            time.sleep(0.5)

        elif key == Key.right:
            self.get_logger().info("Takeoff command detected (right key press).")
            time.sleep(0.1)
            try:
                self.drone(TakeOff())
            except Exception as e:
                self.get_logger().info("Failed to Take Off.")
            time.sleep(0.5)

        elif hasattr(key, 'char') and key.char:
            if key.char == 'w':
                self.x_manual = 50
            elif key.char == 's':
                self.x_manual = -50
            elif key.char == 'a':
                self.y_manual = 50
            elif key.char == 'd':
                self.y_manual = -50
            elif key.char == 'r':
                self.z_manual = 25
            elif key.char == 'f':
                self.z_manual = -25            
            elif key.char == 'c':
                self.yaw_manual = 100
            elif key.char == 'x':
                self.yaw_manual = -100

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

    def receive_mpc_command(self, msg):
        self.is_mpc_on = True
        self.is_manual_on = False
        self.x_mpc = msg.control_x
        self.y_mpc = msg.control_y
        self.z_mpc = msg.control_z
        self.yaw_mpc = msg.control_yaw

        self.get_logger().info(f"MPC command received: x={self.x_mpc}, y={self.y_mpc}, z={self.z_mpc}, yaw={self.yaw_mpc}")
   
    def publish_pcmd(self):
        if self.out_bound is True:
            self.drone(PCMD(1,
                            -self.y_error,
                            self.x_error,
                            0,
                            self.z_error,
                            timestampAndSeqNum=0,))

        elif self.out_bound is False and self.is_mpc_on is False and self.is_manual_on is True:
            self.get_logger().info(f"Publishing manual PCMD: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")
            self.drone(PCMD(1,
                            -self.y_manual,
                            self.x_manual,
                            -self.yaw_manual,
                            self.z_manual,
                            timestampAndSeqNum=0,))
            
        elif self.out_bound is False and self.is_mpc_on is True and self.is_manual_on is False:
            self.get_logger().info(f"Publishing MPC PCMD: x={self.x_mpc}, y={self.y_mpc}, z={self.z_mpc}, yaw={self.yaw_mpc}")
            self.drone(PCMD(1,
                            -self.y_mpc,
                            self.x_mpc,
                            -self.yaw_mpc,
                            self.z_mpc,
                            timestampAndSeqNum=0,))

        else:
            self.get_logger().warn("Failed to publish PCMD from Sphinx.")

    def get_current_state(self):

        pose = self.sphinx.get_drone_pose(machine_name=self.name)
        if pose is not None:
            self.position.x = pose[0]
            self.position.y = pose[1]
            self.position.z = pose[2]
            self.position.roll = -pose[3]
            self.position.pitch = pose[4]
            self.position.yaw = pose[5]
        else:
            self.get_logger().warn("Failed to get drone pose from Sphinx.")
        
        
        current_time = self.get_clock().now().nanoseconds / 1e9

        speeds = {"x":0.0, "y":0.0, "z":0.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0}
        positions = {"x":self.position.x, "y":self.position.y, "z":self.position.z,
                     "yaw": self.position.yaw, "roll":self.position.roll, "pitch":self.position.pitch}
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
        
        self.speed.x_speed = speeds["x"]
        self.speed.y_speed = speeds["y"]
        self.speed.z_speed = speeds["z"]
        self.speed.yaw_speed = speeds["yaw"]
        self.speed.roll_speed = speeds["roll"]
        self.speed.pitch_speed = speeds["pitch"]

        self.previous_x = self.position.x
        self.previous_y = self.position.y
        self.previous_z = self.position.z
        self.previous_yaw = self.position.yaw
        self.previous_roll = self.position.roll
        self.previous_pitch = self.position.pitch

        self.previous_time = current_time

        self.current_state.position = self.position
        self.current_state.speed = self.speed
        self.publisher_current_state.publish(self.current_state)

        if self.current_state.position.x > 5.0 or \
        self.current_state.position.x < -5.0 or \
        self.current_state.position.y > 5.0 or \
        self.current_state.position.y < -5.0 or \
        self.current_state.position.z > 3.0 or \
        self.current_state.position.z < 0.8:
            
            self.out_bound = True
        
            e_u = np.zeros(3)
            if self.current_state.position.x > 5.0:
                e_u[0] = -100
            if self.current_state.position.x < -5.0:
                e_u[0] = 100
            if self.current_state.position.y > 5.0:
                e_u[1] = -100
            if self.current_state.position.y < -5.0:
                e_u[1] = 100
            if self.current_state.position.z > 3.0:
                e_u[2] = -100
            if self.current_state.position.z < 0.8:
                e_u[2] = 100
            
            e_R = self.get_R(self.current_state.position.yaw)
            e_R_inv = np.linalg.inv(e_R)
            e_u_world_xy = np.array([e_u[0], e_u[1]])
            e_u_body_xy = e_R_inv @ e_u_world_xy
            self.x_error = int(e_u_body_xy[0])
            self.y_error = int(e_u_body_xy[1])
            self.z_error = int(e_u[2])

        else:
            self.out_bound = False
    
    def get_R(self, yaw):
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        return R

    def Connect(self):
        self.get_logger().info('Connecting to Anafi drone...')
        self.DRONE_IP = os.getenv("DRONE_IP", "10.202.0.1")
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