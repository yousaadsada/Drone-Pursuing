import rclpy
from rclpy.node import Node
from threading import Thread
import os
import numpy as np
import casadi as ca
import do_mpc
from anafi_msg.msg import Position, Output, CurrentState
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from scipy.signal import place_poles
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
from std_msgs.msg import Bool
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
import pysphinx
from pynput.keyboard import Listener, Key
import csv
import olympe
import threading
import logging
logging.getLogger("olympe").setLevel(logging.WARNING)

class MPC_Control(Node):
    def __init__(self):
        super().__init__('mpc_controller')

        self.running = True
        self.connected = False

        self.is_save_data_on = False

        self.Connect()

        self.previous_time_update = True

        self.time_stamp = 0.0

        self.freq_do_pid = 40
        self.freq_publish_pcmd = 40
        self.tolerance = 0.1
        self.out_bound = False
        self.num_intermediate_point = 0
        self.intermediate_points = []
        self.index = 0
        self.parameter = 2

        self.x_manual = 0
        self.y_manual = 0
        self.z_manual = 0
        self.yaw_manual = 0

        self.x_pid = 0
        self.y_pid = 0
        self.z_pid = 0
        self.yaw_pid = 0

        self.previous_time = None

        self.previous_x = None
        self.previous_y = None
        self.previous_z = None

        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None

        self.save_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','move2ref_point_pid')
        os.makedirs(self.save_data_dir, exist_ok=True)
        self.save_data_csv_file = os.path.join(self.save_data_dir, 'drone_data.csv')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sphinx = pysphinx.Sphinx(ip="127.0.0.1", port=8383)
        self.get_logger().info("Drone position publisher node has been started.")
        self.name = self.sphinx.get_default_machine_name()
        
        self.drone_state_publisher = self.create_publisher(CurrentState, '/drone_state', qos_profile)
        self.pid_publisher = self.create_publisher(Output, '/pid', qos_profile)
        self.pcmd_publisher = self.create_publisher(Output, '/mpc_control', qos_profile)

        self.mode = 'manual'

        self.reference_state = CurrentState()
        self.drone_state = CurrentState()
        self.pcmd_value = Output()

        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data')
        self.A = np.loadtxt(os.path.join(data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(data_dir, 'B_matrix.csv'), delimiter=',')
        
        indices_x = [0,4]
        self.A_x = self.A[np.ix_(indices_x, indices_x)]
        self.B_x = self.B[np.ix_(indices_x, [0])]
        indices_y = [1,5]
        self.A_y = self.A[np.ix_(indices_y, indices_y)]
        self.B_y = self.B[np.ix_(indices_y, [1])]
        indices_z = [2,6]
        self.A_z = self.A[np.ix_(indices_z, indices_z)]
        self.B_z = self.B[np.ix_(indices_z, [2])]
        indices_yaw = [3,7]
        self.A_yaw = self.A[np.ix_(indices_yaw, indices_yaw)]
        self.B_yaw = self.B[np.ix_(indices_yaw, [3])]

        # self.desired_poles_x = np.array([0.9, 0.8])
        # self.desired_poles_y = np.array([0.9, 0.8])
        # self.desired_poles_z = np.array([0.9, 0.8])
        # self.desired_poles_yaw = np.array([0.9, 0.8])

        self.desired_poles_x = np.array([0.95, 0.945])
        self.desired_poles_y = np.array([0.95, 0.945])
        self.desired_poles_z = np.array([0.95, 0.945])
        self.desired_poles_yaw = np.array([0.95, 0.945])
        
        self.start_user_input_thread()

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.get_current_state_thread = threading.Thread(target=self.get_current_state_thread_callback)
        self.get_current_state_thread.daemon = True
        self.get_current_state_thread.start()

        self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
        self.publish_pcmd_thread.daemon = True
        self.publish_pcmd_thread.start()

        self.save_data_thread = threading.Thread(target=self.save_data_thread_callback)
        self.save_data_thread.daemon = True
        self.save_data_thread.start()

        self.do_pid_thread = threading.Thread(target=self.do_pid_thread_callback)
        self.do_pid_thread.daemon = True
        self.do_pid_thread.start()

        self.K_x = self.calculate_K(self.A_x, self.B_x, self.desired_poles_x) 
        self.K_y = self.calculate_K(self.A_y, self.B_y, self.desired_poles_y)
        self.K_z = self.calculate_K(self.A_z, self.B_z, self.desired_poles_z)
        self.K_yaw = self.calculate_K(self.A_yaw, self.B_yaw, self.desired_poles_yaw)
    



    def calculate_K(self, A, B, desired_poles):
        result = place_poles(A, B, desired_poles)
        K = result.gain_matrix
        print("State feedback gains K:", K)
        return K
    



    def start_user_input_thread(self):
        input_thread = Thread(target=self.handle_user_input)
        input_thread.daemon = True
        input_thread.start()

    def record_data_init(self):
        self.time_stamp = 0.0
        with open(self.save_data_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['Timestamp', 
                            'Reference X', 'Reference Y', 'Reference Z', 'Reference Yaw',
                            'Current X', 'Current Y', 'Current Z', 'Current Yaw', 
                            'X_output', 'Y_output', 'Z_output', 'Yaw_output'
                            ])
    
    def publish_reference(self, x, y, z, yaw):
        self.reference_state.position.x = x
        self.reference_state.position.y = y
        self.reference_state.position.z = z
        self.reference_state.position.yaw = yaw

    def handle_user_input(self):
        while rclpy.ok():
            try:
                user_input = input('Enter [x y z yaw]: ')
                data = [float(value) for value in user_input.split()]
                if len(data) == 4:
                    print("Pid Start")
                    self.mode = 'pid'
                    self.publish_reference(*data)
                    self.record_data_init()
                    self.is_save_data_on = True
                    print(self.mode)
                    
                else:
                    print("Invalid input. Please enter 4 values.")
            except ValueError:
                print("Invalid input. Please enter numeric values.")




    def on_press(self, key):
        if hasattr(key, 'char') and (key.char == 'w' or key.char == 's' or key.char == 'a' or key.char == 'd' or key.char == 'c' or key.char == 'x' or key.char == 'f' or key.char == 'r'): 
            self.mode = 'manual'

        if key == Key.left:
            # self.get_logger().info("Landing command detected (left key press).")
            self.mode = 'manual'
            time.sleep(0.1)
            try:
                self.drone(Landing())
            except Exception as e:
                self.get_logger().info("Failed to Land.")
            time.sleep(0.5)

        elif key == Key.right:
            # self.get_logger().info("Takeoff command detected (right key press).")
            self.mode = 'manual'
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

            # self.get_logger().info(f"Manual control: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")

    def on_release(self, key):
        if hasattr(key, 'char') and key.char in ['w', 's']:
            self.x_manual = 0
        if hasattr(key, 'char') and key.char in ['a', 'd']:
            self.y_manual = 0
        if hasattr(key, 'char') and key.char in ['r', 'f']:
            self.z_manual = 0
        if hasattr(key, 'char') and key.char in ['x', 'c']:
            self.yaw_manual = 0

        # self.get_logger().info(f"Manual control released: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")




    def publish_pcmd_thread_callback(self):

        while self.running:

            if self.mode == 'manual':
                self.drone(PCMD(1,
                                -self.y_manual,
                                self.x_manual,
                                -self.yaw_manual,
                                self.z_manual,
                                timestampAndSeqNum=0,))
                
            elif self.mode == 'pid':
                self.drone(PCMD(1,
                                -self.y_pid,
                                self.x_pid,
                                -self.yaw_pid,
                                self.z_pid,
                                timestampAndSeqNum=0,))

            else:
                self.drone(PCMD(1,
                                0,
                                0,
                                0,
                                0,
                                timestampAndSeqNum=0,))
        




    def get_current_state_thread_callback(self):

        while self.running:

            pose = self.sphinx.get_drone_pose(machine_name=self.name)
            if pose is not None:
                self.drone_state.position.x = pose[0]
                self.drone_state.position.y = pose[1]
                self.drone_state.position.z = pose[2]
                self.drone_state.position.yaw = pose[5]
            else:
                self.get_logger().warn("Failed to get drone pose from Sphinx.")
            
            
            current_time = self.get_clock().now().nanoseconds / 1e9
                
            if self.previous_time is not None:
                speeds = {"x":0.0, "y":0.0, "z":0.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0}
                positions = {"x":self.drone_state.position.x, "y":self.drone_state.position.y, "z":self.drone_state.position.z,
                            "yaw": self.drone_state.position.yaw, "roll":self.drone_state.position.roll, "pitch":self.drone_state.position.pitch}
                previous_positions = {"x": self.previous_x, "y": self.previous_y, "z": self.previous_z,
                                    "yaw": self.previous_yaw, "roll":self.previous_roll, "pitch":self.previous_pitch}
                
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
                self.previous_time = current_time
        
                self.drone_state.speed.x_speed_world = speeds["x"]
                self.drone_state.speed.y_speed_world = speeds["y"]
                self.drone_state.speed.z_speed = speeds["z"]
                self.drone_state.speed.yaw_speed = speeds["yaw"]

                self.previous_x = self.drone_state.position.x
                self.previous_y = self.drone_state.position.y
                self.previous_z = self.drone_state.position.z
                self.previous_yaw = self.drone_state.position.yaw
                self.previous_roll = self.drone_state.position.roll
                self.previous_pitch = self.drone_state.position.pitch

            if self.previous_time_update == True:
                self.previous_time = current_time
                self.previous_time_update = False

            self.drone_state_publisher.publish(self.drone_state)
            
            time.sleep(0.04)
          


    
    def do_pid_thread_callback(self):

        def get_R(yaw):
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            return R

        while self.running:

            if self.mode == 'pid':
            
                reference_state_x = np.array([self.reference_state.position.x, 0]) 
                reference_state_y = np.array([self.reference_state.position.y, 0]) 
                reference_state_z = np.array([self.reference_state.position.z, 0])
                reference_state_yaw = np.array([self.reference_state.position.yaw, 0])

                
                current_state_x = np.array([self.drone_state.position.x,
                                            self.drone_state.speed.x_speed_world])
                current_state_y = np.array([self.drone_state.position.y,
                                            self.drone_state.speed.y_speed_world])
                current_state_z = np.array([self.drone_state.position.z,
                                            self.drone_state.speed.z_speed])       
                current_state_yaw = np.array([self.drone_state.position.yaw,
                                            self.drone_state.speed.yaw_speed])       

                u_x = -self.K_x.dot(current_state_x - reference_state_x)
                u_y = -self.K_y.dot(current_state_y - reference_state_y)
                u_z = -self.K_z.dot(current_state_z - reference_state_z)
                correct_current_state_yaw = current_state_yaw
                if current_state_yaw[0] - reference_state_yaw[0] > math.pi:
                    correct_current_state_yaw[0] = current_state_yaw[0] - 2 * math.pi
                elif current_state_yaw[0] - reference_state_yaw[0] < -math.pi:
                    correct_current_state_yaw[0] = current_state_yaw[0] + 2 * math.pi
                else:
                    correct_current_state_yaw[0] = current_state_yaw[0]
                u_yaw = -self.K_yaw.dot(correct_current_state_yaw - reference_state_yaw)
                
                R = get_R(self.drone_state.position.yaw)
                R_inv = np.linalg.inv(R)
                u_world_xy = np.array([u_x, u_y])
                u_body_xy = R_inv @ u_world_xy

                u = np.zeros(4)
                u[0] = u_body_xy[0] 
                u[1] = u_body_xy[1] 
                u[2] = u_z
                u[3] = u_yaw

                if u[0] > 20:
                    u[0] = int(20)
                if u[0] < -20:
                    u[0] = int(-20)
                if u[0] < 1 and u[0] > 0:
                    u[0] = int(1)
                if u[0] > -1 and u[0] < 0:
                    u[0] = int(-1)

                if u[1] > 20:
                    u[1] = int(20)
                if u[1] < -20:
                    u[1] = int(-20)
                if u[1] < 1 and u[1] > 0:
                    u[1] = int(1)
                if u[1] > -1 and u[1] < 0:
                    u[1] = int(-1)

                if u[2] > 50:
                    u[2] = int(50)
                if u[2] < -50:
                    u[2] = int(-50)
                if u[2] < 1 and u[2] > 0:
                    u[2] = int(1)
                if u[2] > -1 and u[2] < 0:
                    u[2] = int(-1)

                if u[3] > 100:
                    u[3] = int(100)
                if u[3] < -100:
                    u[3] = int(-100)
                if u[3] < 1 and u[3] > 0:
                    u[3] = int(1)
                if u[3] > -1 and u[3] < 0:
                    u[3] = int(-1)

                self.pcmd_value.control_x = int(u[0])
                self.pcmd_value.control_y = int(u[1])
                self.pcmd_value.control_z = int(u[2])
                self.pcmd_value.control_yaw = int(u[3])

                self.x_pid = self.pcmd_value.control_x
                self.y_pid = self.pcmd_value.control_y 
                self.z_pid = self.pcmd_value.control_z 
                self.yaw_pid = self.pcmd_value.control_yaw

                self.pid_publisher.publish(self.pcmd_value)
        
            time.sleep(0.01)




    def save_data_thread_callback(self):

        while self.running:

            x = self.drone_state.position.x
            y = self.drone_state.position.y
            z = self.drone_state.position.z
            yaw = self.drone_state.position.yaw
            x_ref = self.reference_state.position.x
            y_ref = self.reference_state.position.y
            z_ref = self.reference_state.position.z
            yaw_ref = self.reference_state.position.yaw

            # distance = math.sqrt((x - x_ref) ** 2 + (y - y_ref) ** 2 + (z - z_ref) ** 2 + (yaw - yaw_ref) ** 2)
            # if distance < 0.05:
            #     self.is_save_data_on = False
            if self.time_stamp > 8:
                self.is_save_data_on = False

            data = [self.time_stamp,
                    round(self.reference_state.position.x, 3), round(self.reference_state.position.y, 3), round(self.reference_state.position.z, 3), round(self.reference_state.position.yaw, 3), 
                    round(self.drone_state.position.x, 3), round(self.drone_state.position.y, 3),round(self.drone_state.position.z, 3),round(self.drone_state.position.yaw, 3),
                    self.x_pid, self.y_pid, self.z_pid, self.yaw_pid
                    ]
                    
            if self.is_save_data_on == True:
                with open(self.save_data_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
            
            self.time_stamp += 0.04

            time.sleep(0.04)




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
    node = MPC_Control()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()