import rclpy
import olympe
from rclpy.node import Node
from std_msgs.msg import String, Bool
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from olympe.messages.ardrone3.PilotingState import SpeedChanged
from geometry_msgs.msg import Vector3
import time
import os
from anafi_msg.msg import Position, Speed, Output, CurrentState, CollectCurrentState
import threading
import csv
import numpy as np
import casadi as ca
import do_mpc
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from scipy.signal import place_poles
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from pynput.keyboard import Listener, Key
import logging
import math
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import transforms3d
from scipy.fft import fft, ifft, fftfreq
from rclpy.executors import MultiThreadedExecutor


logging.getLogger("olympe").setLevel(logging.WARNING)
olympe.log.update_config({"loggers": {"olympe": {"level": "CRITICAL"}}})

class CollectDataNode(Node):
    def __init__(self):
        super().__init__('collect_data_node')
        self.connected = False
        self.running = True
        self.previous_time_update = True

        self.Connect()
        
        self.write_data_flag = False
        self.freq = 25
        self.speed_level = None
        self.is_test_on = False
        self.test_axis = None        
        self.csv_file = None  # Initialize csv_file as None
        self.running = True
        self.connected = False

        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None
        self.previous_time = None

        self.mpc_or_manual = 'manual'

        self.freq_do_mpc = 25
        self.freq_publish_pcmd = 25
        self.nx = 8
        self.nu = 4
        self.mpc_intervel = 0.04
        self.predictive_horizon = 50

        self.target_frame = 'anafi'
        self.cumulative_time_stamp = 0.0
        self.time_stamp = 0.0
        self.pcmd = Output()
        self.anafi_state = CurrentState()
        self.correct_current_state = CurrentState()
        self.reference_state = CurrentState()

        load_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','smoothed_data02','state_function')
        self.A = np.loadtxt(os.path.join(load_data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(load_data_dir, 'B_matrix.csv'), delimiter=',')


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

        self.freq_publish_pcmd = 25
        self.duration = 3
        self.update_interval = 0.01
        self.num_iterations = int(self.duration / self.update_interval)

        self.previous_time = None
        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None

        self.subscribe_drone_tf = self.create_subscription(TFMessage, '/tf',self.drone_tf_callback, 10)
        self.publish_ref_state = self.create_publisher(CurrentState, '/ref_state', 10)
        self.publish_drone_state = self.create_publisher(CurrentState, '/drone_state', 10)
        self.publish_pcmd = self.create_publisher(Output, '/pcmd', 10)
        
        self.data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'mpc_delay')
        os.makedirs(self.data_dir, exist_ok=True) 

        self.handle_user_input_thread = threading.Thread(target=self.handle_user_input_thread_callback)
        self.handle_user_input_thread.daemon = True
        self.handle_user_input_thread.start()

        self.process_test_thread = threading.Thread(target=self.process_test_thread_callback)
        self.process_test_thread.daemon = True
        self.process_test_thread.start()

        self.mpc_controller_init()
        self.do_mpc_thread = threading.Thread(target=self.do_mpc_thread_callback)
        self.do_mpc_thread.daemon = True
        self.do_mpc_thread.start()

        self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
        self.publish_pcmd_thread.daemon = True
        self.publish_pcmd_thread.start()

        self.write_data_thread = threading.Thread(target=self.write_data_thread_callback)
        self.write_data_thread.daemon = True
        self.write_data_thread.start()
        
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()




    def on_press(self, key):
        if hasattr(key, 'char') and (key.char == 'w' or key.char == 's' or key.char == 'a' or key.char == 'd' or key.char == 'c' or key.char == 'x' or key.char == 'f' or key.char == 'r'): 
            self.mpc_or_manual = 'manual'
            self.is_test_on = False
            self.write_data_flag = False

        if key == Key.left:
            self.mpc_or_manual = 'manual'
            self.is_test_on = False
            self.write_data_flag = False
            self.get_logger().info("Landing command detected (left key press).")
            time.sleep(0.1)
            self.drone(Landing())
            time.sleep(0.5)

        elif key == Key.right:
            self.mpc_or_manual = 'manual'
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

    def on_release(self, key):
        if hasattr(key, 'char') and key.char in ['w', 's']:
            self.x_manual = 0
        if hasattr(key, 'char') and key.char in ['a', 'd']:
            self.y_manual = 0
        if hasattr(key, 'char') and key.char in ['r', 'f']:
            self.z_manual = 0
        if hasattr(key, 'char') and key.char in ['x', 'c']:
            self.yaw_manual = 0




    def drone_tf_callback(self, msg):    
        for transform in msg.transforms:
            if transform.child_frame_id == self.target_frame:
                self.process_transform(transform)
    
    def process_transform(self, transform: TransformStamped):
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        
        x = translation.x
        y = translation.y + 1
        z = translation.z 

        quaternion = (
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w
        )

        euler = transforms3d.euler.quat2euler(quaternion)
        yaw, pitch, roll = euler[0], euler[1], euler[2]
        
        self.anafi_state.position.x = x
        self.anafi_state.position.y = y
        self.anafi_state.position.z = z 
        self.anafi_state.position.yaw = yaw
        self.anafi_state.position.pitch = pitch
        self.anafi_state.position.roll = roll 

        current_time = self.get_clock().now().nanoseconds / 1e9
            
        if self.previous_time is not None:
            if current_time - self.previous_time >= 0.04:
                speeds = {"x":0.0, "y":0.0, "z":0.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0}
                positions = {"x":self.anafi_state.position.x, "y":self.anafi_state.position.y, "z":self.anafi_state.position.z,
                            "yaw": self.anafi_state.position.yaw, "roll":self.anafi_state.position.roll, "pitch":self.anafi_state.position.pitch}
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
        
                self.anafi_state.speed.x_speed_world = speeds["x"]
                self.anafi_state.speed.y_speed_world = speeds["y"]
                self.anafi_state.speed.z_speed = speeds["z"]
                self.anafi_state.speed.yaw_speed = speeds["yaw"]

                self.previous_x = self.anafi_state.position.x
                self.previous_y = self.anafi_state.position.y
                self.previous_z = self.anafi_state.position.z
                self.previous_yaw = self.anafi_state.position.yaw
                self.previous_roll = self.anafi_state.position.roll
                self.previous_pitch = self.anafi_state.position.pitch

        if self.previous_time_update == True:
            self.previous_time = current_time
            self.previous_time_update = False





    def handle_user_input_thread_callback(self):
        while rclpy.ok():
            try:
                user_input = input('Enter [1 for x, 2 for y, or 3 for z]: ').strip().lower()
                if user_input in ['1', '2', '3']:
                    if user_input == '1':
                        self.test_axis = 'x'
                        print(f"Test axis set to {self.test_axis}.")
                    elif user_input == '2':
                        self.test_axis = 'y'
                        print(f"Test axis set to {self.test_axis}.")
                    elif user_input == '3':
                        self.test_axis = 'z'
                        print(f"Test axis set to {self.test_axis}.")

                    ref_moving_speed_command = input('Enter [1 or 2 or 3 or 4 or 5 or 6 or 7 or 8 or 9 or 10 for speed level]').strip()
                    if ref_moving_speed_command in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                        save_csv_name_number = int(ref_moving_speed_command)
                        self.speed_level = int(ref_moving_speed_command) * 0.1

                        start_command = input('Type "1" to begin the test: ')
                        if start_command == "1":                    
                            self.mpc_or_manual = 'mpc'
                            self.is_test_on = True
                            self.csv_file = os.path.join(self.data_dir, f'state_data_{self.test_axis}_{save_csv_name_number}.csv')
                            self.write_csv_header()  # Write CSV header after setting the file name
                            self.time_stamp = 0.0
                            break
                        else: 
                            print("Invalid command. Please type 'start' to begin.")
                    else:
                        print("Invalid speed level input. Please enter 1, 2, 3, 4 or 5.")
                else:
                    print("Invalid axis input. Please enter x, y or z.")
            except ValueError:
                print("Invalid input!")

    def write_csv_header(self):
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time_stamp',
                             'ref_moving_speed',
                             'ref_x', 'ref_y', 'ref_z',
                             'drone_x', 'drone_y', 'drone_z'])




    def process_test_thread_callback(self):

        while self.running:
            if self.is_test_on == True:
                try:
                    self.reference_state.position.x = 0.0
                    self.reference_state.position.y = 0.0
                    self.reference_state.position.z = 1.0
                    self.reference_state.position.yaw = 0.0
                    self.reference_state.speed.x_speed_world = 0.0
                    self.reference_state.speed.y_speed_world = 0.0
                    self.reference_state.speed.z_speed = 0.0
                    self.reference_state.speed.yaw_speed = 0.0

                    time.sleep(10)  # Additional delay if needed
                    self.write_data_flag = True 

                    for _ in range(self.num_iterations):
                        if self.test_axis == 'x':
                            self.reference_state.position.x += self.speed_level * self.update_interval 
                        elif self.test_axis == 'y':
                            self.reference_state.position.y += self.speed_level * self.update_interval
                        elif self.test_axis == 'z':
                            self.reference_state.position.z += self.speed_level * self.update_interval * 0.6

                        time.sleep(self.update_interval)
                        self.time_stamp += 0.01

                    self.write_data_flag = False  # Stop writing data after test

                    
                finally:
                    print('Test finished')
                    self.time_stamp = 0.0
                    self.is_test_on = False
            
            else:
                pass




    def mpc_controller_init(self):

        t = self.mpc_intervel 
        h = self.predictive_horizon 

        a_x = self.A[4,4]
        a_y = self.A[5,5]
        a_z = self.A[6,6]
        a_yaw = self.A[7,7]
        b_x = self.B[4,0]
        b_y = self.B[5,1]
        b_z = self.B[6,2]
        b_yaw = self.B[7,3]

        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        z = ca.SX.sym('z')
        yaw = ca.SX.sym('yaw')
        v_x = ca.SX.sym('v_x')
        v_y = ca.SX.sym('v_y')
        v_z = ca.SX.sym('v_z')
        v_yaw = ca.SX.sym('v_yaw')
        
        x_input = ca.SX.sym('x_input')  
        y_input = ca.SX.sym('y_input')  
        z_input = ca.SX.sym('z_input')  
        yaw_input = ca.SX.sym('yaw_input')  

        # Define state and control vectors
        states = ca.vertcat(x, y, z, yaw, v_x, v_y, v_z, v_yaw)
        controls = ca.vertcat(x_input, y_input, z_input, yaw_input)

        # Define the system dynamics
        next_states = ca.vertcat(
            x + t * v_x,
            y + t * v_y, 
            z + t * v_z,
            yaw + t * v_yaw,
            a_x * v_x + b_x * x_input,
            a_y * v_y + b_y * y_input,
            a_z * v_z + b_z * z_input,
            a_yaw * v_yaw + b_yaw * yaw_input
        )
        
        f = ca.Function('f', [states, controls], [next_states])

        # Optimization variables
        U = ca.SX.sym('U', 4, h)  # Control inputs over the horizon (v, w)
        X = ca.SX.sym('X', 8, h + 1)  # State variables over the horizon (x, y, theta)
        
        x_input_min = -30
        y_input_min = -30
        z_input_min = -35
        yaw_input_min = -100
        x_input_max = 30
        y_input_max = 30
        z_input_max = 35
        yaw_input_max = 100


        # Define cost function
    
        Q = np.diag([1, 1, 1, 1, 0.1, 0.1, 0.01, 0.01])
        delta_R = np.diag([0.0, 0.0, 0.0, 0.0])
        R = np.diag([0.0, 0.0, 0.0, 0.0])   

        self.lbx = np.concatenate(
            [np.full(self.nx * (h + 1), -ca.inf), np.tile([x_input_min, y_input_min, z_input_min, yaw_input_min], h)]
        )
        self.ubx = np.concatenate(
            [np.full(self.nx * (h + 1), ca.inf), np.tile([x_input_max, y_input_max, z_input_max, yaw_input_max], h)]
        )
        
        cost_fn = 0
        g = []

        P = ca.SX.sym('P', 2 * self.nx)

        g.append(X[:,0] - P[:self.nx])

        # Loop over the prediction horizon
        for k in range(h):
            st = X[:, k]
            con = U[:, k]
            x_ref = P[self.nx:]

            if k == h-1:
                cost_fn += (st - x_ref).T @ Q @ (st - x_ref)
            else:
                cost_fn += (st - x_ref).T @ Q @ (st - x_ref) * 0.000002

            cost_fn += con.T @ R @ con
            if k < h - 1:
                delta_U = U[:, k+1] - U[:, k]
                cost_fn += delta_U.T @ delta_R @ delta_U
    
            st_next = X[:, k+1]
            f_value = f(st, con)
            g.append(st_next - f_value)  # Dynamics constraint

        # Concatenate constraints and optimization variables
        g = ca.vertcat(*g)
        OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Define optimization problem
        nlp_prob = {
            'f':cost_fn,
            'x':OPT_variables,
            'g':g,
            'p':P
        }

        opts = {
            'ipopt.max_iter':1000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-6
        }

        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def do_mpc_thread_callback(self):
        def get_R(yaw):
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            return R

        while self.running:

            if self.mpc_or_manual == 'mpc':

                h = self.predictive_horizon
                n_states = self.nx
                n_controls = self.nu

                self.correct_current_state = self.anafi_state

                if self.anafi_state.position.yaw - self.reference_state.position.yaw > math.pi:
                    self.correct_current_state.position.yaw = self.anafi_state.position.yaw - 2 * math.pi
                elif self.anafi_state.position.yaw - self.reference_state.position.yaw < -math.pi:
                    self.correct_current_state.position.yaw = self.anafi_state.position.yaw + 2 * math.pi
                else:
                    self.correct_current_state.position.yaw = self.anafi_state.position.yaw
                
                x_current_state = np.array([self.correct_current_state.position.x,
                                            self.correct_current_state.position.y, 
                                            self.correct_current_state.position.z, 
                                            self.correct_current_state.position.yaw,
                                            self.correct_current_state.speed.x_speed_world, 
                                            self.correct_current_state.speed.y_speed_world, 
                                            self.correct_current_state.speed.z_speed, 
                                            self.correct_current_state.speed.yaw_speed])
                
                x_ref = np.array([self.reference_state.position.x, 
                                  self.reference_state.position.y, 
                                  self.reference_state.position.z, 
                                  self.reference_state.position.yaw,
                                  self.reference_state.speed.x_speed_world, 
                                  self.reference_state.speed.y_speed_world, 
                                  self.reference_state.speed.z_speed, 
                                  self.reference_state.speed.yaw_speed])

                u0 = np.zeros((n_controls * h, 1 ))
                u0 = u0.flatten()
                x_init = np.tile(x_current_state, (h + 1, 1)).T.flatten()
                P = np.concatenate((x_current_state, x_ref))
                
                args = {
                    'x0': np.concatenate([x_init, u0]),  # Initial guess for states and controls
                    'lbx': self.lbx,
                    'ubx': self.ubx,
                    'lbg': np.zeros((n_states * (h + 1),)),  # Lower bounds on constraints
                    'ubg': np.zeros((n_states * (h + 1),)),  # Upper bounds on constraints
                    'p': P  # Pass the current state and reference as parameters
                }

                sol = self.solver(**args)

                u_opt = sol['x'][n_states * (h + 1):].full().reshape((h, n_controls))

                R = get_R(self.anafi_state.position.yaw)
                R_inv = np.linalg.inv(R)
                u_world_xy = np.array([u_opt[0, 0], u_opt[0, 1]])
                u_body_xy = R_inv @ u_world_xy

                u = np.zeros(4)
                u[0] = u_body_xy[0] 
                u[1] = u_body_xy[1] 
                u[2] = u_opt[0, 2]
                u[3] = u_opt[0, 3]


                if u[0] < 1 and u[0] > 0:
                    u[0] = int(1)
                if u[0] > -1 and u[0] < 0:
                    u[0] = int(-1)

                if u[1] < 1 and u[1] > 0:
                    u[1] = int(1)
                if u[1] > -1 and u[1] < 0:
                    u[1] = int(-1)

                if u[2] < 1 and u[2] > 0.5:
                    u[2] = int(1)
                if u[2] > -1 and u[2] < -0.5:
                    u[2] = int(-1)

                if u[3] < 1 and u[3] > 0:
                    u[3] = int(1)
                if u[3] > -1 and u[3] < 0:
                    u[3] = int(-1)

                # Publish control command
                self.x_mpc = int(u[0])
                self.y_mpc = int(u[1])
                self.z_mpc = int(u[2])
                self.yaw_mpc = int(u[3])

            time.sleep(0.01)




    def publish_pcmd_thread_callback(self):

        while self.running:
    
            if self.mpc_or_manual == 'manual':
                #self.get_logger().info(f"Publishing manual PCMD: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")
                self.drone(PCMD(1,
                                -self.y_manual,
                                self.x_manual,
                                -self.yaw_manual,
                                self.z_manual,
                                timestampAndSeqNum=0,))
                
                self.pcmd.control_x = self.x_manual
                self.pcmd.control_y = self.y_manual
                self.pcmd.control_z = self.z_manual
                self.pcmd.control_yaw = self.yaw_manual
                
            elif self.mpc_or_manual == 'mpc':
                #self.get_logger().info(f"Publishing MPC PCMD: x={self.x_control}, y={self.y_control}, z={self.z_control}, yaw={self.yaw_control}")
                self.drone(PCMD(1,
                                -self.y_mpc,
                                self.x_mpc,
                                -self.yaw_mpc,
                                self.z_mpc,
                                timestampAndSeqNum=0,))
                
                self.pcmd.control_x = self.x_mpc
                self.pcmd.control_y = self.y_mpc
                self.pcmd.control_z = self.z_mpc
                self.pcmd.control_yaw = self.yaw_mpc
            
            else:
                self.get_logger().warn("Failed to publish PCMD from Sphinx.")

                self.drone(PCMD(1,
                    0,
                    0,
                    0,
                    0,
                    timestampAndSeqNum=0,))
                
                self.pcmd.control_x = 0
                self.pcmd.control_y = 0
                self.pcmd.control_z = 0
                self.pcmd.control_yaw = 0
            
            self.publish_pcmd.publish(self.pcmd)
            self.publish_drone_state.publish(self.anafi_state)
            self.publish_ref_state.publish(self.reference_state)




    def write_data_thread_callback(self):
        while self.running:
            # self.get_logger().info("Saving data...")

            formatted_time_stamp = f"{self.time_stamp:.2f}"
            ref_moving_speed = self.speed_level
            ref_x = self.reference_state.position.x
            ref_y = self.reference_state.position.y
            ref_z = self.reference_state.position.z
            drone_x = self.anafi_state.position.x
            drone_y = self.anafi_state.position.y
            drone_z = self.anafi_state.position.z

            data = [formatted_time_stamp,
                    ref_moving_speed,
                    ref_x,
                    ref_y,
                    ref_z,
                    drone_x,
                    drone_y,
                    drone_z]
            
            if self.write_data_flag == True:
                with open(self.csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
                

            time.sleep(0.04)




    def Connect(self):
        self.get_logger().info('Connecting to Anafi drone...')
        self.DRONE_IP = os.getenv("DRONE_IP", "192.168.42.1")
        #self.DRONE_IP = os.getenv("DRONE_IP", "10.202.0.1")
        self.DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")
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
    rclpy.init()
    node = CollectDataNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()  # Spin with multiple threads
    except KeyboardInterrupt:
        node.Stop()
    finally:
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
