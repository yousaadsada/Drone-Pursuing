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

logging.getLogger("olympe").setLevel(logging.WARNING)

class MPC_Control(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        self.running = True
        self.connected = False
        self.previous_time_update = True

        self.DRONE_IP = os.getenv("DRONE_IP", "192.168.42.1")
        self.drone = olympe.Drone(self.DRONE_IP)
        self.target_frame = 'anafi_1'
        
        self.state_lock = threading.Lock()

        self.freq_do_mpc = 25
        self.freq_publish_pcmd = 25
        self.nx = 8
        self.nu = 4
        self.reference = np.zeros(self.nx)

        self.mpc_intervel = 0.04
        self.predictive_horizon = 50

        load_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','smoothed_data02','state_function')
        self.A = np.loadtxt(os.path.join(load_data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(load_data_dir, 'B_matrix.csv'), delimiter=',')

        self.save_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','move2ref_point')
        os.makedirs(self.save_data_dir, exist_ok=True)
        self.save_data_csv_file = os.path.join(self.save_data_dir, 'drone_data.csv')
        
        self.mpc_or_manual = 'manual'
        self.out_bound = False
        self.takeoff = False
        self.land = False
        self.is_save_data_on = False

        self.time_stamp = 0.0

        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.gaz = 0

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

        self.position = Position()
        self.speed = Speed()
        self.drone_state = CurrentState()
        self.anafi_state = CurrentState()
        self.reference_state = CurrentState()
        self.reference_point = Position()
        self.pcmd = Output()

        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None
        self.previous_time = None

        # self.sphinx = pysphinx.Sphinx(ip="127.0.0.1", port=8383)
        # self.get_logger().info("Drone position publisher node has been started.")
        # self.name = self.sphinx.get_default_machine_name()

        self.Connect()

        self.reference_point_publisher = self.create_publisher(Position, '/reference_point', 10)
        self.publish_current_state = self.create_publisher(CurrentState, '/simulation_current_state',10)
        self.publisher_pcmd = self.create_publisher(Output, '/pub_pcmd',1)
        self.subscribe_drone_state = self.create_subscription(TFMessage, '/tf', self.subscribe_drone_state_callback, 10)
       
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        
        self.handle_input_thread = threading.Thread(target=self.handle_input_thread_callback)
        self.handle_input_thread.daemon = True
        
        self.mpc_controller_init()
        self.do_mpc_thread = threading.Thread(target=self.do_mpc_thread_callback)
        self.do_mpc_thread.daemon = True

        self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
        self.publish_pcmd_thread.daemon = True

        self.save_data_init()
        self.save_data_thread = threading.Thread(target=self.save_data_thread_callback)
        self.save_data_thread.daemon = True

        self.start_threads()




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


    
    def start_threads(self):
        self.handle_input_thread.start()
        self.do_mpc_thread.start()
        self.publish_pcmd_thread.start()
        self.save_data_thread.start()




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




    def publish_reference(self, x, y, z, yaw):
        self.reference_point.x = x
        self.reference_point.y = y
        self.reference_point.z = z
        self.reference_point.yaw = yaw
        self.reference_point_publisher.publish(self.reference_point)

        self.reference_state.position.x = x
        self.reference_state.position.y = y
        self.reference_state.position.z = z
        self.reference_state.position.yaw = yaw
        self.reference_state.speed.x_speed = 0.0
        self.reference_state.speed.y_speed = 0.0
        self.reference_state.speed.z_speed = 0.0
        self.reference_state.speed.yaw_speed = 0.0

    def handle_input_thread_callback(self):
        while rclpy.ok():
            try:
                user_input = input('Enter [x y z yaw]: ')
                data = [float(value) for value in user_input.split()]
                if len(data) == 4:
                    self.publish_reference(*data)
                    self.mpc_or_manual = 'mpc'
                    self.is_save_data_on = True
                    self.time_stamp = 0.0
                    self.save_data_init()
                else:
                    print("Invalid input. Please enter 4 values.")
            except ValueError:
                print("Invalid input. Please enter numeric values.")




    def subscribe_drone_state_callback(self, msg):
 
        def process_transform(transform: TransformStamped):
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

            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.previous_time is not None:
                if current_time - self.previous_time >= 0.04:
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
                    # self.drone_state.speed.roll_speed = speeds["roll"]
                    # self.drone_state.speed.pitch_speed = speeds["pitch"]

                    self.previous_x = self.drone_state.position.x
                    self.previous_y = self.drone_state.position.y
                    self.previous_z = self.drone_state.position.z
                    self.previous_yaw = self.drone_state.position.yaw
                    self.previous_roll = self.drone_state.position.roll
                    self.previous_pitch = self.drone_state.position.pitch

            if self.previous_time_update == True:
                self.previous_time = current_time
                self.previous_time_update = False
          

        for transform in msg.transforms:
            if transform.child_frame_id == self.target_frame:
                process_transform(transform)

        # speed_dict = self.drone.get_state(SpeedChanged)
        # self.drone_state.speed.x_speed = -speed_dict['speedX']
        # self.drone_state.speed.y_speed = speed_dict['speedY']
        # self.drone_state.speed.z_speed = -speed_dict['speedZ']
        self.publish_current_state.publish(self.drone_state)

        self.anafi_state = self.drone_state




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
                start_time = time.time()

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

                finish_time = time.time()
                time_usage = finish_time - start_time
                #print(time_usage) 

            time.sleep(0.01)





    def publish_pcmd_thread_callback(self):

        while self.running:
           
            if self.out_bound is True:
                self.pcmd.control_x = self.x_error
                self.pcmd.control_y = self.y_error
                self.pcmd.control_z = self.z_error
                self.pcmd.control_yaw = 0

                self.drone(PCMD(1,
                                -self.y_error,
                                self.x_error,
                                0,
                                self.z_error,
                                timestampAndSeqNum=0,))

            elif self.out_bound is False and self.mpc_or_manual == 'manual':
                self.pcmd.control_x = self.x_manual
                self.pcmd.control_y = self.y_manual
                self.pcmd.control_z = self.z_manual
                self.pcmd.control_yaw = self.yaw_manual

                self.drone(PCMD(1,
                                -self.y_manual,
                                self.x_manual,
                                -self.yaw_manual,
                                self.z_manual,
                                timestampAndSeqNum=0,))
                
            elif self.out_bound is False and self.mpc_or_manual == 'mpc':
                self.pcmd.control_x = self.x_mpc
                self.pcmd.control_y = self.y_mpc
                self.pcmd.control_z = self.z_mpc
                self.pcmd.control_yaw = self.yaw_mpc

                # self.get_logger().info(f"Publishing MPC PCMD: x={self.x_mpc}, y={self.y_mpc}, z={self.z_mpc}, yaw={self.yaw_mpc}")   
                self.drone(PCMD(1,
                                -self.y_mpc,
                                self.x_mpc,
                                -self.yaw_mpc,
                                self.z_mpc,
                                timestampAndSeqNum=0,))

            else:
                self.get_logger().warn("Failed to publish PCMD from Sphinx.")
    
            self.publisher_pcmd.publish(self.pcmd)
    

    


    def save_data_init(self):
        with open(self.save_data_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['Timestamp', 
                            'Reference X', 'Reference Y', 'Reference Z', 'Reference Yaw',
                            'Current X', 'Current Y', 'Current Z', 'Current Yaw', 
                            'Current X_speed', 'Current Y_speed', 'Current Z_speed', 'Current Yaw_speed', 
                            'X_output', 'Y_output', 'Z_output', 'Yaw_output'
                            ])
    



    def save_data_thread_callback(self):

        while self.running:

            x = self.drone_state.position.x
            y = self.drone_state.position.y
            z = self.drone_state.position.z
            yaw = self.drone_state.position.yaw
            x_ref = self.reference_point.x
            y_ref = self.reference_point.y
            z_ref = self.reference_point.z
            yaw_ref = self.reference_point.yaw

            distance = math.sqrt((x - x_ref) ** 2 + (y - y_ref) ** 2 + (z - z_ref) ** 2 + (yaw - yaw_ref) ** 2)
            if distance < 0.1:
                self.is_save_data_on = False

            data = [self.time_stamp,
                    round(self.reference_point.x, 3), round(self.reference_point.y, 3), round(self.reference_point.z, 3), round(self.reference_point.yaw, 3), 
                    round(self.drone_state.position.x, 3), round(self.drone_state.position.y, 3),round(self.drone_state.position.z, 3),round(self.drone_state.position.yaw, 3),
                    round(self.drone_state.speed.x_speed, 3), round(self.drone_state.speed.y_speed, 3), round(self.drone_state.speed.z_speed, 3), round(self.drone_state.speed.yaw_speed, 3),
                    self.x_mpc, self.y_mpc, self.z_mpc, self.yaw_mpc
                    ]
                    
            if self.is_save_data_on == True:
                with open(self.save_data_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
            
            self.time_stamp += 0.04

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
    node = MPC_Control()
    run_control_loop(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()