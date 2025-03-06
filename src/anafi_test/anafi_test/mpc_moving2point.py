import rclpy
from rclpy.node import Node
from anafi_msg.msg import CurrentState, Output, Position
import numpy as np
import casadi as ca
import os
import threading
import cvxpy as cp
import rclpy.time
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
        self.DRONE_IP = os.getenv("DRONE_IP", "192.168.42.1")
        self.drone = olympe.Drone(self.DRONE_IP)
        self.target_frame = 'anafi'
        
        self.state_lock = threading.Lock()

        self.freq_do_mpc = 25
        self.freq_publish_pcmd = 25
        self.n_x = 8
        self.n_u = 4
        self.horizon = 100
        self.reference = np.zeros(self.n_x)

        load_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','smoothed_data02','state_function')
        self.A = np.loadtxt(os.path.join(load_data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(load_data_dir, 'B_matrix.csv'), delimiter=',')

        self.save_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','move2ref_point')
        os.makedirs(self.save_data_dir, exist_ok=True)
        self.save_data_csv_file = os.path.join(self.save_data_dir, 'drone_data.csv')

        self.running = True
        self.connected = False
        self.is_mpc_on = False
        self.is_manual_on = True
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

        self.reference_point_publisher = self.create_publisher(Position, '/reference_point', 10)
        self.publish_current_state = self.create_publisher(CurrentState, '/simulation_current_state',10)
        self.publisher_pcmd = self.create_publisher(Output, '/pub_pcmd',10)
        self.subscribe_drone_state = self.create_subscription(TFMessage, '/tf', self.subscribe_drone_state_callback, 10)
       
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        self.state_lock = threading.Lock()
        
        self.handle_input_thread = threading.Thread(target=self.handle_input_thread_callback)
        self.handle_input_thread.daemon = True
        
        # self.subscribe_drone_state_thread = threading.Thread(target=self.subscribe_drone_state_thread_callback)
        # self.subscribe_drone_state_thread.daemon = True
 
        self.do_mpc_thread = threading.Thread(target=self.do_mpc_thread_callback)
        self.do_mpc_thread.daemon = True

        self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
        self.publish_pcmd_thread.daemon = True

        self.save_data_init()
        self.save_data_thread = threading.Thread(target=self.save_data_thread_callback)
        self.save_data_thread.daemon = True

        self.setup_drone_connection()

        self.start_threads()




    def setup_drone_connection(self):
        start_time = time.time()

        while not self.connected and (time.time() - start_time < 3):  # Retry for up to 3 seconds
            try:
                self.drone = olympe.Drone("192.168.42.1")  # Replace with the correct IP if needed
                connected = self.drone.connect()
                if connected:
                    self.get_logger().info("Connected to Anafi drone!")
                    self.connected = True
            except Exception as e:
                self.get_logger().error(f"Failed to connect to drone: {e}")
                time.sleep(0.5)  # Wait briefly before retrying

        if not connected:
            self.get_logger().error("Unable to connect to Anafi drone after 3 seconds.")


    
    def start_threads(self):
        # Start threads after connection is successful
        self.handle_input_thread.start()
        # self.subscribe_drone_state_thread.start()
        self.do_mpc_thread.start()
        self.publish_pcmd_thread.start()
        self.save_data_thread.start()


    def on_press(self, key):
        if hasattr(key, 'char') and (key.char == 'w' or key.char == 's' or key.char == 'a' or key.char == 'd' or key.char == 'c' or key.char == 'x' or key.char == 'f' or key.char == 'r'): 
            self.is_mpc_on = False
            self.is_manual_on = True

        if key == Key.left:
            #self.get_logger().info("Landing command detected (left key press).")
            time.sleep(0.1)
            self.is_mpc_on = False
            self.is_manual_on = True
            try:
                self.drone(Landing())
            except Exception as e:
                self.get_logger().info("Failed to Land.")
            time.sleep(0.5)

        elif key == Key.right:
            #self.get_logger().info("Takeoff command detected (right key press).")
            time.sleep(0.1)
            self.is_mpc_on = False
            self.is_manual_on = True
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

    def handle_input_thread_callback(self):
        while rclpy.ok():
            try:
                user_input = input('Enter [x y z yaw]: ')
                data = [float(value) for value in user_input.split()]
                if len(data) == 4:
                    self.publish_reference(*data)
                    self.is_mpc_on = True
                    self.is_manual_on = False
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
                current_position = self.drone_state.position.yaw
                if self.previous_yaw is not None:
                    if current_position < 0 and self.previous_yaw > 0 and current_position < -0.9 * math.pi and self.previous_yaw  > 0.9 * math.pi:
                        delta_position = 2 * math.pi + current_position - self.previous_yaw
                    elif current_position > 0 and self.previous_yaw < 0 and current_position > 0.9 * math.pi and self.previous_yaw  < -0.9 * math.pi:
                        delta_position = -2 * math.pi + current_position - self.previous_yaw
                    else:
                        delta_position = current_position - self.previous_yaw
            
                    delta_time = current_time - self.previous_time
                    self.drone_state.speed.yaw_speed = delta_position / delta_time
        
            self.previous_yaw = self.drone_state.position.yaw
            self.previous_time = current_time

        for transform in msg.transforms:
            if transform.child_frame_id == self.target_frame:
                process_transform(transform)

        speed_dict = self.drone.get_state(SpeedChanged)
        self.drone_state.speed.x_speed = -speed_dict['speedX']
        self.drone_state.speed.y_speed = speed_dict['speedY']
        self.drone_state.speed.z_speed = -speed_dict['speedZ']
        self.publish_current_state.publish(self.drone_state)
    
    # def subscribe_drone_state_thread_callback(self):
    #     self.subscribe_drone_state = self.create_subscription(TFMessage, '/tf', self.subscribe_drone_state_callback, 10)
    #     while self.running:
    #         rclpy.spin_once(self)
    



    def do_mpc_thread_callback(self):
        
        while self.running:

            if self.is_mpc_on == True or self.is_manual_on is False:
                start_time = time.time()
                
                def get_R(yaw):
                    R = np.array([
                        [np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]
                    ])
                    return R

                Q_p = np.eye(self.n_x)
                Q_f = np.eye(self.n_x)
                Q_f[4, 4] = 0.1
                Q_f[5, 5] = 0.1
                Q_f[6, 6] = 0.1
                Q_f[7, 7] = 0.1
                

                # Constraints
                u_min = np.array([-20, -20, -40, -200])
                u_max = np.array([20, 20, 40, 200])

                correct_current_yaw = self.drone_state.position.yaw
                if self.drone_state.position.yaw - self.reference_point.yaw > math.pi:
                    correct_current_yaw = self.drone_state.position.yaw - 2 * math.pi
                elif self.drone_state.position.yaw - self.reference_point.yaw < -math.pi:
                    correct_current_yaw = self.drone_state.position.yaw + 2 * math.pi
                else:
                    correct_current_yaw = self.drone_state.position.yaw

                # Initial state
                x0 = np.array([self.drone_state.position.x,
                            self.drone_state.position.y,
                            self.drone_state.position.z,
                            correct_current_yaw,
                            self.drone_state.speed.x_speed,
                            self.drone_state.speed.y_speed,
                            self.drone_state.speed.z_speed,
                            self.drone_state.speed.yaw_speed
                            ])
                
                # Reference state
                x_ref = np.array([self.reference_point.x,
                                self.reference_point.y,
                                self.reference_point.z,
                                self.reference_point.yaw,
                                0.0,
                                0.0,
                                0.0,
                                0.0])
                
                distance_square = float(0.0)
                for i in range(self.n_u):
                    distance_square += (x0[i] - x_ref[i]) ** 2
                distance = distance_square ** (1/2)

                correct_horizon = int(self.horizon * distance)
                if correct_horizon < int(50):
                    correct_horizon = int(50)
                if correct_horizon > int(100):
                    correct_horizon = int(100)

                # Define optimization variables
                x = cp.Variable((self.n_x, correct_horizon + 1))
                u = cp.Variable((self.n_u, correct_horizon))

                # Define the cost function
                cost = 0
                constraints = [x[:, 0] == x0]

                for k in range(correct_horizon):
                    # if k < 20:
                    #     u_k = u[:, k]
                    # else:
                    #     u_k = u[:, 19]
                    u_k = u[:, k]
                    #cost += cp.quad_form(u[:, k], R)
                    #cost += 0.0001 * cp.quad_form(x[:, k] - x_ref, Q_p)
                    constraints += [x[:, k + 1] == self.A @ x[:, k] + self.B @ u_k]
                    constraints += [u_min <= u_k, u_k <= u_max]

                # Terminal cost
                cost += cp.quad_form(x[:, correct_horizon] - x_ref, Q_f)

                # Define the optimization problem
                problem = cp.Problem(cp.Minimize(cost), constraints)

                # Solve the problem
                problem.solve(solver = cp.ECOS)

                # Extract the optimal control inputs
                u_opt = u.value[:, 0]  # Use the first control input
                #print("Optimal control inputs:", u_opt)

                R = get_R(self.drone_state.position.yaw)
                R_inv = np.linalg.inv(R)
                u_world_xy = np.array([u_opt[0], u_opt[1]])
                u_body_xy = R_inv @ u_world_xy

                u = np.zeros(4)
                u[0] = u_body_xy[0] 
                u[1] = u_body_xy[1] 
                u[2] = u_opt[2]
                u[3] = u_opt[3]

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
                print(time_usage)


            time.sleep(0.04)




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

            elif self.out_bound is False and self.is_mpc_on is False and self.is_manual_on is True:
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
                
            elif self.out_bound is False and self.is_mpc_on is True and self.is_manual_on is False:
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
    
            #time.sleep(0.04)

    


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

        rate = self.create_rate(25)  
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