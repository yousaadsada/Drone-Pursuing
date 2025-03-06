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
from anafi_msg.msg import CurrentState, Output, PlotData
import math
import tkinter as tk
from tkinter import messagebox
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import transforms3d
from rclpy.executors import MultiThreadedExecutor
import matplotlib
matplotlib.use("TkAgg")  # Ensure GUI support when running in ROS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter


warnings.simplefilter("ignore")
olympe.log.update_config({"loggers": {"olympe": {"level": "CRITICAL"}}})


class DronePursuer(Node):
    def __init__(self):
        super().__init__('af_pursuer')

        self.target_frame = 'anafi'

        self.yuv_frame_processing_thread = threading.Thread(target=self.yuv_frame_processing_thread_callback)
        self.yuv_frame_processing_thread.daemon = True
        self.time_stamp = 0.0

        self.running = True
        self.connected = False
        self.is_save_data_on = False
        self.previous_time_update = True
        self.kalman_filter_flag = False
        self.Connect()

        self.previous_time_parrot_velocity = None
        self.parrot_previous_x = 0.0
        self.parrot_previous_y = 0.0
        self.parrot_previous_z = 0.0
        
        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None
        self.previous_time = None
        self.previous_yaw = None

        self.previous_time_speed_updata = None
        self.current_time_speed_updata = None
        self.dt = None

        self.parrot_position_buffer_x = []
        self.parrot_position_buffer_y = []
        self.parrot_position_buffer_z = []

        self.parrot_velocity_buffer_x = []
        self.parrot_velocity_buffer_y = []
        self.parrot_velocity_buffer_z = []

        delta_t = 1 / 28.5

        self.x = np.zeros((6,1))
        self.x_vicon = np.zeros((6,1))

        self.A_K = np.array([
                  [1,0,0,delta_t,0,0],
                  [0,1,0,0,delta_t,0],
                  [0,0,1,0,0,delta_t],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]
        ])

        self.H = np.array([
                  [1,0,0,0,0,0],
                  [0,1,0,0,0,0],
                  [0,0,1,0,0,0]
        ])
        
        process_noise = 1
        measurement_noise = 0.05
        self.Q = np.eye(6) * process_noise  
        self.R = np.eye(3) * measurement_noise**2  
        self.S = np.eye(6)

        self.R_revise = np.array([
                            [0.9969, -0.0784, 0.0],
                            [0.0784,  0.9969, 0.0],
                            [0.0,     0.0,    1.0]
                            ])


        self.publisher_pcmd = self.create_publisher(Output, '/pub_pcmd',10)
        self.publisher_anafi_state = self.create_publisher(CurrentState, '/anafi_state',10)
        self.publisher_reference_state = self.create_publisher(CurrentState, '/reference_state',10)
        self.publisher_parrot_state = self.create_publisher(CurrentState, '/parrot_state',10)
        self.publisher_parrot_state_predictive = self.create_publisher(CurrentState, '/parrot_state_predictive',10)
        self.image_pub = self.create_publisher(Image, '/anafi/frames', 1)
        self.pub_plot_data = self.create_publisher(PlotData, '/plotdata', 1)
        self.pos_sub = self.create_subscription(PnPDataYolo,'/position', self.get_position_callback, 1)
        self.subscribe_anafi_state = self.create_subscription(CurrentState, '/anafi_state_raw', self.subscribe_anafi_state_callback, 1)
        self.subscribe_parrot_state = self.create_subscription(CurrentState, '/parrot_state_with_speed', self.subscribe_parrot_state_callback, 1)
        
        load_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','smoothed_data02','state_function')
        self.A = np.loadtxt(os.path.join(load_data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(load_data_dir, 'B_matrix.csv'), delimiter=',')

        self.freq = 25
        self.nx = 8
        self.nu = 4

        self.mpc_intervel = 0.04
        self.predictive_horizon = 50
        self.distance = 3.0

        self.window_size = 50
        self.poly_order = 1

        self.time_advanced_x = 0.862
        self.time_advanced_y = 0.868
        self.time_advanced_z = 0.448

        self.x_manual = 0
        self.y_manual = 0
        self.z_manual = 0
        self.yaw_manual = 0
        self.x_mpc = 0
        self.y_mpc = 0
        self.z_mpc = 0
        self.yaw_mpc = 0

        self.mpc_or_manual = 'manual'

        

        self.save_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','tracking_parrot_yolo_speed_prediction')
        os.makedirs(self.save_data_dir, exist_ok=True)
        self.save_data_csv_file = os.path.join(self.save_data_dir, 'drone_data.csv')
        self.save_data_csv_file_yolo_vicon_comparison = os.path.join(self.save_data_dir, 'drone_state_yolo_vicon_comparison.csv')

        

        self.parrot_pos = PnPDataYolo()
        self.plot_data = PlotData()
        self.anafi_state = CurrentState()
        self.relative_state = CurrentState()
        self.parrot_state = CurrentState()
        self.parrot_state_predictive = CurrentState()
        self.reference_state = CurrentState()
        self.anafi_state_record = CurrentState()
        self.parrot_state_record = CurrentState()
        self.parrot_state_record_filter = CurrentState()
        self.pcmd = Output()

        self.frame_queue = queue.LifoQueue()
        self.frame_id_pub = 0
        self.cv2_cvt_color_flag = {
                    olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
                    olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,}
        self.bridge = CvBridge()

        gui_thread = threading.Thread(target=self.create_gui)
        gui_thread.daemon = True
        gui_thread.start()

        # self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        # self.listener.daemon = True
        # self.listener.start()

        self.mpc_controller_init()
        self.do_mpc_thread = threading.Thread(target=self.do_mpc_thread_callback)
        self.do_mpc_thread.daemon = True
        self.do_mpc_thread.start()

        self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
        self.publish_pcmd_thread.daemon = True
        self.publish_pcmd_thread.start()

        self.save_data_thread = threading.Thread(target=self.save_data_thread_callback)
        self.save_data_thread.daemon = True
        self.save_data_thread.start()


    def create_gui(self):
        """Create the Tkinter GUI."""
        root = tk.Tk()
        root.title("Drone Pursuer Control")
        root.geometry("300x150")

        # Add a label
        label = tk.Label(root, text="Control Panel", font=("Helvetica", 14))
        label.pack(pady=10)

        # Add the "Start Tracking" button
        start_button = tk.Button(
            root,
            text="Start Tracking",
            command=self.start_tracking,
            font=("Helvetica", 12),
            bg="green",
            fg="white",
        )
        start_button.pack(pady=20)

        # Run the Tkinter event loop
        root.mainloop()


    
    def subscribe_anafi_state_callback(self, msg):
        self.anafi_state.position.x = msg.position.x
        self.anafi_state.position.y = msg.position.y
        self.anafi_state.position.z = msg.position.z   
        self.anafi_state.position.yaw = msg.position.yaw
        self.anafi_state.speed.x_speed_world = msg.speed.x_speed_world
        self.anafi_state.speed.y_speed_world = msg.speed.y_speed_world
        self.anafi_state.speed.z_speed = msg.speed.z_speed
        self.anafi_state.speed.yaw_speed = msg.speed.yaw_speed

        self.publisher_anafi_state.publish(self.anafi_state)

        self.anafi_state_record.position.x = msg.position.x
        self.anafi_state_record.position.y = msg.position.y
        self.anafi_state_record.position.z = msg.position.z

    def subscribe_parrot_state_callback(self, msg):
        self.parrot_state_record.position.x = msg.position.x
        self.parrot_state_record.position.y = msg.position.y
        self.parrot_state_record.position.z = msg.position.z
        self.parrot_state_record.position.yaw = msg.position.yaw
        self.parrot_state_record.speed.x_speed_world = msg.speed.x_speed_world
        self.parrot_state_record.speed.y_speed_world = msg.speed.y_speed_world
        self.parrot_state_record.speed.z_speed = msg.speed.z_speed
        self.parrot_state_record.speed.yaw_speed = msg.speed.yaw_speed



    def get_position_callback(self, msg):

        def Calibrate(R_revise, relative_pos_x, relative_pos_y, relative_pos_z):
      
            x_y_z = np.array([relative_pos_x, relative_pos_y, relative_pos_z])

            x_y_z_revise = R_revise @ x_y_z
            relative_pos_calibrate_x = x_y_z_revise[0]
            relative_pos_calibrate_y = x_y_z_revise[1]
            relative_pos_calibrate_z = x_y_z_revise[2]

            return relative_pos_calibrate_x, relative_pos_calibrate_y, relative_pos_calibrate_z


        def get_rotation_matrix(roll, pitch, yaw):
            R_x = np.array([[1, 0, 0],
                            [0, np.cos(roll), -np.sin(roll)],
                            [0, np.sin(roll), np.cos(roll)]])

            R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                            [0, 1, 0],
                            [-np.sin(pitch), 0, np.cos(pitch)]])

            R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
            
            R = np.dot(R_z, np.dot(R_y, R_x))
            return R
        
        def get_transformation_matrix(x, y, z, R):
            # Create the 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            
            return T


        def kf_predict():
            self.x = self.A_K @ self.x
            self.S = self.A_K @ self.S @ self.A_K.T + self.Q

        def kf_predict_vicon():
            self.x_vicon = self.A_K @ self.x_vicon
            self.S = self.A_K @ self.S @ self.A_K.T + self.Q
        
        def kf_update(measured_position):
            Z = np.array(measured_position).reshape(3, 1)
            K = self.S @ self.H.T @ np.linalg.inv(self.H @ self.S @ self.H.T + self.R)
            self.x = self.x + K @ (Z - self.H @ self.x)
            I = np.eye(6)
            self.S = (I - K @ self.H) @ self.S        
            
        def kf_update_vicon(measured_position):
            Z = np.array(measured_position).reshape(3, 1)
            K = self.S @ self.H.T @ np.linalg.inv(self.H @ self.S @ self.H.T + self.R)
            self.x_vicon = self.x_vicon + K @ (Z - self.H @ self.x_vicon)
            I = np.eye(6)
            self.S = (I - K @ self.H) @ self.S

        


        if msg.target == False:
            self.relative_state.position.x = 0.0
            self.relative_state.position.y = 0.0
            self.relative_state.position.z = 0.0
        
        elif msg.target == True:
            self.relative_state.position.x = msg.tz
            self.relative_state.position.y = -msg.tx
            self.relative_state.position.z = -msg.ty

        
        relative_pos_yolo_x = self.relative_state.position.x 
        relative_pos_yolo_y = self.relative_state.position.y
        relative_pos_yolo_z = self.relative_state.position.z 
        
    
        relative_pos_yolo_revise_x, relative_pos_yolo_revise_y, relative_pos_yolo_revise_z = Calibrate(self.R_revise,relative_pos_yolo_x,relative_pos_yolo_y,relative_pos_yolo_z)


        R_anafi2world = get_rotation_matrix(0.0, 0.0, self.anafi_state.position.yaw)
        T_anafi2world = get_transformation_matrix(self.anafi_state.position.x, self.anafi_state.position.y, self.anafi_state.position.z, R_anafi2world)

        P_anafi = np.array([relative_pos_yolo_revise_x, relative_pos_yolo_revise_y, relative_pos_yolo_revise_z, 1])
        P_world = np.dot(T_anafi2world, P_anafi)

        self.parrot_state.position.x = P_world[0]
        self.parrot_state.position.y = P_world[1]
        self.parrot_state.position.z = P_world[2]

        current_time = self.get_clock().now().nanoseconds / 1e9

        if self.previous_time_parrot_velocity is not None:
            delta_time = current_time - self.previous_time_parrot_velocity
            delta_x = self.parrot_state.position.x - self.parrot_previous_x
            delta_y = self.parrot_state.position.y - self.parrot_previous_y
            delta_z = self.parrot_state.position.z - self.parrot_previous_z
            self.parrot_state.speed.x_speed_world = delta_x / delta_time
            self.parrot_state.speed.y_speed_world = delta_y / delta_time
            self.parrot_state.speed.z_speed = delta_z / delta_time

        
        else:
            self.parrot_state.speed.x_speed_world = 0.0
            self.parrot_state.speed.y_speed_world = 0.0
            self.parrot_state.speed.z_speed = 0.0

        self.previous_time_parrot_velocity = current_time
        self.parrot_previous_x = self.parrot_state.position.x
        self.parrot_previous_y = self.parrot_state.position.y
        self.parrot_previous_z = self.parrot_state.position.z

        self.publisher_parrot_state.publish(self.parrot_state)
        
        if self.kalman_filter_flag == False:
            self.x = np.array([[self.parrot_state.position.x],
                               [self.parrot_state.position.y],
                               [self.parrot_state.position.z],
                               [0.0],
                               [0.0],
                               [0.0]
                                ])  

            self.parrot_state_predictive.position.x = self.parrot_state.position.x
            self.parrot_state_predictive.position.y = self.parrot_state.position.y
            self.parrot_state_predictive.position.z = self.parrot_state.position.z
            self.parrot_state_predictive.speed.x_speed_world = 0.0
            self.parrot_state_predictive.speed.y_speed_world = 0.0
            self.parrot_state_predictive.speed.z_speed = 0.0
            
            self.x_vicon = np.array([[self.parrot_state_record.position.x],
                               [self.parrot_state_record.position.y],
                               [self.parrot_state_record.position.z],
                               [0.0],
                               [0.0],
                               [0.0]
                                ])  

            self.parrot_state_record_filter.position.x = self.parrot_state_record.position.x
            self.parrot_state_record_filter.position.y = self.parrot_state_record.position.y
            self.parrot_state_record_filter.position.z = self.parrot_state_record.position.z
            self.parrot_state_record_filter.speed.x_speed_world = 0.0
            self.parrot_state_record_filter.speed.y_speed_world = 0.0
            self.parrot_state_record_filter.speed.z_speed = 0.0
            
            self.kalman_filter_flag = True

 
        
        else:
            self.A_K = np.array([
                  [1,0,0,delta_time,0,0],
                  [0,1,0,0,delta_time,0],
                  [0,0,1,0,0,delta_time],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]
            ])
            kf_predict()
            measured_position = [self.parrot_state.position.x, self.parrot_state.position.y, self.parrot_state.position.z]
            kf_update(measured_position)

            kf_predict_vicon()
            measured_position_vicon = [self.parrot_state_record.position.x, self.parrot_state_record.position.y, self.parrot_state_record.position.z]
            kf_update_vicon(measured_position_vicon)

            self.parrot_state_predictive.position.x = self.x[0,0]
            self.parrot_state_predictive.position.y = self.x[1,0]
            self.parrot_state_predictive.position.z = self.x[2,0]
            self.parrot_state_predictive.speed.x_speed_world = self.x[3,0]
            self.parrot_state_predictive.speed.y_speed_world = self.x[4,0]
            self.parrot_state_predictive.speed.z_speed = self.x[5,0]


            self.parrot_state_record_filter.position.x = self.x_vicon[0,0]
            self.parrot_state_record_filter.position.y = self.x_vicon[1,0]
            self.parrot_state_record_filter.position.z = self.x_vicon[2,0]
            self.parrot_state_record_filter.speed.x_speed_world = self.x_vicon[3,0]
            self.parrot_state_record_filter.speed.y_speed_world = self.x_vicon[4,0]
            self.parrot_state_record_filter.speed.z_speed = self.x_vicon[5,0]


        self.publisher_parrot_state_predictive.publish(self.parrot_state_predictive)

        self.reference_state.position.x = self.parrot_state_predictive.position.x + self.parrot_state_predictive.speed.x_speed_world * self.time_advanced_x - self.distance
        self.reference_state.position.y = self.parrot_state_predictive.position.y + self.parrot_state_predictive.speed.y_speed_world * self.time_advanced_y
        self.reference_state.position.z = self.parrot_state_predictive.position.z + self.parrot_state_predictive.speed.z_speed * self.time_advanced_z

        self.reference_state.position.yaw = 0.0
        self.reference_state.speed.x_speed_world = 0.0
        self.reference_state.speed.y_speed_world = 0.0
        self.reference_state.speed.z_speed = 0.0
        self.reference_state.speed.yaw_speed = 0.0

        self.publisher_reference_state.publish(self.reference_state)
    

    def on_press(self, key):
        if hasattr(key, 'char') and (key.char == 'w' or key.char == 's' or key.char == 'a' or key.char == 'd' or key.char == 'c' or key.char == 'x' or key.char == 'f' or key.char == 'r'): 
            self.mpc_or_manual = 'manual'
            self.is_save_data_on = False

        if key == Key.left:
            #self.get_logger().info("Landing command detected (left key press).")
            time.sleep(0.1)
            self.mpc_or_manual = 'manual'
            self.is_save_data_on = False
            try:
                self.drone(Landing())
            except Exception as e:
                self.get_logger().info("Failed to Land.")
            time.sleep(0.5)

        elif key == Key.right:
            #self.get_logger().info("Takeoff command detected (right key press).")
            time.sleep(0.1)
            self.mpc_or_manual = 'manual'
            self.is_save_data_on = False
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


    def save_data_init(self):
        with open(self.save_data_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['Timestamp', 
                            'Parrot X', 'Parrot Y', 'Parrot Z', 
                            'Anafi X', 'Anafi Y', 'Anafi Z', 
                            'Relative Pos Real X', 'Relative Pos Real Y', 'Relative Pos Real Z',
                            'Relative Pos Yolo X', 'Relative Pos Yolo Y', 'Relative Pos Yolo Z',
                            'Input X', 'Input Y', 'Input Z',
                            ])
            
        with open(self.save_data_csv_file_yolo_vicon_comparison, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['Timestamp', 
                            'Parrot X RAW', 'Parrot Y RAW', 'Parrot Z RAW', 
                            'Parrot X YOLO', 'Parrot Y YOLO', 'Parrot Z YOLO', 
                            'Parrot X Vicon', 'Parrot Y Vicon', 'Parrot Z Vicon', 
                            'Parrot X_Speed RAW', 'Parrot Y_Speed RAW', 'Parrot Z_Speed RAW', 
                            'Parrot X_Speed YOLO', 'Parrot Y_Speed YOLO', 'Parrot Z_Speed YOLO', 
                            'Parrot X_Speed Vicon', 'Parrot Y_Speed Vicon', 'Parrot Z_Speed Vicon', 
                            ])
    
    def start_tracking(self):
        if not self.connected:
            messagebox.showerror("Error", "Drone not connected.")
            return

        self.time_stamp = 0.0
        self.save_data_init()
        self.mpc_or_manual = 'mpc'
        self.is_save_data_on = True

        self.get_logger().info("Start tracking process initiated.")
        print("Start tracking")


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
                # self.correct_current_state.position.x = 0.0
                # self.correct_current_state.position.y = 0.0
                # self.correct_current_state.position.z = 0.0

                self.publisher_anafi_state.publish(self.correct_current_state)

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

        while self.running and self.connected == True:
           
            if self.mpc_or_manual == 'manual':
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
                
            elif self.mpc_or_manual == 'mpc':
                self.pcmd.control_x = self.x_mpc
                self.pcmd.control_y = self.y_mpc
                self.pcmd.control_z = self.z_mpc
                self.pcmd.control_yaw = self.yaw_mpc

                self.drone(PCMD(1,
                                -self.y_mpc,
                                self.x_mpc,
                                -self.yaw_mpc,
                                self.z_mpc,
                                timestampAndSeqNum=0,))

            else:
                self.drone(PCMD(1,
                                0,
                                0,
                                0,
                                0,
                                timestampAndSeqNum=0,))

                self.get_logger().warn("Failed to publish PCMD from Sphinx.")
    
            self.publisher_pcmd.publish(self.pcmd)




    def save_data_thread_callback(self):

        while self.running:
        
            x_parrot = self.parrot_state_record.position.x
            y_parrot = self.parrot_state_record.position.y
            z_parrot = self.parrot_state_record.position.z

            x_anafi = self.anafi_state_record.position.x
            y_anafi = self.anafi_state_record.position.y
            z_anafi = self.anafi_state_record.position.z

            relative_pos_real_x = x_parrot - x_anafi
            relative_pos_real_y = y_parrot - y_anafi
            relative_pos_real_z = z_parrot - z_anafi

            relative_pos_yolo_x = self.reference_state.position.x + self.distance
            relative_pos_yolo_y = self.reference_state.position.y
            relative_pos_yolo_z = self.reference_state.position.z


            data = [self.time_stamp,
                    round(x_parrot, 3), round(y_parrot, 3), round(z_parrot, 3), 
                    round(x_anafi, 3), round(y_anafi, 3), round(z_anafi, 3),
                    round(relative_pos_real_x, 3), round(relative_pos_real_y, 3), round(relative_pos_real_z, 3), 
                    round(relative_pos_yolo_x, 3), round(relative_pos_yolo_y, 3), round(relative_pos_yolo_z, 3),
                    self.x_mpc, self.y_mpc, self.z_mpc
                    ]
            
            parrot_state_x_yolo = self.parrot_state_predictive.position.x
            parrot_state_y_yolo = self.parrot_state_predictive.position.y
            parrot_state_z_yolo = self.parrot_state_predictive.position.z
            parrot_state_x_speed_yolo = self.parrot_state_predictive.speed.x_speed_world
            parrot_state_y_speed_yolo = self.parrot_state_predictive.speed.y_speed_world
            parrot_state_z_speed_yolo = self.parrot_state_predictive.speed.z_speed

            # parrot_state_x_vicon = self.parrot_state_record.position.x
            # parrot_state_y_vicon = self.parrot_state_record.position.y
            # parrot_state_z_vicon = self.parrot_state_record.position.z
            # parrot_state_x_speed_vicon = self.parrot_state_record.speed.x_speed_world
            # parrot_state_y_speed_vicon = self.parrot_state_record.speed.y_speed_world
            # parrot_state_z_speed_vicon = self.parrot_state_record.speed.z_speed

            parrot_state_x_vicon = self.parrot_state_record_filter.position.x
            parrot_state_y_vicon = self.parrot_state_record_filter.position.y
            parrot_state_z_vicon = self.parrot_state_record_filter.position.z
            parrot_state_x_speed_vicon = self.parrot_state_record_filter.speed.x_speed_world
            parrot_state_y_speed_vicon = self.parrot_state_record_filter.speed.y_speed_world
            parrot_state_z_speed_vicon = self.parrot_state_record_filter.speed.z_speed

            data_yolo_real_comparison = [self.time_stamp,
                                         round(self.parrot_state.position.x, 3), round(self.parrot_state.position.y, 3), round(self.parrot_state.position.z, 3),
                                         round(parrot_state_x_yolo, 3), round(parrot_state_y_yolo, 3), round(parrot_state_z_yolo, 3),
                                         round(parrot_state_x_vicon, 3), round(parrot_state_y_vicon, 3), round(parrot_state_z_vicon, 3),
                                         round(self.parrot_state.speed.x_speed_world, 3), round(self.parrot_state.speed.y_speed_world, 3), round(self.parrot_state.speed.z_speed, 3),
                                         round(parrot_state_x_speed_yolo, 3), round(parrot_state_y_speed_yolo, 3), round(parrot_state_z_speed_yolo, 3),
                                         round(parrot_state_x_speed_vicon, 3), round(parrot_state_y_speed_vicon, 3), round(parrot_state_z_speed_vicon, 3), ]
                    
            if self.is_save_data_on == True:
                with open(self.save_data_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)

                with open(self.save_data_csv_file_yolo_vicon_comparison, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data_yolo_real_comparison)


                self.plot_data.time_stamp = self.time_stamp
                self.plot_data.x_parrot = x_parrot
                self.plot_data.y_parrot = y_parrot
                self.plot_data.z_parrot = z_parrot
                self.plot_data.x_anafi = x_anafi
                self.plot_data.y_anafi = y_anafi
                self.plot_data.z_anafi = z_anafi

                self.pub_plot_data.publish(self.plot_data)

                # self.time_data.append(self.time_stamp)
                # self.x_parrot_data.append(x_parrot)
                # self.y_parrot_data.append(y_parrot)
                # self.z_parrot_data.append(z_parrot)
                # self.x_anafi_data.append(x_anafi)
                # self.y_anafi_data.append(y_anafi)
                # self.z_anafi_data.append(z_anafi)

                
                self.time_stamp += 0.04

            if self.time_stamp >= 20.0:
                self.is_save_data_on = False

            time.sleep(0.04)


 
    def yuv_frame_processing_thread_callback(self):
        while self.running:
            try:
                t = (self.get_clock().now().nanoseconds)/1000000000
                yuv_frame = self.frame_queue.get(timeout=0.1)
                
                if yuv_frame is not None:
                    x = yuv_frame.as_ndarray()
                    cv2frame = cv2.cvtColor(x, self.cv2_cvt_color_flag[yuv_frame.format()])
                    msg = self.bridge.cv2_to_imgmsg(cv2frame, "bgr8")
                    msg.header.frame_id = str(self.frame_id_pub)
                    self.image_pub.publish(msg)
                    self.frame_id_pub += 1
                    yuv_frame.unref()

            except queue.Empty:
                pass
            except Exception as e:
                pass
 

            

    def yuv_frame_cb(self, yuv_frame):
        
        try:
            yuv_frame.ref()
            self.frame_queue.put_nowait(yuv_frame)
        except Exception as e:
            self.get_logger().info(f"Error handling media removal: {e}")

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
                    self.yuv_frame_processing_thread.start()   
                   
                    break
                
                else:
                    self.get_logger().info(f'Trying to connect (%d)' % (i + 1))
                    time.sleep(2)

        if not self.connected:
            self.get_logger().info("Failed to connect.")

    def Stop(self):
        self.running = False
        self.Pursuing_on = False

        self.drone(PCMD(0,0,0,0,0,timestampAndSeqNum=0,))

        if self.connected:
            FlyingState = str(self.drone.get_state(FlyingStateChanged)['state'])
            if FlyingState != 'FlyingStateChanged_State.landed':
                self.drone(Landing()).wait().success()
            self.drone.streaming.stop()
            self.drone.disconnect()

        self.destroy_node()
        rclpy.shutdown()




def main():
    rclpy.init()
    af_pursuer = DronePursuer()

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

