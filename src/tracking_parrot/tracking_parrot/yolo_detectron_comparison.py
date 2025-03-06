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
        self.Connect()
        
        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None
        self.previous_time = None
        self.previous_yaw = None

        self.identified_number_yolo = 0
        self.identified_number_detectron = 0
        self.total_number_yolo = 0
        self.total_number_detectron = 0


        self.R_revise = np.array([
                                    [0.9969, -0.0784, 0.0],
                                    [0.0784,  0.9969, 0.0],
                                    [0.0,     0.0,    1.0]
                                    ])


        self.publisher_pcmd = self.create_publisher(Output, '/pub_pcmd',10)
        self.publisher_anafi_state = self.create_publisher(CurrentState, '/anafi_state',10)
        self.publisher_reference_state = self.create_publisher(CurrentState, '/reference_state',10)
        self.image_pub = self.create_publisher(Image, '/anafi/frames', 1)
        self.pub_plot_data = self.create_publisher(PlotData, '/plotdata', 1)

        self.pos_sub_detectron = self.create_subscription(PnPData,'/anafi/pnp', self.get_position_callback_detectron, 1)
        self.pos_sub_yolo = self.create_subscription(PnPDataYolo,'/position', self.get_position_callback_yolo, 1)
        self.subscribe_anafi_state = self.create_subscription(CurrentState, '/anafi_state_raw', self.subscribe_anafi_state_callback, 1)
        self.subscribe_parrot_state = self.create_subscription(CurrentState, '/parrot_state_raw', self.subscribe_parrot_state_callback, 1)
        
        load_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','smoothed_data02','state_function')
        self.A = np.loadtxt(os.path.join(load_data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(load_data_dir, 'B_matrix.csv'), delimiter=',')

        self.freq = 25
        self.nx = 8
        self.nu = 4

        self.mpc_intervel = 0.04
        self.predictive_horizon = 50
        self.distance = 3.0

        self.x_manual = 0
        self.y_manual = 0
        self.z_manual = 0
        self.yaw_manual = 0
        self.x_mpc = 0
        self.y_mpc = 0
        self.z_mpc = 0
        self.yaw_mpc = 0

        self.mpc_or_manual = 'manual'

        

        self.save_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','tracking_parrot_yolo_test')
        os.makedirs(self.save_data_dir, exist_ok=True)
        self.save_data_csv_file = os.path.join(self.save_data_dir, 'drone_data_yolo_detectron_comparison_moving.csv')

        

        self.parrot_pos = PnPData()
        self.plot_data = PlotData()
        self.anafi_state = CurrentState()
        self.raw_state_yolo = CurrentState()
        self.raw_state_detectron = CurrentState()
        self.anafi_state_record = CurrentState()
        self.parrot_state_record = CurrentState()
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
        self.anafi_state.speed.x_speed_world = msg.speed.x_speed_world
        self.anafi_state.speed.y_speed_world = msg.speed.y_speed_world
        self.anafi_state.speed.z_speed = msg.speed.z_speed
        self.anafi_state.speed.yaw_speed = msg.speed.yaw_speed
        self.anafi_state.position.yaw = msg.position.yaw
        self.anafi_state.position.x = 0.0
        self.anafi_state.position.y = 0.0
        self.anafi_state.position.z = 0.0
        self.publisher_anafi_state.publish(self.anafi_state)

        self.anafi_state_record.position.x = msg.position.x
        self.anafi_state_record.position.y = msg.position.y
        self.anafi_state_record.position.z = msg.position.z
        self.anafi_state_record.position.yaw = msg.position.yaw

    def subscribe_parrot_state_callback(self, msg):
        self.parrot_state_record.position.x = msg.position.x
        self.parrot_state_record.position.y = msg.position.y
        self.parrot_state_record.position.z = msg.position.z



    

    def get_position_callback_yolo(self, msg):
        
    
        if msg.target == False:
            self.raw_state_yolo.position.x = 0.0
            self.raw_state_yolo.position.y = 0.0
            self.raw_state_yolo.position.z = 0.0
        
        elif msg.target == True:
            self.raw_state_yolo.position.x = msg.tz
            self.raw_state_yolo.position.y = -msg.tx
            self.raw_state_yolo.position.z = -msg.ty
            self.identified_number_yolo += 1
        
        self.total_number_yolo += 1
        

    def get_position_callback_detectron(self, msg):
        self.parrot_pos = msg

        if self.parrot_pos.target == False:
            self.raw_state_detectron.position.x = 0.0
            self.raw_state_detectron.position.y = 0.0
            self.raw_state_detectron.position.z = 0.0

        
        elif self.parrot_pos.target == True:
            self.raw_state_detectron.position.x = self.parrot_pos.tz
            self.raw_state_detectron.position.y = -self.parrot_pos.tx
            self.raw_state_detectron.position.z = -self.parrot_pos.ty
            self.identified_number_detectron += 1
        
        self.total_number_detectron += 1





    def get_position_callback(self, msg):

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
        
        def kf_update(measured_position):
            Z = np.array(measured_position).reshape(3, 1)
            K = self.S @ self.H.T @ np.linalg.inv(self.H @ self.S @ self.H.T + self.R)
            self.x = self.x + K @ (Z - self.H @ self.x)
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

        
        R_anafi2world = get_rotation_matrix(0.0, 0.0, self.anafi_state.position.yaw)
        T_anafi2world = get_transformation_matrix(self.anafi_state.position.x, self.anafi_state.position.y, self.anafi_state.position.z, R_anafi2world)

        P_anafi = np.array([self.relative_state.position.x, self.relative_state.position.y, self.relative_state.position.z, 1])
        P_world = np.dot(T_anafi2world, P_anafi)

        self.parrot_state.position.x = P_world[0]
        self.parrot_state.position.y = P_world[1]
        self.parrot_state.position.z = P_world[2]

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
            self.kalman_filter_flag = True
        
        if self.kalman_filter_flag == True:
            kf_predict()
            measured_position = [self.parrot_state.position.x, self.parrot_state.position.y, self.parrot_state.position.z]
            kf_update(measured_position)

            self.parrot_state_predictive.position.x = self.x[0,0]
            self.parrot_state_predictive.position.y = self.x[1,0]
            self.parrot_state_predictive.position.z = self.x[2,0]
            self.parrot_state_predictive.speed.x_speed_world = self.x[3,0]
            self.parrot_state_predictive.speed.y_speed_world = self.x[4,0]
            self.parrot_state_predictive.speed.z_speed = self.x[5,0]

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
        



    def save_data_init(self):
        with open(self.save_data_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(['Timestamp', 
                            'Parrot X', 'Parrot Y', 'Parrot Z', 
                            'Anafi X', 'Anafi Y', 'Anafi Z', 
                            'Relative Pos Ground Truth X', 'Relative Pos Ground Truth Y', 'Relative Pos Ground Truth Z',
                            'Relative Pos Yolo Calibration X', 'Relative Pos Yolo Calibration Y', 'Relative Pos Yolo Calibration Z',
                            'Relative Pos Detectron Calibration X', 'Relative Pos Detectron Calibration Y', 'Relative Pos Detectron Calibration Z',
                            'YOLO Identify Percentage', 'Detectron Identify Percentage'
                            ])
    
    def start_tracking(self):
        if not self.connected:
            messagebox.showerror("Error", "Drone not connected.")
            return

        self.time_stamp = 0.0
        self.total_number_yolo = 0
        self.total_number_detectron = 0
        self.identified_number_yolo = 0
        self.identified_number_detectron = 0

        self.save_data_init()
        self.mpc_or_manual = 'mpc'
        self.is_save_data_on = True

        self.get_logger().info("Start tracking process initiated.")
        print("Start tracking")




    def save_data_thread_callback(self):

        while self.running:

            def Calibrate(R_revise_1, R_revise_2, relative_pos_x, relative_pos_y, relative_pos_z):
                x_y = np.array([relative_pos_x, relative_pos_y])
                x_y_refine = R_revise_1 @ x_y
                relative_pos_x = x_y_refine[0]
                relative_pos_y = x_y_refine[1]
                x_y_z = np.array([relative_pos_x, relative_pos_y, relative_pos_z])

                x_y_z_revise = R_revise_2 @ x_y_z
                relative_pos_calibrate_x = x_y_z_revise[0]
                relative_pos_calibrate_y = x_y_z_revise[1]
                relative_pos_calibrate_z = x_y_z_revise[2]

                return relative_pos_calibrate_x, relative_pos_calibrate_y, relative_pos_calibrate_z

            def get_R(yaw):
                R = np.array([[np.cos(yaw), -np.sin(yaw)],
                            [np.sin(yaw),  np.cos(yaw)]])
                return R

            x_parrot = self.parrot_state_record.position.x
            y_parrot = self.parrot_state_record.position.y
            z_parrot = self.parrot_state_record.position.z

            x_anafi = self.anafi_state_record.position.x
            y_anafi = self.anafi_state_record.position.y
            z_anafi = self.anafi_state_record.position.z
            yaw_anafi = self.anafi_state_record.position.yaw

            relative_pos_real_x = x_parrot - x_anafi
            relative_pos_real_y = y_parrot - y_anafi
            relative_pos_real_z = z_parrot - z_anafi

            relative_pos_yolo_x = self.raw_state_yolo.position.x 
            relative_pos_yolo_y = self.raw_state_yolo.position.y
            relative_pos_yolo_z = self.raw_state_yolo.position.z

            relative_pos_detectron_x = self.raw_state_detectron.position.x 
            relative_pos_detectron_y = self.raw_state_detectron.position.y
            relative_pos_detectron_z = self.raw_state_detectron.position.z
            
            R = get_R(yaw_anafi)

            relative_pos_yolo_revise_x, relative_pos_yolo_revise_y, relative_pos_yolo_revise_z = Calibrate(R, self.R_revise,relative_pos_yolo_x,relative_pos_yolo_y,relative_pos_yolo_z)
            relative_pos_detectron_revise_x, relative_pos_detectron_revise_y, relative_pos_detectron_revise_z = Calibrate(R, self.R_revise,relative_pos_detectron_x,relative_pos_detectron_y,relative_pos_detectron_z)

            if self.total_number_yolo != 0:
                yolo_identify_percentage = self.identified_number_yolo / self.total_number_yolo
            else:
                yolo_identify_percentage = 0.0

            if self.total_number_detectron != 0:
                detectron_identify_percentage = self.identified_number_detectron / self.total_number_detectron
            else:
                detectron_identify_percentage = 0.0


            self.parrot_pose_yolo_x = x_anafi + relative_pos_yolo_revise_x
            self.parrot_pose_yolo_y = y_anafi + relative_pos_yolo_revise_y
            self.parrot_pose_yolo_z = z_anafi + relative_pos_yolo_revise_z

            

            data = [self.time_stamp,
                    round(x_parrot, 3), round(y_parrot, 3), round(z_parrot, 3), 
                    round(x_anafi, 3), round(y_anafi, 3), round(z_anafi, 3),
                    round(relative_pos_real_x, 3), round(relative_pos_real_y, 3), round(relative_pos_real_z, 3), 
                    round(relative_pos_yolo_revise_x, 3), round(relative_pos_yolo_revise_y, 3), round(relative_pos_yolo_revise_z, 3),
                    round(relative_pos_detectron_revise_x, 3), round(relative_pos_detectron_revise_y, 3), round(relative_pos_detectron_revise_z, 3),
                    round(yolo_identify_percentage, 3), round(detectron_identify_percentage, 3)
                    ]
            
            if self.time_stamp >= 40.0:
                self.is_save_data_on = False
                    
            if self.is_save_data_on == True:
                with open(self.save_data_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
                
                self.time_stamp += 0.04

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

