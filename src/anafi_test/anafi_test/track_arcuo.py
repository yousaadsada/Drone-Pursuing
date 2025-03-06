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
import cv2, queue
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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose

logging.getLogger("olympe").setLevel(logging.WARNING)

class MPC_Control(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        self.freq_do_mpc = 25
        self.freq_publish_pcmd = 25
        self.n_x = 8
        self.n_u = 4
        self.horizon = 100
        self.reference = np.zeros(self.n_x)

        self.target_frame = 'anafi'
        self.frameid = 0

        #data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data')
        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data','real_drone_state_function','smoothed_data02','state_function')
        self.A = np.loadtxt(os.path.join(data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(data_dir, 'B_matrix.csv'), delimiter=',')

 
        print(self.A)
        print(self.B)

        self.tracking_distance = 1.5

        self.running = True
        self.connected = False
        self.is_tracking_on = False
        self.is_manual_on = True
        self.out_bound = False

        self.pitch = 0
        self.roll = 0
        self.yaw = 0
        self.gaz = 0
        self.takeoff = False
        self.land = False

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

        self.aruco_pose = Position()
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

        self.bridge = CvBridge()
        self.frame_queue = queue.LifoQueue()
        self.processing_thread = threading.Thread(target=self.yuv_frame_processing)
        self.cv2_cvt_color_flag = {
                    olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
                    olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
                }

        self.mpc_thread = threading.Thread(target=self.mpc_timer_thread)
        self.mpc_thread.start()

        self.timer_pubPCMD = self.create_timer(callback=self.publish_pcmd, timer_period_sec=1/self.freq_publish_pcmd)
    
        self.subscribe_aruco_state = self.create_subscription(Pose, '/aruco_pose', self.subscribe_aruco_pose_callback, 10)
        self.subscribe_drone_state = self.create_subscription(CurrentState, '/drone_current_state', self.subscribe_drone_state_callback, 10)
        self.reference_point_publisher = self.create_publisher(Position, '/reference_point', 10)
        self.image_pub = self.create_publisher(Image, 'anafi/frames', 1000)        
        #self.publish_current_state = self.create_publisher(CurrentState, '/simulation_current_state',10)
        self.publisher_pcmd = self.create_publisher(Output, '/pub_pcmd',10)
       
        self.start_user_input_thread()

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if hasattr(key, 'char') and (key.char == 'w' or key.char == 's' or key.char == 'a' or key.char == 'd' or key.char == 'c' or key.char == 'x' or key.char == 'f' or key.char == 'r'): 
            self.is_tracking_on = False
            self.is_manual_on = True

        if key == Key.left:
            #self.get_logger().info("Landing command detected (left key press).")
            time.sleep(0.1)
            try:
                self.drone(Landing())
            except Exception as e:
                self.get_logger().info("Failed to Land.")
            time.sleep(0.5)

        elif key == Key.right:
            #self.get_logger().info("Takeoff command detected (right key press).")
            time.sleep(0.1)
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
    def start_user_input_thread(self):
        input_thread = threading.Thread(target=self.handle_user_input)
        input_thread.daemon = True
        input_thread.start()

    def handle_user_input(self):
        while rclpy.ok():
            try:
                user_input = input('Press 1 for tracking, 2 for manual control: ')

                if user_input == '1':
                    self.is_tracking_on = True
                    self.is_manual_on = False
                    print("Tracking process started.")
                elif user_input == '2':
                    self.is_tracking_on = False
                    self.is_manual_on = True
                    print("Manual control process started.")
                else:
                    print("Invalid input. Please press 1 for tracking or 2 for maunal control.")
            except ValueError:
                print("Invalid input. Please press 1 for tracking and 2 for maunal control.")

    def publish_pcmd(self):
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

        elif self.out_bound is False and self.is_tracking_on is False and self.is_manual_on is True:
            self.pcmd.control_x = self.x_manual
            self.pcmd.control_y = self.y_manual
            self.pcmd.control_z = self.z_manual
            self.pcmd.control_yaw = self.yaw_manual


           
            #self.get_logger().info(f"Publishing manual PCMD: x={self.x_manual}, y={self.y_manual}, z={self.z_manual}, yaw={self.yaw_manual}")
            self.drone(PCMD(1,
                            -self.y_manual,
                            self.x_manual,
                            -self.yaw_manual,
                            self.z_manual,
                            timestampAndSeqNum=0,))
            
        elif self.out_bound is False and self.is_tracking_on is True and self.is_manual_on is False:
            self.pcmd.control_x = self.x_mpc
            self.pcmd.control_y = self.y_mpc
            self.pcmd.control_z = self.z_mpc
            self.pcmd.control_yaw = self.yaw_mpc
            
            #self.get_logger().info(f"Publishing MPC PCMD: x={self.x_mpc}, y={self.y_mpc}, z={self.z_mpc}, yaw={self.yaw_mpc}")   
            self.drone(PCMD(1,
                            -self.y_mpc,
                            self.x_mpc,
                            -self.yaw_mpc,
                            self.z_mpc,
                            timestampAndSeqNum=0,))

        else:
            self.get_logger().warn("Failed to publish PCMD from Sphinx.")

        #print(self.pcmd)
        
        self.publisher_pcmd.publish(self.pcmd)

    def get_R(self, yaw):
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        return R

    def subscribe_drone_state_callback(self, msg):    
        self.drone_state.position = msg.position
        self.drone_state.speed = msg.speed
        # speed_dict = self.drone.get_state(SpeedChanged)
        # self.drone_state.speed.x_speed = -speed_dict['speedX']
        # self.drone_state.speed.y_speed = speed_dict['speedY']
        # self.drone_state.speed.z_speed = -speed_dict['speedZ']
        #self.publish_current_state.publish(self.drone_state)
    
    def subscribe_aruco_pose_callback(self, msg):
        self.aruco_pose.x = msg.position.x
        self.aruco_pose.y = msg.position.y
        self.aruco_pose.z = msg.position.z
        
        def transform2world():
            R_aruco = self.get_R(self.drone_state.position.yaw)
            x_body = np.array([self.aruco_pose.x, self.aruco_pose.y])
            x_world = np.dot(R_aruco, x_body)
            return x_world

        aruco_world_xy = transform2world()
    
        self.reference_point.x = self.drone_state.position.x + aruco_world_xy[0] - self.tracking_distance
        self.reference_point.y = self.drone_state.position.y + aruco_world_xy[1]
        self.reference_point.z = self.drone_state.position.z + self.aruco_pose.z
        self.reference_point.yaw = 0.0

        self.reference_point_publisher.publish(self.reference_point)
    
    def mpc_timer_thread(self):
        while rclpy.ok():
            self.do_mpc_callback()
            time.sleep(1 / self.freq_do_mpc)


    def do_mpc_callback(self):
        if self.is_tracking_on == False or self.is_manual_on is True:
            return
        
        
        Q_p = np.eye(self.n_x)
        Q_f = np.eye(self.n_x)
        Q_f[7, 7] = 0.1

        # Constraints
        u_min = np.array([-20, -20, -10, -100])
        u_max = np.array([20, 20, 10, 100])

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

        R = self.get_R(self.drone_state.position.yaw)
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


            # if (self.current_state_value.position.x - self.reference_point.x) **2 + \
            # (self.current_state_value.position.y - self.reference_point.y) **2 < 0.1:
                
            #     print("Drop the ball!")
            #     self.shutdown == True

    def yuv_frame_cb(self, yuv_frame):
        
        try:
            yuv_frame.ref()
            self.frame_queue.put_nowait(yuv_frame)

        except Exception as e:
            self.get_logger().info(f"Error handling media removal: {e}")


    def yuv_frame_processing(self):
        t=0
        prev_t = 0
        while self.running:
            try:
                t = (self.get_clock().now().nanoseconds)/1000000000
                yuv_frame = self.frame_queue.get(timeout=0.1)
                
                if yuv_frame is not None:
                    x = yuv_frame.as_ndarray()
                    cv2frame = cv2.cvtColor(x, self.cv2_cvt_color_flag[yuv_frame.format()])
                    msg = self.bridge.cv2_to_imgmsg(cv2frame, "bgr8")
                    msg.header.frame_id = str(self.frameid)
                    self.image_pub.publish(msg)
                    self.frameid += 1
                    yuv_frame.unref()
                    #self.get_logger().info(f'Publishing [{self.frameid}]')
                    #self.get_logger().info(f' // FPS: {round(1/(t-prev_t),2)})')
                
                prev_t = t
            except queue.Empty:
                pass
            except Exception as e:
                self.get_logger().info(f"Error processing frame: {e}")

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
                    self.processing_thread.start()   
                   
                    break
                
                else:
                    self.get_logger().info(f'Trying to connect (%d)' % (i + 1))
                    time.sleep(2)

        if not self.connected:
            self.get_logger().info("Failed to connect.")

    def Stop(self):
        self.running = False
        if self.Connected:
            self.processing_thread.join(timeout=1.0)
            self.drone.streaming.stop()
            self.drone.disconnect()
        
        self.get_logger().info('Shutting down...\n')
        self.image_pub.destroy()
        self.listener.stop()
        self.Connected = False
        time.sleep(0.2)
        self.destroy_node()
        rclpy.shutdown()



def main(args=None):
    rclpy.init(args=args)
    node = MPC_Control()
    node.Connect()
    rclpy.spin(node)


if __name__ == '__main__':
    main()