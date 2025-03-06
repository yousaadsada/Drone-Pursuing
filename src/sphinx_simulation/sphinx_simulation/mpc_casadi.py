import rclpy
from rclpy.node import Node
import warnings
from sensor_msgs.msg import Image
import cv2, os, olympe, psutil, time, threading, queue
from cv_bridge import CvBridge
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from olympe.messages.ardrone3.PilotingState import FlyingStateChanged, SpeedChanged, AttitudeChanged
from anafi_msg.msg import PnPData
from pynput.keyboard import Listener, Key
import csv
import os
import casadi as ca
import numpy as np
from anafi_msg.msg import CurrentState, Output, Position, Speed
import math
import pysphinx

warnings.simplefilter("ignore")
olympe.log.update_config({"loggers": {"olympe": {"level": "CRITICAL"}}})


class DroneMPC(Node):
    def __init__(self):
        super().__init__('af_pursuer')
        self.running = True
        self.connected = False
        self.Connect()

        self.sphinx = pysphinx.Sphinx(ip="127.0.0.1", port=8383)
        self.get_logger().info("Drone position publisher node has been started.")
        self.name = self.sphinx.get_default_machine_name()

        self.publisher_pcmd = self.create_publisher(Output, '/pub_pcmd',1)
        self.publisher_current_state = self.create_publisher(CurrentState, '/current_state_pos',1)

        load_data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data')
        self.A = np.loadtxt(os.path.join(load_data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(load_data_dir, 'B_matrix.csv'), delimiter=',')

        self.time_stamp = 0.0
        self.is_save_data_on = False

        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        self.previous_yaw = None
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_time = None


        self.freq = 25
        self.nx = 8
        self.nu = 4

        self.mpc_intervel = 0.04
        self.predictive_horizon = 50

        self.x_manual = 0
        self.y_manual = 0
        self.z_manual = 0
        self.yaw_manual = 0
        self.x_mpc = 0
        self.y_mpc = 0
        self.z_mpc = 0
        self.yaw_mpc = 0

        self.mpc_or_manual = 'manual'

        self.save_data_dir = os.path.join(os.path.expanduser("~"),'anafi_simulation', 'data', 'simulation')
        os.makedirs(self.save_data_dir, exist_ok=True)
        self.save_data_csv_file = os.path.join(self.save_data_dir, 'mpc_move2points.csv')

        self.speed = Speed()
        self.position = Position()
        self.anafi_state = CurrentState()
        self.reference_state = CurrentState()
        self.pcmd = Output()

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.daemon = True

        self.handle_input_thread = threading.Thread(target=self.handle_input_thread_callback)
        self.handle_input_thread.daemon = True
 
        self.get_anafi_state_thread = threading.Thread(target=self.get_anafi_state_thread_callback)
        self.get_anafi_state_thread.daemon = True

        self.mpc_controller_init()
        self.do_mpc_thread = threading.Thread(target=self.do_mpc_thread_callback)
        self.do_mpc_thread.daemon = True

        self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
        self.publish_pcmd_thread.daemon = True

        self.save_data_thread = threading.Thread(target=self.save_data_thread_callback)
        self.save_data_thread.daemon = True
    
        self.start_thread()

        self.move_anafi()

    def move_anafi(self):
        time.sleep(10)

        self.drone(Landing())

        time.sleep(2)

        handling_script = """
        [
        {
            "stop_condition_expr": "step_time > 2",
            "first_pose": {
            "x_expr": "0",
            "y_expr": "0",
            "z_expr": "0.5",
            "roll_expr": "0",
            "pitch_expr": "0",
            "yaw_expr": "0",
            "reset_all_forces": "true"
            }
        }
        ]
        """

        handling = self.sphinx.get_component("anafi_ai", "userscript", "handling")
        handling.set_param("script", handling_script)
        handling.trig_action("start")

       
        time.sleep(2)
        self.drone(TakeOff())


    def start_thread(self):
        self.listener.start()
        self.handle_input_thread.start()
        self.get_anafi_state_thread.start()
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




    def handle_input_thread_callback(self):
    
        def save_data_init():
            with open(self.save_data_csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write CSV header
                writer.writerow(['Timestamp', 
                                'Anafi X', 'Anafi Y', 'Anafi Z',
                                'Anafi X_speed', 'Anafi Y_speed', 'Anafi Z_speed',  
                                'X Input', 'Y Input', 'Z Input', 'Yaw Input'
                                ])
    
        while rclpy.ok():
            try:
                user_input = input('Enter [x y z yaw]: ')
                data = [float(value) for value in user_input.split()]
                if len(data) == 4:
                    self.time_stamp = 0.0
                    save_data_init()
                    self.set_reference(*data)
                    self.mpc_or_manual = 'mpc'
                    self.is_save_data_on = True
                else:
                    print("Invalid input. Please enter 4 values.")
            except ValueError:
                print("Invalid input. Please enter numeric values.")

    def set_reference(self, x, y, z, yaw):
        self.reference_state.position.x = x
        self.reference_state.position.y = y
        self.reference_state.position.z = z
        self.reference_state.position.yaw = yaw
        self.reference_state.speed.x_speed_world = 0.0
        self.reference_state.speed.y_speed_world = 0.0
        self.reference_state.speed.x_speed = 0.0
        self.reference_state.speed.y_speed = 0.0
        self.reference_state.speed.z_speed = 0.0
        self.reference_state.speed.yaw_speed = 0.0






    def get_anafi_state_thread_callback(self):
        while self.running:
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

            self.anafi_state.position.x = self.position.x
            self.anafi_state.position.y = self.position.y
            self.anafi_state.position.z = self.position.z
            self.anafi_state.position.yaw = self.position.yaw
            self.anafi_state.speed.x_speed = self.speed.x_speed * math.cos(self.position.yaw) + self.speed.y_speed * math.sin(self.position.yaw)
            self.anafi_state.speed.y_speed = self.speed.x_speed * (- math.sin(self.position.yaw)) + self.speed.y_speed * math.cos(self.position.yaw)
            self.anafi_state.speed.x_speed_world = self.speed.x_speed
            self.anafi_state.speed.y_speed_world = self.speed.y_speed
            self.anafi_state.speed.z_speed = self.speed.z_speed
            self.anafi_state.speed.yaw_speed = self.speed.yaw_speed

            self.publisher_current_state.publish(self.anafi_state)

            time.sleep(0.01)

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
        
        x_input_min = -20
        y_input_min = -20
        z_input_min = -50
        yaw_input_min = -100
        x_input_max = 20
        y_input_max = 20
        z_input_max = 50
        yaw_input_max = 100


        # Define cost function
    
        Q = np.diag([1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1])
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
            # else:
                # cost_fn += (st - x_ref).T @ Q @ (st - x_ref) * 0.005

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

            x_anafi = self.anafi_state.position.x
            y_anafi = self.anafi_state.position.y
            z_anafi = self.anafi_state.position.z
            yaw_anafi = self.anafi_state.position.yaw
            v_x_anafi = self.anafi_state.speed.x_speed
            v_y_anafi = self.anafi_state.speed.y_speed
            v_z_anafi = self.anafi_state.speed.z_speed
            v_yaw_anafi = self.anafi_state.speed.yaw_speed

            data = [self.time_stamp, 
                    round(x_anafi, 3), round(y_anafi, 3), round(z_anafi, 3), round(yaw_anafi, 3), 
                    round(v_x_anafi, 3), round(v_y_anafi, 3), round(v_z_anafi, 3), round(v_yaw_anafi, 3), 
                    self.x_mpc, self.y_mpc, self.z_mpc, self.yaw_mpc
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
        # self.DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
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

    def Stop(self):
        self.running = False

        self.destroy_node()
        rclpy.shutdown()




def main():
    rclpy.init()
    mpc_controller = DroneMPC()
    try:
        while rclpy.ok():
            rclpy.spin(mpc_controller)
    except:
        mpc_controller.Stop()

if __name__ == "__main__":
    main()
