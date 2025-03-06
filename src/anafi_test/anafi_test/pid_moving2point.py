import rclpy
from rclpy.node import Node
from threading import Thread
import os
import numpy as np
import casadi as ca
import do_mpc
from my_custom_msgs.msg import Position, Output, CurrentState
import time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from scipy.signal import place_poles
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
from std_msgs.msg import Bool

class Pid_Control(Node):
    def __init__(self):
        super().__init__('mpc_controller')
        self.freq_do_pid = 10

        self.do_pid = self.create_timer(timer_period_sec=1 / self.freq_do_pid, callback=self.do_pid_callback)

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.publish_control_onAndoff = self.create_publisher(Bool, '/control_on_off', 10)
        self.reference_point_publisher = self.create_publisher(Position, '/reference_point', qos_profile)
        self.pcmd_publisher = self.create_publisher(Output, '/control_command', qos_profile)
        self.subscription = self.create_subscription(CurrentState, '/drone_current_state', self.current_state_callback, qos_profile)

        self.on_off_flag = Bool()
        self.on_off_flag.data = False
        self.is_pid_on = False
        self.reference_point = Position()
        self.current_state_value = CurrentState()
        self.pcmd_value = Output()

        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data')
        self.A = np.loadtxt(os.path.join(data_dir, 'A_matrix.csv'), delimiter=',')
        self.B = np.loadtxt(os.path.join(data_dir, 'B_matrix.csv'), delimiter=',')
        
        indices_x = [0,4]
        self.A_x = self.A[np.ix_(indices_x, indices_x)]
        self.B_x = self.B[np.ix_(indices_x, [0])]
        print(self.A_x)
        print(self.B_x)
        indices_y = [1,5]
        self.A_y = self.A[np.ix_(indices_y, indices_y)]
        self.B_y = self.B[np.ix_(indices_y, [1])]
        indices_z = [2,6]
        self.A_z = self.A[np.ix_(indices_z, indices_z)]
        self.B_z = self.B[np.ix_(indices_z, [2])]
        indices_yaw = [3,7]
        self.A_yaw = self.A[np.ix_(indices_yaw, indices_yaw)]
        self.B_yaw = self.B[np.ix_(indices_yaw, [3])]
 
        self.desired_poles_x = np.array([0.95, 0.945])
        self.desired_poles_y = np.array([0.95, 0.945])
        self.desired_poles_z = np.array([0.95, 0.945])
        self.desired_poles_yaw = np.array([0.9, 0.945])
       
        self.reference = np.zeros(4)

        self.start_user_input_thread()

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

    def handle_user_input(self):
        while rclpy.ok():
            try:
                user_input = input('Enter [x y z yaw]: ')
                data = [float(value) for value in user_input.split()]
                if len(data) == 4:
                    self.publish_reference(*data)
                    self.is_pid_on = True
                    self.on_off_flag.data = True
                    self.publish_control_onAndoff.publish(self.on_off_flag)
                else:
                    print("Invalid input. Please enter 4 values.")
            except ValueError:
                print("Invalid input. Please enter numeric values.")

    def publish_reference(self, x, y, z, yaw):
        self.reference_point.x = x
        self.reference_point.y = y
        self.reference_point.z = z
        self.reference_point.yaw = yaw
        self.reference_point_publisher.publish(self.reference_point)

        self.reference[0] = x
        self.reference[1] = y
        self.reference[2] = z
        self.reference[3] = yaw


    def current_state_callback(self, msg):
        self.current_state_value.position = msg.position
        self.current_state_value.speed = msg.speed


    def get_R(self, yaw):
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        return R


    def do_pid_callback(self):
        if not self.is_pid_on:
            return

        
        reference_state_x = np.array([self.reference_point.x, 0]) 
        reference_state_y = np.array([self.reference_point.y, 0]) 
        reference_state_z = np.array([self.reference_point.z, 0])
        reference_state_yaw = np.array([self.reference_point.yaw, 0])
        
        current_state_x = np.array([self.current_state_value.position.x,
                                    self.current_state_value.speed.x_speed])
        current_state_y = np.array([self.current_state_value.position.y,
                                    self.current_state_value.speed.y_speed])
        current_state_z = np.array([self.current_state_value.position.z,
                                    self.current_state_value.speed.z_speed])       
        current_state_yaw = np.array([self.current_state_value.position.yaw,
                                      self.current_state_value.speed.yaw_speed])       

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
        
        R = self.get_R(self.current_state_value.position.yaw)
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

        if u[2] > 10:
            u[2] = int(10)
        if u[2] < -10:
            u[2] = int(-10)
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

        self.pcmd_publisher.publish(self.pcmd_value)


def main(args=None):
    rclpy.init(args=args)
    node = Pid_Control()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()