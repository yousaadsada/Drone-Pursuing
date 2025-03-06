import rclpy
import olympe
from rclpy.node import Node
from std_msgs.msg import String
import pysphinx
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from olympe.messages.ardrone3.PilotingState import AttitudeChanged, PositionChanged, SpeedChanged
from pynput.keyboard import Listener, Key
from geometry_msgs.msg import Vector3
import time
import os
from anafi_msg.msg import Position
from anafi_msg.msg import Speed
from anafi_msg.msg import Output
from anafi_msg.msg import CurrentState
from anafi_msg.msg import CollectCurrentState
import math
import threading
import csv
import random

class CollectDataNode(Node):
    def __init__(self):
        super().__init__('collect_data_node')
        self.running = True
        self.connected = False
        self.Connect()

        self.sphinx = pysphinx.Sphinx(ip="127.0.0.1", port=8383)
        self.get_logger().info("Drone position publisher node has been started.")
        self.name = self.sphinx.get_default_machine_name()

        self.publisher_pcmd = self.create_publisher(Output, '/pub_pcmd',1)
        self.publisher_current_state = self.create_publisher(CurrentState, '/current_state_pos',1)

        self.time_stamp = 0.0
        self.is_save_data_on = False

        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        self.previous_yaw = None
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_time = None

        self.x_input = 0
        self.y_input = 0
        self.z_input = 0
        self.yaw_input = 0

        self.save_data_dir = os.path.join(os.path.expanduser("~"),'anafi_simulation', 'data', 'simulation')
        os.makedirs(self.save_data_dir, exist_ok=True)
        self.save_data_csv_file = os.path.join(self.save_data_dir, 'collect_state_x_verify.csv')

        self.speed = Speed()
        self.position = Position()
        self.pcmd = Output()
        self.anafi_state = CurrentState()

        self.get_anafi_state_thread = threading.Thread(target=self.get_anafi_state_thread_callback)
        self.get_anafi_state_thread.daemon = True
        self.get_anafi_state_thread.start()

        self.publish_pcmd_thread = threading.Thread(target=self.publish_pcmd_thread_callback)
        self.publish_pcmd_thread.daemon = True
        self.publish_pcmd_thread.start()

        self.save_data_init()
        self.save_data_thread = threading.Thread(target=self.save_data_thread_callback)
        self.save_data_thread.daemon = True
        self.save_data_thread.start()

        self.process()


    def save_data_init(self):
        with open(self.save_data_csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['elapsed_time', 
                             'x', 'y', 'z', 'pitch','roll','yaw',
                             'x_speed', 'y_speed', 'z_speed', 'pitch_speed','roll_speed','yaw_speed',
                             'control_x', 'control_y', 'control_z', 'control_yaw'])

    def save_data_thread_callback(self):
        while self.running == True:
            start_time = time.time()
            if self.is_save_data_on == True:
                self.get_logger().info("Saving data...")
                formatted_time_stamp = f"{self.time_stamp:.2f}"
                data = [formatted_time_stamp,
                        self.anafi_state.position.x,
                        self.anafi_state.position.y,
                        self.anafi_state.position.z,
                        self.anafi_state.position.pitch,
                        self.anafi_state.position.roll,
                        self.anafi_state.position.yaw,
                        self.anafi_state.speed.x_speed,
                        self.anafi_state.speed.y_speed,
                        self.anafi_state.speed.z_speed,
                        self.anafi_state.speed.pitch_speed,
                        self.anafi_state.speed.roll_speed,
                        self.anafi_state.speed.yaw_speed,
                        self.pcmd.control_x,
                        self.pcmd.control_y,
                        self.pcmd.control_z,
                        self.pcmd.control_yaw,
                        ]
                
                with open(self.save_data_csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
                self.time_stamp += 0.04
            else:
                self.get_logger().info("Not saving data. write_data_flag is False.")
            
            elapsed_time = time.time() - start_time
            if elapsed_time < 0.04:
                time.sleep(0.04 - elapsed_time)

    def process(self):
        if self.running:
            try:
                self.drone(TakeOff())
                time.sleep(10)

                self.is_save_data_on = True  # Start writing data
                for i in range(5):
                    start_time_1 = time.time()
                    self.pcmd.control_x = int(random.uniform(-40, 40))
                    self.pcmd.control_y = 0
                    self.pcmd.control_z = 0
                    self.pcmd.control_yaw = 0
                    elapsed_time_1 = time.time() - start_time_1
                    if elapsed_time_1 < 2.0:
                        time.sleep(2.0 - elapsed_time_1) 
                    
                    start_time_2 = time.time()
                    self.pcmd.control_x = 0
                    self.pcmd.control_y = 0
                    self.pcmd.control_z = 0
                    self.pcmd.control_yaw = 0
                    elapsed_time_2 = time.time() - start_time_2
                    if elapsed_time_2 < 2.0:
                        time.sleep(2.0 - elapsed_time_2)     
    

                self.is_save_data_on = False
                self.pcmd.control_x = 0
                self.pcmd.control_y = 0
                self.pcmd.control_z = 0
                self.pcmd.control_yaw = 0
                
                self.drone(Landing())

            finally:
                self.drone.disconnect()




    def publish_pcmd_thread_callback(self):
        while self.running:
            self.drone(PCMD(1,
                            -self.pcmd.control_y,
                            self.pcmd.control_x,
                            -self.pcmd.control_yaw,
                            self.pcmd.control_z,
                            timestampAndSeqNum=0,)).wait()
            
            time.sleep(0.04)
        



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
            self.anafi_state.position.pitch = self.position.pitch
            self.anafi_state.position.roll = self.position.roll
            self.anafi_state.speed.x_speed = self.speed.x_speed * math.cos(self.position.yaw) + self.speed.y_speed * math.sin(self.position.yaw)
            self.anafi_state.speed.y_speed = self.speed.x_speed * (- math.sin(self.position.yaw)) + self.speed.y_speed * math.cos(self.position.yaw)
            self.anafi_state.speed.x_speed_world = self.speed.x_speed
            self.anafi_state.speed.y_speed_world = self.speed.y_speed
            self.anafi_state.speed.z_speed = self.speed.z_speed
            self.anafi_state.speed.yaw_speed = self.speed.yaw_speed
            self.anafi_state.speed.pitch_speed = self.speed.pitch_speed
            self.anafi_state.speed.roll_speed = self.speed.roll_speed

            self.publisher_current_state.publish(self.anafi_state)

            time.sleep(0.01)




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




def main(args=None):
    rclpy.init(args=args)
    node = CollectDataNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()