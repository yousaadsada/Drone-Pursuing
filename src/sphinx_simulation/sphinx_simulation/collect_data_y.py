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


class CollectDataNode(Node):
    def __init__(self):
        super().__init__('collect_data_node')
        self.connected = False
        self.running = True
        self.write_data_flag = False  # Separate flag for writing data
        self.freq = 25
        self.drone = None
        self.cumulative_time_stamp = 0.0
        self.time_stamp = 0.0
        self.control_output = Output()

        self.previous_yaw = None
        self.previous_pitch = None
        self.previous_roll = None
        self.previous_time = None

        self.position = Position()
        self.speed = Speed()
        self.collect_current_state = CollectCurrentState()

        self.sphinx = pysphinx.Sphinx(ip="127.0.0.1", port=8383)
        self.get_logger().info("Drone position publisher node has been started.")
        self.name = self.sphinx.get_default_machine_name()

        self.publisher = self.create_publisher(CollectCurrentState, '/collect_current_state', 10)

        self.data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'angle2speed')
        os.makedirs(self.data_dir, exist_ok=True)
        self.csv_file = os.path.join(self.data_dir, 'roll2y_speed.csv')

        self.write_csv_header()
        self.Connect()

        process_data_thread = threading.Thread(target=self.process)
        process_data_thread.start()

        self.save_data()  # Start the save_data timer

    def save_data(self):
        self.save_data_callback()  # Call the save_data_callback initially
        self.timer = threading.Timer(1 / self.freq, self.save_data)  # Schedule the next call to save_data
        self.timer.start()

    def save_data_callback(self):
        if self.write_data_flag:
            self.get_logger().info("Saving data...")
            formatted_time_stamp = f"{self.time_stamp:.2f}"
            data = [formatted_time_stamp,
                    self.position.x, self.position.y, self.position.z, self.position.pitch, self.position.roll, self.position.yaw,
                    self.speed.x_speed, self.speed.y_speed, self.speed.z_speed, self.speed.pitch_speed, self.speed.roll_speed, self.speed.yaw_speed, 
                    self.control_output.control_x, self.control_output.control_y, self.control_output.control_z, self.control_output.control_yaw]
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            self.time_stamp += 0.04
        else:
            self.get_logger().info("Not saving data. write_data_flag is False.")

    def write_csv_header(self):
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['elapsed_time',
                            'x', 'y', 'z', 'pitch', 'roll','yaw',
                             'x_speed', 'y_speed', 'z_speed', 'pitch_speed','roll_speed','yaw_speed',
                             'control_x', 'control_y', 'control_z', 'control_yaw'])

    def process(self):
        if self.running:
            try:
                self.drone(TakeOff())
                time.sleep(10)

                self.write_data_flag = True  # Start writing data
                for _ in range(5):
                    self.collect_and_publish_pcmd(duration=5, interval=1 / self.freq, output_file='drone_state_data_pcmd.csv',
                                                  x_input=0, y_input=20, z_input=0, yaw_input=0)
                    self.collect_and_publish_pcmd(duration=5, interval=1 / self.freq, output_file='drone_state_data_pcmd.csv',
                                                  x_input=0, y_input=-20, z_input=0, yaw_input=0)
                self.write_data_flag = False  # Stop writing data

                self.drone(Landing())

            finally:
                self.drone.disconnect()

    def collect_and_publish_pcmd(self, duration, interval, output_file, x_input, y_input, z_input, yaw_input):
        stop_event = threading.Event()

        self.control_output.control_x = int(x_input)
        self.control_output.control_y = int(y_input)
        self.control_output.control_z = int(z_input)
        self.control_output.control_yaw = int(yaw_input)

        def publish_pcmd():
            if not stop_event.is_set():
                self.send_pcmd(x_input, y_input, z_input, yaw_input)
                self.get_current_state()
                threading.Timer(interval, publish_pcmd).start()

        publish_pcmd()

        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(interval)

        stop_event.set()

    def send_pcmd(self, x_input, y_input, z_input, yaw_input):
        self.drone(PCMD(1,
                        -y_input,
                        x_input,
                        -yaw_input,
                        z_input,
                        timestampAndSeqNum=0,)).wait()

    def get_current_state(self):
        linear_speed = self.drone.get_state(SpeedChanged)
        x_speed_get = linear_speed['speedX']
        y_speed_get = linear_speed['speedY']
        z_speed_get = linear_speed['speedZ']
        self.speed.x_speed = y_speed_get
        self.speed.y_speed = x_speed_get
        self.speed.z_speed = -z_speed_get

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
        speeds = {"yaw": 0.0, "roll": 0.0, "pitch": 0.0}
        positions = {"yaw": self.position.yaw, "roll":self.position.roll, "pitch":self.position.pitch}
        previous_positions = {"yaw": self.previous_yaw, "roll":self.previous_roll, "pitch":self.previous_pitch}
        
        if self.previous_time is not None:
            for axis in speeds.keys():
                current_position = positions[axis]
                previous_position = previous_positions[axis]
                if previous_position is not None:

                    if current_position < 0 and previous_position > 0 and current_position < -0.9 * math.pi and previous_position > 0.9 * math.pi:
                        delta_position = 2 * math.pi + current_position - previous_position
                    elif current_position > 0 and previous_position < 0 and current_position > 0.9 * math.pi and previous_position < -0.9 * math.pi:
                        delta_position = -2 * math.pi + current_position - previous_position
                    else:
                        delta_position = current_position - previous_position


                    delta_time = current_time - self.previous_time
                    speeds[axis] = delta_position / delta_time

        self.speed.yaw_speed = speeds["yaw"]
        self.speed.roll_speed = speeds["roll"]
        self.speed.pitch_speed = speeds["pitch"]

        self.previous_yaw = self.position.yaw
        self.previous_roll = self.position.roll
        self.previous_pitch = self.position.pitch
        self.previous_time = current_time

        self.collect_current_state.current_state.position = self.position
        self.collect_current_state.current_state.speed = self.speed
        self.collect_current_state.elapsed_time = float(self.time_stamp)
        self.collect_current_state.output = self.control_output
        self.publisher.publish(self.collect_current_state)

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


                    # self.collect_and_publish_pcmd(duration=5, interval=1/self.freq, output_file='drone_state_data_pcmd.csv',
                    #                               x_input=0, y_input=-40, z_input=0, yaw_input=0)
                    # self.collect_and_publish_pcmd(duration=2, interval=1/self.freq, output_file='drone_state_data_pcmd.csv',
                    #                               x_input=0, y_input=15, z_input=0, yaw_input=0)
                    # self.collect_and_publish_pcmd(duration=3, interval=1/self.freq, output_file='drone_state_data_pcmd.csv',
                    #                               x_input=0, y_input=-35, z_input=0, yaw_input=0)
                    # self.collect_and_publish_pcmd(duration=4, interval=1/self.freq, output_file='drone_state_data_pcmd.csv',
                    #                               x_input=0, y_input=25, z_input=0, yaw_input=0)                
                    # self.collect_and_publish_pcmd(duration=5, interval=1/self.freq, output_file='drone_state_data_no_pcmd.csv',
                    #                               x_input=0, y_input=-10, z_input=0, yaw_input=0)