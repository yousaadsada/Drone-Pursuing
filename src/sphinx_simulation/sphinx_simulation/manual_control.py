import rclpy
import pysphinx
import olympe
from rclpy.node import Node
from std_msgs.msg import String

from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD
from olympe.messages.ardrone3.PilotingState import AttitudeChanged, PositionChanged, SpeedChanged
from pynput.keyboard import Listener, Key
from geometry_msgs.msg import Vector3
import time
import os
from anafi_msg.msg import Position 
from anafi_msg.msg import Speed
from anafi_msg.msg import CurrentState

class ManualControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')

        self.running = True
        self.connected = False

        self.freq_publish_pcmd = 40
        self.freq_get_position = 10

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

        self.position = Position()
        self.speedXYZ = Vector3()

        self.timer_pubPCMD = self.create_timer(callback=self.publish_pcmd, timer_period_sec=1/self.freq_publish_pcmd)
        self.timer_getPosition = self.create_timer(callback=self.get_position, timer_period_sec=1/self.freq_get_position)

        # Register key press and release events
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

        #self.publisher = self.create_publisher(Position, '/simulation_position', 10)
        self.publisher_speed = self.create_publisher(Vector3, '/simulation_speed',10)

    def on_press(self, key):
        if key == Key.left:
            self.get_logger().info("Landing command detected (left key press).")
            time.sleep(0.1)
            try:
                self.drone(Landing())
            except Exception as e:
                self.get_logger().info("Failed to Land.")
            time.sleep(0.5)

        elif key == Key.right:
            self.get_logger().info("Takeoff command detected (right key press).")
            time.sleep(0.1)
            try:
                self.drone(TakeOff())
            except Exception as e:
                self.get_logger().info("Failed to Take Off.")
            time.sleep(0.5)

        elif hasattr(key, 'char') and key.char:
            if key.char == 'a':
                self.x_manual = 50
            elif key.char == 'd':
                self.x_manual = -50
            elif key.char == 'w':
                self.y_manual = 50
            elif key.char == 's':
                self.y_manual = -50
            elif key.char == 'r':
                self.z_manual = 25
            elif key.char == 'f':
                self.z_manual = -25
            elif key.char == 'c':
                self.yaw_manual = 100
            elif key.char == 'x':
                self.yaw_manual = -100

    def on_release(self, key):
        if hasattr(key, 'char') and key.char in ['a', 'd']:
            self.x_manual = 0
        if hasattr(key, 'char') and key.char in ['w', 's']:
            self.y_manual = 0
        if hasattr(key, 'char') and key.char in ['r', 'f']:
            self.z_manual = 0
        if hasattr(key, 'char') and key.char in ['x', 'c']:
            self.yaw_manual = 0

    def publish_pcmd(self):
        self.drone(PCMD(1,
                        self.x_manual,
                        self.y_manual,
                        self.yaw_manual,
                        self.z_manual,
                        timestampAndSeqNum=0,))

    def get_position(self):
        attitude = self.drone.get_state(AttitudeChanged)
        yaw = attitude['yaw']

        linear_speed = self.drone.get_state(SpeedChanged)
        x_speed = linear_speed['speedX']
        y_speed = linear_speed['speedY']
        z_speed = linear_speed['speedZ']

        # # Get position data (latitude, longitude, altitude)
        # drone_position = self.drone.get_state(PositionChanged)
        # latitude = drone_position['latitude']
        # longitude = drone_position['longitude']
        # altitude = drone_position['altitude']

        # Convert latitude, longitude to XYZ (assuming a local NED frame for simplicity)
        # For more accurate conversion, use a geodetic library like pyproj
        # x = latitude  # Placeholder for actual conversion
        # y = longitude  # Placeholder for actual conversion
        # z = altitude

        # self.position.x = x
        # self.position.y = y
        # self.position.z = z
        self.position.yaw = yaw

        # self.publisher.publish(self.position)
        # pose = self.sphinx.get_drone_pose(machine_name="anafi")
        # tx, ty, tz, roll, pitch, yaw = pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]

        self.position.yaw = yaw

        self.speedXYZ.x = y_speed
        self.speedXYZ.y = x_speed
        self.speedXYZ.z = -z_speed

        #self.publisher.publish(self.position)
        self.publisher_speed.publish(self.speedXYZ)

    def Connect(self):
        self.get_logger().info('Connecting to Anafi drone...')
        # self.sphinx = pysphinx.Sphinx(ip="127.0.0.1", port=8383)
        # gps = self.sphinx.get_component(machine_name="anafi_ai", name="gps", type_="gps")
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

    def Stop(self):
        self.running = False
        # Ensure the drone lands before shutting everything down
        # try:
        #     if self.connected:  # Check if the drone is connected
        #         self.get_logger().info("Initiating landing sequence...")
        #         self.sphinx(Landing()).wait().success()  # Wait for the landing command to complete
        # except Exception as e:
        #     self.get_logger().info(f"Failed to land the drone: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ManualControlNode()
    node.Connect()
    rclpy.spin(node)
    
    # try:
    #     rclpy.spin(node)
    # finally:
    #     node.Stop()
    #     node.destroy_node()
        # rclpy.shutdown()

if __name__ == '__main__':
    main()