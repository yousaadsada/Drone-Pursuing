
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import transforms3d
from anafi_msg.msg import CurrentState, Output
import math
import time
from rclpy.executors import MultiThreadedExecutor
from anafi_msg.msg import PnPData, PnPDataYolo


class AnafiState(Node):
    def __init__(self):
        super().__init__('anafi_state')
        self.target_frame_anafi = 'anafi'
        self.target_frame_jackal = 'jackal'

        self.previous_x_anafi = None
        self.previous_y_anafi = None
        self.previous_z_anafi = None
        self.previous_roll_anafi = None
        self.previous_pitch_anafi = None
        self.previous_yaw_anafi = None
        self.previous_time_anafi = None
        self.previous_yaw_anafi = None
        self.previous_time_update_anafi = True

        self.previous_x_jackal = None
        self.previous_y_jackal = None
        self.previous_z_jackal = None
        self.previous_roll_jackal = None
        self.previous_pitch_jackal = None
        self.previous_yaw_jackal = None
        self.previous_time_jackal = None
        self.previous_yaw_jackal = None
        self.previous_time_update_jackal = True

        self.anafi_state = CurrentState()
        self.jackal_state = CurrentState()
        self.refernece_state = CurrentState()

        self.subscribe_drone_state = self.create_subscription(TFMessage, '/tf', self.subscribe_drone_state_callback, 10)
        self.refernece_state_pub = self.create_publisher(CurrentState,'/position', 10)

    def subscribe_drone_state_callback(self, msg):

        def process_transform_anafi(transform: TransformStamped):
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

            self.anafi_state.position.x = x
            self.anafi_state.position.y = y
            self.anafi_state.position.z = z 
            self.anafi_state.position.yaw = yaw

            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.previous_time_anafi is not None:
                if current_time - self.previous_time_anafi >= 0.04:
                    speeds = {"x":0.0, "y":0.0, "z":0.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0}
                    positions = {"x":self.anafi_state.position.x, "y":self.anafi_state.position.y, "z":self.anafi_state.position.z,
                                "yaw": self.anafi_state.position.yaw, "roll":self.anafi_state.position.roll, "pitch":self.anafi_state.position.pitch}
                    previous_positions = {"x": self.previous_x_anafi, "y": self.previous_y_anafi, "z": self.previous_z_anafi,
                                        "yaw": self.previous_yaw_anafi, "roll":self.previous_roll_anafi, "pitch":self.previous_pitch_anafi}
                    
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
        
                            delta_time = current_time - self.previous_time_anafi
                            speeds[axis] = delta_position / delta_time
                    self.previous_time_anafi = current_time
            
                    self.anafi_state.speed.x_speed_world = speeds["x"]
                    self.anafi_state.speed.y_speed_world = speeds["y"]
                    self.anafi_state.speed.z_speed = speeds["z"]
                    self.anafi_state.speed.yaw_speed = speeds["yaw"]
                    # self.drone_state.speed.roll_speed = speeds["roll"]
                    # self.drone_state.speed.pitch_speed = speeds["pitch"]

                    self.previous_x_anafi = self.anafi_state.position.x
                    self.previous_y_anafi = self.anafi_state.position.y
                    self.previous_z_anafi = self.anafi_state.position.z
                    self.previous_yaw_anafi = self.anafi_state.position.yaw
                    self.previous_roll_anafi = self.anafi_state.position.roll
                    self.previous_pitch_anafi = self.anafi_state.position.pitch

            if self.previous_time_update_anafi == True:
                self.previous_time_anafi = current_time
                self.previous_time_update_anafi = False
        
        def process_transform_jackal(transform: TransformStamped):
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

            self.jackal_state.position.x = x
            self.jackal_state.position.y = y
            self.jackal_state.position.z = z 
            self.jackal_state.position.yaw = yaw

            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.previous_time_jackal is not None:
                if current_time - self.previous_time_jackal >= 0.04:
                    speeds = {"x":0.0, "y":0.0, "z":0.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0}
                    positions = {"x":self.jackal_state.position.x, "y":self.jackal_state.position.y, "z":self.jackal_state.position.z,
                                "yaw": self.jackal_state.position.yaw, "roll":self.jackal_state.position.roll, "pitch":self.jackal_state.position.pitch}
                    previous_positions = {"x": self.previous_x_jackal, "y": self.previous_y_jackal, "z": self.previous_z_jackal,
                                        "yaw": self.previous_yaw_jackal, "roll":self.previous_roll_jackal, "pitch":self.previous_pitch_jackal}
                    
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
        
                            delta_time = current_time - self.previous_time_jackal
                            speeds[axis] = delta_position / delta_time
                    self.previous_time_jackal = current_time
            
                    self.jackal_state.speed.x_speed_world = speeds["x"]
                    self.jackal_state.speed.y_speed_world = speeds["y"]
                    self.jackal_state.speed.z_speed = speeds["z"]
                    self.jackal_state.speed.yaw_speed = speeds["yaw"]
                    # self.drone_state.speed.roll_speed = speeds["roll"]
                    # self.drone_state.speed.pitch_speed = speeds["pitch"]

                    self.previous_x_jackal = self.anafi_state.position.x
                    self.previous_y_jackal = self.anafi_state.position.y
                    self.previous_z_jackal = self.anafi_state.position.z
                    self.previous_yaw_jackal = self.anafi_state.position.yaw
                    self.previous_roll_jackal = self.anafi_state.position.roll
                    self.previous_pitch_jackal = self.anafi_state.position.pitch

            if self.previous_time_update_jackal == True:
                self.previous_time_jackal = current_time
                self.previous_time_update_jackal = False

        for transform in msg.transforms:
            if transform.child_frame_id == self.target_frame_anafi:
                print(transform.child_frame_id)
                process_transform_anafi(transform)
            elif transform.child_frame_id == self.target_frame_parrot:
                print(transform.child_frame_id)
                process_transform_jackal(transform)


        self.refernece_state.position.x = self.jackal_state.position.x - self.anafi_state.position.x
        self.refernece_state.position.y = self.jackal_state.position.y - self.anafi_state.position.y
        self.refernece_state.position.z = 2.0
        self.refernece_state.position.yaw = 0.0

        self.refernece_state_pub.publish(self.refernece_state)

        #time.sleep(0.04)


def run_control_loop(node):
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"Error in control loop: {e}")
    finally:
        print("Disconnected from Anafi drone.")



def main(args=None):
    rclpy.init(args=args)
    node = AnafiState()

    # Use a MultiThreadedExecutor to handle callbacks in parallel threads
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()  # Spin the node with multithreading
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
