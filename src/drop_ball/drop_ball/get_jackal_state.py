
import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
import transforms3d
from anafi_msg.msg import CurrentState, Output
import math
import time
from rclpy.executors import MultiThreadedExecutor

class AnafiState(Node):
    def __init__(self):
        super().__init__('jackal_state')
        self.target_frame_jackal = 'jackal'
        self.previous_x = None
        self.previous_y = None
        self.previous_z = None
        self.previous_roll = None
        self.previous_pitch = None
        self.previous_yaw = None
        self.previous_time = None
        self.previous_yaw = None
        self.previous_time_update = True
        self.jackal_state = CurrentState()

        self.subscribe_tf_msg = self.create_subscription(TFMessage, '/tf', self.subscribe_tf_msg_callback, 10)
        self.publisher_jackal_state = self.create_publisher(CurrentState, '/jackal_state_vicon',10)


    def subscribe_tf_msg_callback(self, msg):

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

            self.jackal_state.position.x = x
            self.jackal_state.position.y = y
            self.jackal_state.position.z = z 
            self.jackal_state.position.yaw = yaw

            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.previous_time is not None:
                if current_time - self.previous_time >= 0.04:
                    speeds = {"x":0.0, "y":0.0, "z":0.0, "yaw": 0.0, "roll": 0.0, "pitch": 0.0}
                    positions = {"x":self.jackal_state.position.x, "y":self.jackal_state.position.y, "z":self.jackal_state.position.z,
                                "yaw": self.jackal_state.position.yaw, "roll":self.jackal_state.position.roll, "pitch":self.jackal_state.position.pitch}
                    previous_positions = {"x": self.previous_x, "y": self.previous_y, "z": self.previous_z,
                                        "yaw": self.previous_yaw, "roll":self.previous_roll, "pitch":self.previous_pitch}
                    
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
                    self.previous_time = current_time
            
                    self.jackal_state.speed.x_speed_world = speeds["x"]
                    self.jackal_state.speed.y_speed_world = speeds["y"]
                    self.jackal_state.speed.z_speed = speeds["z"]
                    self.jackal_state.speed.yaw_speed = speeds["yaw"]
                    # self.drone_state.speed.roll_speed = speeds["roll"]
                    # self.drone_state.speed.pitch_speed = speeds["pitch"]

                    self.previous_x = self.jackal_state.position.x
                    self.previous_y = self.jackal_state.position.y
                    self.previous_z = self.jackal_state.position.z
                    self.previous_yaw = self.jackal_state.position.yaw
                    self.previous_roll = self.jackal_state.position.roll
                    self.previous_pitch = self.jackal_state.position.pitch

            if self.previous_time_update == True:
                self.previous_time = current_time
                self.previous_time_update = False
        



        for transform in msg.transforms:
            if transform.child_frame_id == self.target_frame_jackal:
                print(transform.child_frame_id)
                process_transform_anafi(transform)
       
        self.publisher_jackal_state.publish(self.jackal_state)
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
