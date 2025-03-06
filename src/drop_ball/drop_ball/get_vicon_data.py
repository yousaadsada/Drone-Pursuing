import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from anafi_msg.msg import Output, Position, Speed, CurrentState
import math
from geometry_msgs.msg import TransformStamped
import transforms3d
import threading
import numpy as np
from geometry_msgs.msg import Point32
from anafi_msg.msg import DroneSize, KeyPoints, ImageKeyPoints
from rclpy.executors import MultiThreadedExecutor
import time
from geometry_msgs.msg import Vector3
import math

class GetViconData(Node):
    def __init__(self):
        super().__init__('get_vicon_data')
        self.running = True

        self.target_frame_anafi = 'anafi'
        self.target_frame_parrot = 'jackal'
        self.keypoints_name = [
        "front_top_left",
        "front_top_right",
        "front_bottom_left",
        "front_bottom_right",
        "rear_top_left",
        "rear_top_right",
        "rear_bottom_left",
        "rear_bottom_right"
        ]
        self.drone_size = DroneSize()
        self.drone_size.length = 0.43
        self.drone_size.width = 0.60
        self.drone_size.height = 0.60

        self.camera_pitch = 0.0
        
        self.keypoints_world_frame = KeyPoints()
        self.keypoints_anafi_frame = KeyPoints()
        self.keypoints_camera_frame = KeyPoints()
        self.keypoints_camera_frame_2d = KeyPoints()

        self.keypoints_parrot_frame = KeyPoints()
        self.keypoints_parrot_frame.front_top_left.x = self.drone_size.width /2 
        self.keypoints_parrot_frame.front_top_left.y = self.drone_size.length /2
        self.keypoints_parrot_frame.front_top_left.z = self.drone_size.height /2

        self.keypoints_parrot_frame.front_top_right.x = self.drone_size.width /2
        self.keypoints_parrot_frame.front_top_right.y = -self.drone_size.length /2
        self.keypoints_parrot_frame.front_top_right.z = self.drone_size.height /2

        self.keypoints_parrot_frame.front_bottom_left.x = self.drone_size.width /2
        self.keypoints_parrot_frame.front_bottom_left.y = self.drone_size.length /2
        self.keypoints_parrot_frame.front_bottom_left.z = -self.drone_size.height /2

        self.keypoints_parrot_frame.front_bottom_right.x = self.drone_size.width /2
        self.keypoints_parrot_frame.front_bottom_right.y = -self.drone_size.length /2
        self.keypoints_parrot_frame.front_bottom_right.z = -self.drone_size.height /2

        self.keypoints_parrot_frame.rear_top_left.x = -self.drone_size.width /2 
        self.keypoints_parrot_frame.rear_top_left.y = self.drone_size.length /2
        self.keypoints_parrot_frame.rear_top_left.z = self.drone_size.height /2

        self.keypoints_parrot_frame.rear_top_right.x = -self.drone_size.width /2
        self.keypoints_parrot_frame.rear_top_right.y = -self.drone_size.length /2
        self.keypoints_parrot_frame.rear_top_right.z = self.drone_size.height /2

        self.keypoints_parrot_frame.rear_bottom_left.x = -self.drone_size.width /2
        self.keypoints_parrot_frame.rear_bottom_left.y = self.drone_size.length /2
        self.keypoints_parrot_frame.rear_bottom_left.z = -self.drone_size.height /2

        self.keypoints_parrot_frame.rear_bottom_right.x = -self.drone_size.width /2
        self.keypoints_parrot_frame.rear_bottom_right.y = -self.drone_size.length /2
        self.keypoints_parrot_frame.rear_bottom_right.z = -self.drone_size.height /2
        

        self.camera_matrix = np.array([
                            [899.288082, 0.0, 623.516603],
                            [0.0, 902.894688, 375.501706],
                            [0.0, 0.0, 1.0]
                        ])

        self.drone_state_anafi = CurrentState()
        self.drone_state_parrot = CurrentState()

        self.subscribe_drone_tf = self.create_subscription(TFMessage, '/tf', self.drone_tf_callback, 10)
        self.subscribe_camera_angle = self.create_subscription(Vector3, '/gimbal/angles', self.camera_angle_callback, 10)
        
        self.keypoints_camera_frame_2d_publish = self.create_publisher(KeyPoints, '/keypoints_2d', 1)
        
        self.get_keypoints_thread = threading.Thread(target = self.get_raw_keypoints_thread_callback)
        self.get_keypoints_thread.daemon = True
        self.get_keypoints_thread.start()

    def drone_tf_callback(self, msg):
        for transform in msg.transforms:
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            print(transform.child_frame_id)
            
            if translation.x == 0.0 and translation.y == 0.0 and translation.z == 0.0 and \
                rotation.x == 0.0 and rotation.y == 0.0 and rotation.z == 0.0 and rotation.w == 0.0:
                continue
   
            
            if transform.child_frame_id == self.target_frame_anafi:
    
                self.drone_state_anafi = self.process_transform(transform)
               
            elif transform.child_frame_id == self.target_frame_parrot:
                self.drone_state_parrot = self.process_transform(transform)
 
    
    def process_transform(self, transform: TransformStamped):
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

        if roll > 0.0:
            roll = roll - math.pi
        elif roll < 0.0:
            roll = roll + math.pi
        else:
            roll = roll
        
        roll = -roll

        pitch = -pitch
        
        drone_state = CurrentState()

        drone_state.position.x = x
        drone_state.position.y = y
        drone_state.position.z = z 
        drone_state.position.yaw = yaw
        drone_state.position.pitch = pitch
        drone_state.position.roll = roll 

        return(drone_state)
    
    def camera_angle_callback(self, msg):
        self.camera_pitch = msg.y



    def get_raw_keypoints_thread_callback(self):
        
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
        
        def transform_keypoints_world(T):
            def transform_point(T, point):
                    point_homogeneous = np.array([point.x, point.y, point.z, 1.0])
                    point_world_homogeneous = np.dot(T, point_homogeneous)
                    return Point32(x=point_world_homogeneous[0], y=point_world_homogeneous[1], z=point_world_homogeneous[2])

            for i, keypoint_name in enumerate(self.keypoints_name):
                orignal_frame_point = getattr(self.keypoints_parrot_frame, keypoint_name)
                transform_frame_point = transform_point(T, orignal_frame_point)
                setattr(self.keypoints_world_frame, keypoint_name, transform_frame_point)

        def transform_keypoints_anafi(T):
            def transform_point(T, point):
                    point_homogeneous = np.array([point.x, point.y, point.z, 1.0])
                    point_world_homogeneous = np.dot(T, point_homogeneous)
                    return Point32(x=point_world_homogeneous[0], y=point_world_homogeneous[1], z=point_world_homogeneous[2])

            for i, keypoint_name in enumerate(self.keypoints_name):
                orignal_frame_point = getattr(self.keypoints_world_frame, keypoint_name)
                transform_frame_point = transform_point(T, orignal_frame_point)
                setattr(self.keypoints_anafi_frame, keypoint_name, transform_frame_point)

        def transform_keypoints_camera(T):
            def transform_point(T, point):
                    point_homogeneous = np.array([point.x, point.y, point.z, 1.0])
                    point_world_homogeneous = np.dot(T, point_homogeneous)
                    return Point32(x=point_world_homogeneous[0], y=point_world_homogeneous[1], z=point_world_homogeneous[2])

            for i, keypoint_name in enumerate(self.keypoints_name):
                orignal_frame_point = getattr(self.keypoints_anafi_frame, keypoint_name)
                transform_frame_point = transform_point(T, orignal_frame_point)
                setattr(self.keypoints_camera_frame, keypoint_name, transform_frame_point)

        while self.running:
            anafi_roll = self.drone_state_anafi.position.roll
            anafi_yaw = self.drone_state_anafi.position.yaw
            anafi_pitch = self.drone_state_anafi.position.pitch
            anafi_roll = 0.0
            anafi_pitch = 0.0

            anafi_x = self.drone_state_anafi.position.x
            anafi_y = self.drone_state_anafi.position.y
            anafi_z = self.drone_state_anafi.position.z

            parrot_roll = self.drone_state_parrot.position.roll
            parrot_yaw = self.drone_state_parrot.position.yaw
            parrot_pitch = self.drone_state_parrot.position.pitch
            parrot_x = self.drone_state_parrot.position.x
            parrot_y = self.drone_state_parrot.position.y
            parrot_z = self.drone_state_parrot.position.z

            camera_roll = 0.0
            camera_yaw = 0.0
            camera_pitch = -self.camera_pitch * math.pi / 180
            camera_x = 0.05
            camera_y = 0.0
            camera_z = 0.0



            R_parrot2world = get_rotation_matrix(parrot_roll, parrot_pitch, parrot_yaw)
            T_parrot2world = get_transformation_matrix(parrot_x, parrot_y, parrot_z, R_parrot2world)
            #from self.keypoints_parrotframe to self.keypoints_world_frame
            transform_keypoints_world(T_parrot2world)

            R_anafi2world = get_rotation_matrix(anafi_roll, anafi_pitch, anafi_yaw)
            T_ananfi2world = get_transformation_matrix(anafi_x, anafi_y, anafi_z, R_anafi2world)
            T_world2anafi = np.linalg.inv(T_ananfi2world)
            #from self.keypoints_world_frame to self.keypoints_anafi_frame
            transform_keypoints_anafi(T_world2anafi)


            R_camera2anafi = get_rotation_matrix(camera_roll, camera_pitch, camera_yaw)
            T_camera2anafi = get_transformation_matrix(camera_x, camera_y, camera_z, R_camera2anafi)
            T_anafi2camera = np.linalg.inv(T_camera2anafi)
            #from self.keypoints_world_frame to self.keypoints_anafi_frame
            transform_keypoints_camera(T_anafi2camera)

            for i, keypoint_name in enumerate(self.keypoints_name):
                keypoint3d_receive = getattr(self.keypoints_camera_frame, keypoint_name)
                keypoint_3d = np.zeros(3)
                keypoint_3d[0] = -keypoint3d_receive.y
                keypoint_3d[1] = -keypoint3d_receive.z
                keypoint_3d[2] = keypoint3d_receive.x

                keypoint_2d_homogeneous = np.dot(self.camera_matrix, keypoint_3d)
                if keypoint_2d_homogeneous[2] != 0:
                    keypoint_2d = keypoint_2d_homogeneous[:2] / keypoint_2d_homogeneous[2]
                else:
                    keypoint_2d = np.array([float('inf'), float('inf')])

                keypoint_2d_point32 = Point32(x=keypoint_2d[0], y=keypoint_2d[1], z=0.0)
                setattr(self.keypoints_camera_frame_2d, keypoint_name, keypoint_2d_point32)
        
            self.keypoints_camera_frame_2d_publish.publish(self.keypoints_camera_frame_2d)

            #time.sleep(0.01)





def main(args = None):
    rclpy.init(args=args)
    node = GetViconData()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        # Spin the executor
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down gracefully...')
    finally:
        # Cleanup
        executor.shutdown()
        # node.Stop()  # Custom method to stop your node's operations
        rclpy.shutdown()


if __name__ == '__main__':
    main()