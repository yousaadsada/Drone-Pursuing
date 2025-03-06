import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from anafi_msg.msg import PnPDataYolo, KpYolo

import numpy as np
import cv2, time, csv
from pynput.keyboard import Listener



class PnpNode(Node):
    def __init__(self):
        super().__init__('af_pnp')
        
        self.targets_sub = self.create_subscription(KpYolo, '/keypoints', self.kp_callback, 1)
        self.position_pub = self.create_publisher(PnPDataYolo, '/position',1)
        self.kp_data = []
        self.TargetPoints = np.zeros((8,2), dtype=np.float32)
        self.target_position = PnPDataYolo()

        #SOLVEPNP PARAMETERS
        # self.drone_points =  np.array([
        #     [0.19, 0.05, 0.15],
        #     [-0.19, 0.05, 0.15],
        #     [0.19, -0.05, 0.15],
        #     [-0.19, -0.05, 0.15],
        #     [0.19, 0.05, -0.15],
        #     [-0.19, 0.05, -0.15],
        #     [0.19, -0.05, -0.15],
        #     [-0.19, -0.05, -0.15]
        # ], dtype=np.float32)

        self.drone_points =  np.array([
            [0.215, 0.30, 0.30],
            [-0.215, 0.30, 0.30],
            [0.215, -0.30, 0.30],
            [-0.215, -0.30, 0.30],
            [0.215, 0.30, -0.30],
            [-0.215, 0.30, -0.30],
            [0.215, -0.30, -0.30],
            [-0.215, -0.30, -0.30]
        ], dtype=np.float32)
        
        self.camera_matrix = np.array([
            [899.288082, 0.0, 623.516603],
            [0.0, 902.894688, 375.501706],
            [0.0, 0.0, 1.0], 
        ],dtype=np.float32)




   
    def kp_callback(self, msg):

        def estimate_3d_position(keypoints_flat):

            # 2. Reshape keypoints_flat into Nx2 array for image points
            if len(keypoints_flat) != 16:
                raise ValueError("Expected 16 values (8 keypoints with x, y), got: {}".format(len(keypoints_flat)))
            image_points = np.array(keypoints_flat).reshape(-1, 2).astype(np.float32)

            # 3. Solve PnP to get rotation and translation vectors
            success, rvec, tvec = cv2.solvePnP(
                self.drone_points,       # 3D model points
                image_points,       # Corresponding 2D image points
                self.camera_matrix,      # Camera intrinsic matrix
                None,        # Distortion coefficients
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                raise RuntimeError("PnP solution failed.")

            return rvec, tvec

        try:
            target_ok = msg.target
            keypoints_flat = msg.keypoints.data 
            
            if target_ok:
                rvec, tvec = estimate_3d_position(keypoints_flat)

                tx, ty, tz = tvec.flatten()

                self.target_position.target = True
                self.target_position.tx = tx
                self.target_position.ty = ty
                self.target_position.tz = tz

                self.position_pub.publish(self.target_position)

            
            elif not target_ok:
                self.target_position.target = False
                self.target_position.tx = 0.0
                self.target_position.ty = 0.0
                self.target_position.tz = 0.0

                self.position_pub.publish(self.target_position)

        except Exception as e:
            self.get_logger().info(f"Error: {e}")
    
    def Stop(self):
        self.targets_sub.destroy()
        
        time.sleep(0.2)
        self.destroy_node()
        rclpy.shutdown()

def main():
    rclpy.init()
    af_pnp = PnpNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(af_pnp)
    except:
        af_pnp.Stop()

if __name__ == "__main__":
    main()