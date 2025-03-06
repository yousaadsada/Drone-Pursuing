import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from my_custom_msgs.msg import Position
from cv_bridge import CvBridge
import cv2
import apriltag
import numpy as np
import time

import os
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD

import time
import queue
import threading
import cv2
import cv2.aruco as arcuo
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import warnings
warnings.simplefilter("ignore")

olympe.log.update_config({"loggers": {"olympe": {"level": "CRITICAL"}}})

class AprilTagPoseEstimator(Node):
    def __init__(self):
        super().__init__('apriltag_pose_estimator')
        self.pose_pub = self.create_publisher(Position, 'apriltag_pose', 10)
        self.bridge = CvBridge()
        self.pub_timer = 0.1

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.tag_size = 0.04  # Set your AprilTag size here
        self.detector = apriltag.Detector(options = apriltag.DetectorOptions(families='tag36h11',
                                                                            border=1,
                                                                            nthreads=4,
                                                                            quad_decimate=1.0,
                                                                            quad_blur=0.0,
                                                                            refine_edges=True,
                                                                            refine_decode=False,
                                                                            refine_pose=False,
                                                                            debug=False,
                                                                            quad_contours=True))

        # Camera intrinsic parameters (same as before)
        self.camera_matrix = np.array([
            [899.288082, 0.0, 623.516603],
            [0.0, 902.894688, 375.501706],
            [0.0, 0.0, 1.0]
        ])

        self.image = Image()

        self.save_dir = '/home/yousa/anafi_simulation/data/apriltag'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.dist_coeffs = np.array([-0.130160, 0.177506, -0.001448, 0.000741, 0.0])

        self.running = True
        self.frameid = 0
        self.time_stamp = 0

        self.Simulation = False
        self.DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
        #self.DRONE_IP = os.environ.get("DRONE_IP", "10.202.0.1")
        self.DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT", "554")

        self.frame_queue = queue.LifoQueue()

        self.cv2_cvt_color_flag = {
            olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
        }

        # Start the frame processing thread
        self.processing_thread = threading.Thread(target=self.yuv_frame_processing)
        self.processing_thread.start()

        self.cal_pose = self.create_timer(timer_period_sec=0.1,callback=self.image_callback)


    def yuv_frame_cb(self, yuv_frame):
        try:
            yuv_frame.ref()
            self.frame_queue.put_nowait(yuv_frame)
        except Exception as e:
            print(f"Error handling frame: {e}")

    def yuv_frame_processing(self):
        while self.running:
            start_time = time.time()

            try:
                yuv_frame = self.frame_queue.get(timeout=0.1)
                if yuv_frame is not None:
                    cv2frame = self.process_yuv_frame(yuv_frame)
                    self.publish_frame(cv2frame)
                    self.frameid += 1
                    yuv_frame.unref()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error processing frame: {e}")

            elapsed_time = time.time() - start_time
            time.sleep(max(self.pub_timer - elapsed_time, 0))

    def process_yuv_frame(self, yuv_frame):
        x = yuv_frame.as_ndarray()
        return cv2.cvtColor(x, self.cv2_cvt_color_flag[yuv_frame.format()])

    def publish_frame(self, cv2frame):
        # Get the current timestamp
        timestamp = self.get_clock().now().to_msg()

        # Create the Image message
        self.image = self.bridge.cv2_to_imgmsg(cv2frame, "bgr8")
        self.image.header.stamp = timestamp
        self.image.header.frame_id = str(self.frameid)


    def flush_cb(self, stream):
        if stream["vdef_format"] != olympe.VDEF_I420:
            return True
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait().unref()
        return True
    
    def StartFunc(self):
        print('Starting...\n')
        self.drone = olympe.Drone(self.DRONE_IP)
        self.drone.connect(retry=5)

        self.drone.streaming.set_callbacks(
            raw_cb=self.yuv_frame_cb,
            flush_raw_cb=self.flush_cb,
        )
    
        self.drone.streaming.start()
        print('Publishing frames!\n')

    def StopFunc(self):
        print('Shutting down...\n')
        self.running = False
        self.processing_thread.join()
        self.drone.streaming.stop()
        self.drone.disconnect()
        cv2.destroyAllWindows()

    def FlyFunc(self):
        time.sleep(0.05)
        pass

    def image_callback(self):
        self.get_logger().info('Image received.')
        
        # Convert the ROS Image message to a grayscale CV image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='bgr8')
            #gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f'Error converting image: {str(e)}')
            return

        # Convert the BGR image to grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        filename = os.path.join(self.save_dir, f'grey_image_{self.frameid}.jpg')
        cv2.imwrite(filename, gray_image)

        # Use the grayscale image for AprilTag detection
        detections = self.detector.detect(gray_image)
        
        if detections:
            self.get_logger().info(f'{len(detections)} AprilTag(s) detected.')
        else:
            print("No detection!")

        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray_image, self.aruco_dict, parameters=self.parameters)

        if ids is not None:
            self.get_logger().info(f'{len(ids)} ArUco marker(s) detected.')
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

            for i in range(len(ids)):
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.tag_size, self.camera_matrix, self.dist_coeffs)
                
                # Publish the pose
                position_msg = Position()
                position_msg.x = float(tvec[0][0][0])
                position_msg.y = float(tvec[0][0][1])
                position_msg.z = float(tvec[0][0][2])

                # Convert rotation vector (rvec) to roll, pitch, yaw
                roll, pitch, yaw = self.rotation_matrix_to_euler_angles(rvec[0][0])

                position_msg.roll = float(roll)
                position_msg.pitch = float(pitch)
                position_msg.yaw = float(yaw)

                self.pose_pub.publish(position_msg)
                self.get_logger().info(f'Published position for marker ID {ids[i][0]}.')

        else:
            print("No detection!")

        # Display the image (for debugging purposes)
        cv2.imshow('ArUco Detection', cv_image)
        cv2.waitKey(1)

    def rotation_matrix_to_euler_angles(self, rvec):
        """Convert a rotation vector to roll, pitch, yaw angles."""
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])

        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            yaw = 0

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagPoseEstimator()
    node.StartFunc()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.StartFunc()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
