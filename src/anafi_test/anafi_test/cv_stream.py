#---------------
# IMPORTS
#----------------
import rclpy
from rclpy.node import Node

import os
import olympe
from olympe.messages.ardrone3.Piloting import TakeOff, Landing, PCMD

import time
import queue
import threading
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import warnings
warnings.simplefilter("ignore")

olympe.log.update_config({"loggers": {"olympe": {"level": "CRITICAL"}}})

#---------------
# NODE CLASS
#----------------
class FramePub(Node):
    def __init__(self):
        super().__init__('af_frame_pub')

        self.running = True
        self.frameid = 0
        self.pub_timer = 0.1  # Adjust this timer to control the publishing rate
        self.time_stamp = 0

        self.Simulation = False
        self.DRONE_IP = os.environ.get("DRONE_IP", "192.168.42.1")
        #self.DRONE_IP = os.environ.get("DRONE_IP", "10.202.0.1")
        self.DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT", "554")

        self.camera_info = CameraInfo()
        self.setup_camera_info()

        self.frame_queue = queue.LifoQueue()
        self.image_pub = self.create_publisher(Image, "/image_rect", 10)
        self.camera_matrix_pub = self.create_publisher(CameraInfo, "/camera_info", 10)
        self.bridge = CvBridge()

        self.cv2_cvt_color_flag = {
            olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
            olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
        }

        # Start the frame processing thread
        self.processing_thread = threading.Thread(target=self.yuv_frame_processing)
        self.processing_thread.start()

    def setup_camera_info(self):
        self.camera_info.width = 1280
        self.camera_info.height = 720
        self.camera_info.distortion_model = "plumb_bob"
        self.camera_info.d = [-0.13016039031434645, 0.17750641931687638, -0.001447952097018792, 0.0007412108596650789, 0.0]
        self.camera_info.k = [
            899.2880822008101, 0.0, 623.5166025666749,
            0.0, 902.8946878311265, 375.5017059319537,
            0.0, 0.0, 1.0
        ]
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_info.p = [
            895.1912309311427, 0.0, 626.1114790974117, 0.0,
            0.0, 899.7156382800594, 373.7534032986378, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

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
                    self.display_frame(cv2frame)
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
        msg = self.bridge.cv2_to_imgmsg(cv2frame, "bgr8")
        msg.header.stamp = timestamp
        msg.header.frame_id = str(self.frameid)

        # Set the CameraInfo header to the same timestamp
        self.camera_info.header.stamp = timestamp
        self.camera_info.header.frame_id = str(self.frameid)

        # Publish the synchronized messages
        self.image_pub.publish(msg)
        self.camera_matrix_pub.publish(self.camera_info)

    def display_frame(self, cv2frame):
        cv2.imshow('Drone Camera Feed', cv2frame)
        cv2.waitKey(1)

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
        self.image_pub.destroy()
        self.drone.streaming.stop()
        self.drone.disconnect()
        cv2.destroyAllWindows()

    def FlyFunc(self):
        time.sleep(0.05)
        pass

#---------------
# MAIN FUNCTION
#----------------

def main(args=None):
    rclpy.init(args=args)
    af_pub = FramePub()
    af_pub.StartFunc()

    try:
        rclpy.spin(af_pub)
    except KeyboardInterrupt:
        pass

    af_pub.StopFunc()
    af_pub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
