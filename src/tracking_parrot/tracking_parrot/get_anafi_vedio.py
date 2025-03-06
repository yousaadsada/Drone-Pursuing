import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from anafi_msg.msg import Output, Position, Speed, CurrentState
from geometry_msgs.msg import TransformStamped
import threading
import numpy as np
from geometry_msgs.msg import Point32
from anafi_msg.msg import DroneSize, KeyPoints, ImageKeyPoints
from rclpy.executors import MultiThreadedExecutor
import time
import os
import cv2, queue
import olympe
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import logging

logging.getLogger("olympe").setLevel(logging.WARNING)

class GetVedioData(Node):
    def __init__(self):
        super().__init__('get_vedio_data')
        self.running = True
        self.frameid = 0
        self.bridge = CvBridge()
        self.frame_queue = queue.LifoQueue()
        self.processing_image_thread = threading.Thread(target = self.yuv_frame_processing)
        self.cv2_cvt_color_flag = {
                    olympe.VDEF_I420: cv2.COLOR_YUV2BGR_I420,
                    olympe.VDEF_NV12: cv2.COLOR_YUV2BGR_NV12,
                }
        

        self.image_pub = self.create_publisher(Image, 'anafi/vedio', 1)   


    def yuv_frame_cb(self, yuv_frame):
        try:
            yuv_frame.ref()
            self.frame_queue.put_nowait(yuv_frame)

        except Exception as e:
            self.get_logger().info(f"Error handling media removal: {e}")


    def yuv_frame_processing(self):
        while self.running:
            try:
                yuv_frame = self.frame_queue.get(timeout=0.1)
                
                if yuv_frame is not None:
                    x = yuv_frame.as_ndarray()
                    cv2frame = cv2.cvtColor(x, self.cv2_cvt_color_flag[yuv_frame.format()])
                    self.image = cv2frame

                    msg = self.bridge.cv2_to_imgmsg(cv2frame, "bgr8")
                    msg.header.frame_id = str(self.frameid)
                    self.image_pub.publish(msg)
                    self.frameid += 1
                    yuv_frame.unref()
                
            except queue.Empty:
                pass
            except Exception as e:
                self.get_logger().info(f"Error processing frame: {e}")

    def flush_cb(self, stream):
        if stream["vdef_format"] != olympe.VDEF_I420:
            return True
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait().unref()
        return True


    def Connect(self):
        self.get_logger().info('Connecting to Anafi drone...')
        #self.DRONE_IP = os.getenv("DRONE_IP", "10.202.0.1")
        self.DRONE_IP = os.getenv("DRONE_IP", "192.168.42.1")
        self.DRONE_RTSP_PORT = os.environ.get("DRONE_RTSP_PORT")
        self.drone = olympe.Drone(self.DRONE_IP)

        for i in range(5):
            if self.running:
                connection = self.drone.connect(retry=1)
                if connection:
                    self.connected = True
                    self.get_logger().info('Connected to Anafi drone!')

                    if self.DRONE_RTSP_PORT is not None:
                        self.drone.streaming.server_addr = f"{self.DRONE_IP}:{self.DRONE_RTSP_PORT}"
                    
                    self.drone.streaming.set_callbacks(
                        raw_cb=self.yuv_frame_cb,
                        flush_raw_cb=self.flush_cb,)
                
                    self.drone.streaming.start()
                    self.processing_image_thread.start()   
                   
                    break
                
                else:
                    self.get_logger().info(f'Trying to connect (%d)' % (i + 1))
                    time.sleep(2.0)

        if not self.connected:
            self.get_logger().info("Failed to connect.")

    def Stop(self):
        self.running = False
        if self.Connected:
            self.processing_image_thread.join(timeout=1.0)
            self.drone.streaming.stop()
            self.drone.disconnect()
        
        self.get_logger().info('Shutting down...\n')
        self.Connected = False
        time.sleep(0.2)
        self.destroy_node()
        rclpy.shutdown()




def main(args = None):
    rclpy.init(args=args)
    node = GetVedioData()
    node.Connect()
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
        node.Stop()  # Custom method to stop your node's operations
        rclpy.shutdown()


if __name__ == '__main__':
    main()