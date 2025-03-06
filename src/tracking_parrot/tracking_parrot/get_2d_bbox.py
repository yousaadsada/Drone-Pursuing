import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from anafi_msg.msg import BoxData
from tracking_parrot.lib.sort import Sort

import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from cv_bridge import CvBridge
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


setup_logger()

Display = True

class DroneTracker(Node):
    def __init__(self):
        super().__init__('af_tracker')

        ##PUBLISHERS / SUBSCRIBERS
        self.frames_sub = self.create_subscription(Image,'/anafi/frames', self.frame_callback, 1)
        self.key_pub = self.create_publisher(BoxData, '/anafi/bbox2D',1)
        self.bridge = CvBridge()

        #MASK R-CNN MODEL
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ('drone_train',)
        self.cfg.DATASETS.TEST = ('drone_test',)
        self.cfg.DATALOADER.NUM_WORKERS = 2
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.40
        self.cfg.MODEL.DEVICE = "cuda"
        self.cfg.SOLVER.BASE_LR = 0.0025 
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.MODEL.WEIGHTS = os.path.join("/home/yousa/anafi_main_ros2/anafi_ros2/src/anafi_ros2/anafi_ros2/code/output/model_final.pth")
        self.predictor = DefaultPredictor(self.cfg)

        #SORT ALGORITH
        self.mot_tracker = Sort(max_age=200, min_hits=20)

        #VISUALIZATION
        self.colours = np.random.rand(200, 3)
        self.fps_time = 0.0

    def frame_callback(self, msg):
        
        data = None
        frameid = msg.header.frame_id
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        outputs = self.predictor(cv_image)
        print(outputs)
        data = self.tracker_props(outputs)
        self.keypointPublish(data, frameid)

        
        if Display:
            if data is not None:
                for bbox in data:
                    id = int(bbox[4])
                    colours = (self.colours*255).astype(int)
                    B, G, R = int(colours[id,0]), int(colours[id,1]), int(colours[id,2])
                    x1, y1, x2, y2 = map(int, [bbox[0], bbox[1], bbox[2], bbox[3]])
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (B,G,R), 2)
                    cv2.putText(cv_image, f"ID: {id}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (B, G, R), 2)
                
            cv2.imshow('Display', cv_image)
            cv2.waitKey(1) & 0xFF == ord('0')
    
    def tracker_props(self, predictions):
        
        ins_data = []
        instances = predictions["instances"]

        if len(instances.pred_masks) >= 1:

            # Extract boxes and scores
            boxes = instances.pred_boxes.tensor.cpu().tolist()
            scores = instances.scores.cpu().tolist()

            # Find the index of the highest scoring box
            max_idx = np.argmax(scores)

            # Store only the highest scoring box
            highest_score_box = boxes[max_idx]  # Get highest scoring box
            highest_score = round(scores[max_idx], 4)  # Get highest score

            # Append score to the box data
            highest_score_box.append(highest_score)
            ins_data.append(highest_score_box)

            # Convert to NumPy array
            ins_data = np.array(ins_data)

        else:
            ins_data = np.empty((0, 5))

        new_box = self.mot_tracker.update(ins_data)


        return new_box.tolist()
    
    
    def keypointPublish(self,detections, frameid):

        msg = BoxData()
        msg.frame_id = int(frameid)
        flattened_data = []
        
        if len(detections)>=1:
            msg.target = True
            flattened_data.extend([int(item) for sublist in detections for item in sublist])
        else:
            msg.target = False

        msg.data = flattened_data
        self.key_pub.publish(msg)

    def Stop(self):
        global Display
        Display = False

        cv2.destroyAllWindows()
        self.frames_sub.destroy()
        # self.csv_writer.writerow(['Time [s]',
        #                           'Detected_x [m]', 'Detected_y [m]', 'Detected_z [m]',
        #                           'Target_x [m]', 'Target_y [m]', 'Target_z [m]',
        #                           'Anafi_x [m]', 'Anafi_y [m]', 'Anafi_z [m]',
        #                           'Bebop_x [m]', 'Bebop_y [m]', 'Bebop_z [m]',
        #                           'Roll [%]', 'Pitch [%]', 'Yaw [%]', 'Gaz [%]'])

def main(args=None):
    rclpy.init(args=args)
    af_tracker = DroneTracker()

    try:
        while rclpy.ok():
            rclpy.spin_once(af_tracker)
    except: 
        af_tracker.Stop()

if __name__ == '__main__':
    main()
