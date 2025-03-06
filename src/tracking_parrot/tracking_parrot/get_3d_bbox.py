import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from anafi_msg.msg import KpData, BoxData

import cv2
import torch
import time
import numpy as np
import tkinter as tk

from collections import deque
from torchvision.transforms import functional as F
from cv_bridge import CvBridge
from tkinter import ttk

Display = True

#NODE CLASS
class BBox(Node):
    def __init__(self):
        super().__init__('af_3D_bbox')
        self.bridge = CvBridge()

        #KP RCNN MODEL
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = torch.load('/home/yousa/anafi_main_ros2/anafi_ros2/src/anafi_ros2/anafi_ros2/Data/frame_kp_vertices.pt')
        self.model.to(self.device)
        #self.get_logger().info(f'Current device: {self.device}')

        #PUBLISHERS / SUBSCRIBERS
        self.frame_sub = self.create_subscription(Image,'anafi/frames', self.frame_callback, 1)
        self.targets_sub = self.create_subscription(BoxData, 'anafi/bbox2D', self.bbox_callback, 1)
        self.kp_bbox_pub = self.create_publisher(KpData, 'anafi/bbox3D/kp',2)

        #PROCESSING VARS
        self.frames_queue = deque(maxlen=5)
        self.frame_roi = None
        self.target_frame = None
        self.keypoints = []
        self.connections = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        self.colors = [(255, 0, 0),(0, 255, 0),(0, 0, 255),(0, 255, 255),(255, 0, 255),(255, 255, 0),(255, 255, 255),(0, 0, 0)]
        self.bbox_data = [0]
        self.targets_frame_id = 0
        self.id_values = [0]
        self.id_list = [0]

        #DISPLAY VARS
        if Display:
            self.name = "3d bbox Display"
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        
        self.Score_thereshold = 0.1
        self.resize = 100

        #POPUP WINDOW
        self.popup = tk.Tk()
        self.popup.title("Select Target")
        self.popup.geometry("300x200")

        self.selected_value = tk.IntVar(self.popup)
        self.selected_value.set('Select target')

        self.target_selection = 0
        self.dropdown_menu = ttk.OptionMenu(self.popup, self.selected_value, *self.id_values)
        self.dropdown_menu.pack()
        
        confirm_button = ttk.Button(self.popup, text="Confirm", command=self.on_confirm)
        confirm_button.pack()

    
    #TARGET CONFIRM
    def on_confirm(self):
        try:
            self.target_selection = self.selected_value.get()
        except Exception:
            self.target_selection = 0
            #self.get_logger().info('No target selected.')

    #FRAME SUBSCRIBER
    def frame_callback(self, msg):
        self.frame = msg.data
        self.frames_queue.append(msg)
    
    #BOUNDING BOX SUBSCRIBER
    def bbox_callback(self, msg): 

        targets_bbox = []
        current_group = []
        obj = []
        self.keypoints = []

        msg_pub = KpData()

        target_ok = msg.target
        data = msg.data

        #Data processing
        if target_ok:


            for n in range(len(data)):
                current_group.append(data[n])
                if n % 5 == 4:
                    targets_bbox.append(current_group)
                    current_group = []

        #Kp processing
        for frame in self.frames_queue:

            if (len(targets_bbox)!=0):
                self.target_frame = self.bridge.imgmsg_to_cv2(frame, "bgr8")
                self.id_list = []

                for target in targets_bbox:
                    x1, y1, x2, y2 = target[0:4]
                    
                    x1_resize = max(x1-self.resize,0)
                    y1_resize = max(y1-self.resize,0)
                    x2_resize = min(x2+self.resize, 1280)
                    y2_resize = min(y2+self.resize, 720)
                    self.frame_roi = self.target_frame[y1_resize:y2_resize, x1_resize:x2_resize]
                    
                    target_kp = self.KpRCNN(self.frame_roi)
                    target_drone_id = int(target[4])

                    if target_kp.size==16:
                        obj.append(target_drone_id)
                        self.id_list.append(target_drone_id)

                        for kp in target_kp:
                            kp[0] += int(x1_resize)
                            obj.append(int(kp[0]))
                            kp[1] += int(y1_resize)
                            obj.append(int(kp[1]))
                        self.keypoints.append(target_kp)

                    msg_pub.kp = obj
                    obj = []
                break

            else:
                self.target_frame = None

        #Update Window
        menu = self.dropdown_menu.children['!menu']
        if self.id_values != self.id_list:
            menu.delete(0,'end')
            self.id_values = self.id_list
            menu.add_command(label='No target', command=lambda val='No target': self.selected_value.set(val))
            for value in self.id_values:
                    menu.add_command(label=str(value), command=lambda val=value: self.selected_value.set(val))


        #Publish 3D Kp
        if len(msg_pub.kp)!=0:
            msg_pub.target = True
            self.kp_bbox_pub.publish(msg_pub)
        else:
            msg_pub.target = False
            msg_pub.frame_id = 0
            msg_pub.kp = []
            self.kp_bbox_pub.publish(msg_pub)


        #Display
        try:
            if Display:
                if target_ok and len(self.keypoints)!=0:
                    self.show3dbbox()
                elif not target_ok and len(self.frames_queue)==5:
                    self.showframe()
        except Exception as e:
            self.get_logger().info(f"{e}")


    #DISPLAY BBOX FUNCTION
    def show3dbbox(self):
        
        for j in range(0,len(self.keypoints)):
            target = self.keypoints[j]
            for i in range(len(self.connections)):
                start_point = (int(target[self.connections[i][0], 0]), int(target[self.connections[i][0], 1]))
                end_point = (int(target[self.connections[i][1], 0]), int(target[self.connections[i][1], 1]))
                if self.id_values[j] == self.target_selection:
                    cv2.line(self.target_frame, start_point, end_point, (0,255,0), 2)
                    cv2.putText(self.target_frame, f"ID: {self.id_values[j]}", (target[7,0], target[7,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                else:
                    cv2.line(self.target_frame, start_point, end_point, (0,0,255), 2)
                    cv2.putText(self.target_frame, f"ID: {self.id_values[j]}", (target[7,0], target[7,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
       
            for i in range(len(target)):
                point = (target[i,0],target[i,1])
                cv2.circle(self.target_frame, point, radius=5, color=self.colors[i], thickness=-1)
            
        if self.target_frame is not None:
            cv2.imshow(self.name, self.target_frame)
            cv2.waitKey(1)

    def showframe(self):
        display_frame = self.bridge.imgmsg_to_cv2(self.frames_queue[-1], "bgr8")
        cv2.imshow(self.name, display_frame)
        cv2.waitKey(1)


    #KP EXTRACTION
    def KpRCNN(self, target):

        img = F.to_tensor(target).to(self.device)
        self.model.eval()
        output = self.model([img])
        
        kp_better = np.empty([])

        scores = output[0]['scores'].detach().cpu().numpy()
        filtered_scores = np.where(scores > self.Score_thereshold)[0]
        if len(filtered_scores)>0:
            max_score = filtered_scores[np.argmax(scores[filtered_scores])]
            kp = output[0]['keypoints'][max_score].detach().cpu().numpy().astype(np.int32)
            kp = kp[:,0:2]
            kp_better = self.kp_enhancement(kp).reshape(8,2)
            #self.get_logger().info(f"Max Score: {scores[max_score]}")

        return kp_better

    #KP ENHANCEMENT FUNCTION
    def kp_enhancement(self, kp):
        vertices = kp
        vertices = vertices.reshape(8, 2)
 
        u0 = vertices[0, 0]
        v0 = vertices[0, 1]
        u1 = vertices[1, 0]
        v3 = vertices[3, 1]
        u4 = vertices[4, 0]
        v4 = vertices[4, 1]
        u5 = vertices[5, 0]
        v7 = vertices[7, 1]
        h1 = v3-v0
        w1 = u1-u0
        h2 = v7-v4
        w2 = u5-u4

        vertices[1, :] = [u0+w1, v0]
        vertices[2, :] = [u0+w1, v0+h1]
        vertices[3, :] = [u0, v0+h1]
        vertices[5, :] = [u4+w2, v4]
        vertices[6, :] = [u4+w2, v4+h2]
        vertices[7, :] = [u4, v4+h2]

        return vertices.reshape(-1)
    
    #STOP FUNCTION
    def Stop(self):
        cv2.destroyAllWindows()
        self.targets_sub.destroy()
        self.frame_sub.destroy()
        
        time.sleep(0.2)
        self.destroy_node()
        rclpy.shutdown()


#MAIN BOOL
def main():
    rclpy.init()
    af_3D_bbox = BBox()

    try:
        while rclpy.ok():
            rclpy.spin_once(af_3D_bbox)
            af_3D_bbox.popup.update()
    except:
        af_3D_bbox.Stop()

if __name__ == "__main__":
    main()