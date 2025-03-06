import rclpy
from rclpy.node import Node
from anafi_msg.msg import KpData, PnPData

import numpy as np
import cv2, time, csv
from pynput.keyboard import Listener

filter_kp = False
filter_pnp = False
save_data = False

class ButterworthFilter():
    def __init__(self, order, a, b):
        self.n = order + 1
        self.x= np.zeros(self.n)
        self.y= np.zeros(self.n)
        self.a= np.array(a)
        self.b= np.array(b)

    def shift(self, x0, vector):
        vector[1:] = vector[:-1]
        vector[0] = x0
        return vector
    
    def filter(self, sample):
        self.x = self.shift(sample, self.x)
        self.y = self.shift(0.0, self.y)

        sum = self.b[0]*self.x[0]
        for i in range(1,self.n):
            sum += (self.b[i]*self.x[i]) - (self.a[i]*self.y[i])
        
        self.y[0] = sum
        return self.y[0]
        

class PnpNode(Node):
    def __init__(self):
        super().__init__('af_pnp')
        
        self.targets_sub = self.create_subscription(KpData, 'anafi/bbox3D/kp', self.kp_callback, 1)
        self.pnp_pub = self.create_publisher(PnPData, 'anafi/pnp',1)
        self.frame_id = 0
        self.kp_data = []
        self.TargetPoints = np.zeros((8,2), dtype=np.float32)

        #SOLVEPNP PARAMETERS
        self.DronePoints =  np.array([
            [-0.1575, 0.045, 0.125],
            [0.1575, 0.045, 0.125],
            [0.1575, -0.045, 0.125],
            [-0.1575, -0.045, 0.125],
            [-0.1575, 0.045, -0.105],
            [0.1575, 0.045, -0.105],
            [0.1575, -0.045, -0.105],
            [-0.1575, -0.045, -0.105]
        ], dtype=np.float32)
        
        self.CameraMatrix = np.array([
            [899.288082, 0.0, 623.516603],
            [0.0, 902.894688, 375.501706],
            [0.0, 0.0, 1.0], 
        ],dtype=np.float32)

        self.dist = None
        self.flags = cv2.SOLVEPNP_DLS

        #FILTERS
        # ax = [0.0675, 0.1349, 0.0675]
        # bx = [1.0, -1.1430, 0.4128]
        # ay = [0.0675, 0.1349, 0.0675]
        # by = [1.0, -1.1430, 0.4128]
        # az = [0.0976, 0.1953, 0.0976]
        # bz = [1.0, -0.9428, 0.3333]

        bx = [0.047634069676470,   0.095268139352940,   0.047634069676470] #Ts = 0.2 / Fc = 0.5 hz
        ax = [ 1.000000000000000,  -1.294816670611602,   0.485352949317482]
        by = [0.047634069676470,   0.095268139352940,   0.047634069676470]
        ay = [ 1.000000000000000,  -1.294816670611602,   0.485352949317482]
        bz = [0.003621681514929,   0.007243363029857,   0.003621681514929]
        az = [1.000000000000000,  -1.822694925196308,   0.837181651256023]

        self.a_kp = [0.061866416713925,   0.123732833427850,   0.061866416713925]
        self.b_kp = [1.0,  -1.183720042079299,   0.431185708934999]

        filter_order = 2 #order 2

        self.filter_tx = ButterworthFilter(filter_order,ax,bx)
        self.filter_ty = ButterworthFilter(filter_order,ay,by)
        self.filter_tz = ButterworthFilter(filter_order,az,bz)
        self.filter_ry = ButterworthFilter(filter_order,az,bz)

        self.kp_filter = []
        for i in range(16):
            # self.kp_filter.append(ButterworthFilter(filter_order, self.a_kp, self.b_kp))
            self.kp_filter.append(ButterworthFilter(filter_order, self.a_kp, self.b_kp))

        ##ANALYSIS OF DATA 
        if save_data:
            self.csv_file = open('PNP_DATA/pnp_data.csv', 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)

    def kp_callback(self, msg):

        try:
            init_t = self.get_clock().now().nanoseconds/1000000000

            target_ok = msg.target
            self.frame_id = msg.frame_id
            raw_keypoints = msg.kp
            kp_data = []

            pub_data = PnPData()
            kp_target_id = None
            tx, ty, tz = None, None, None
            rx, ry, rz = None, None , None

            if target_ok:
                kp_target_id = raw_keypoints[0]

                for i in range(1,17):
                    kp_data.append(raw_keypoints[i])
                
                if filter_kp:
                    for i in range(0,16):
                        i_filter = self.kp_filter[i]
                        kp_data[i]=i_filter.filter(kp_data[i])

                for i in range(0,16,2):
                    k=i//2
                    self.TargetPoints[k,0] = kp_data[i]
                    self.TargetPoints[k,1] = kp_data[i+1]

                ImagePoints = np.array(self.TargetPoints)

                success, rvec, tvec = cv2.solvePnP(
                    objectPoints=self.DronePoints,
                    imagePoints=ImagePoints,
                    cameraMatrix=self.CameraMatrix,
                    distCoeffs=self.dist
                    )
                
                if success:
                    if filter_pnp:
                        pnp_filt = []
                        pnp_filt.append(self.filter_tx.filter(float(tvec[0])))
                        pnp_filt.append(self.filter_ty.filter(float(tvec[1])))
                        pnp_filt.append(self.filter_tz.filter(float(tvec[2])))
                        pnp_filt.append(self.filter_ry.filter(float(rvec[1])))
                        
                        #Convert to degrees
                        tx = float(pnp_filt[0])
                        ty = float(pnp_filt[1])
                        tz = float(pnp_filt[2])
                        rx = float(rvec[0]*(180/np.pi))
                        ry = float(pnp_filt[3]*(180/np.pi))
                        rz = float(rvec[2]*(180/np.pi))
                    else:
                        tx = float(tvec[0])
                        ty = float(tvec[1])
                        tz = float(tvec[2])
                        rx = float(rvec[0]*(180/np.pi))
                        ry = float(rvec[1]*(180/np.pi))
                        rz = float(rvec[2]*(180/np.pi))
                else:
                    self.get_logger().info(f"Not able to extract 3D points")

                pub_data.target = True
                pub_data.time_s = init_t
                pub_data.frame_id = self.frame_id
                pub_data.target_id = kp_target_id
                pub_data.tx = tx
                pub_data.ty = ty
                pub_data.tz = tz
                pub_data.rx = rx
                pub_data.ry = ry
                pub_data.rz = rz

                self.pnp_pub.publish(pub_data)

                if save_data:
                    real_time = (self.get_clock().now().nanoseconds)/1000000000
                    self.csv_writer.writerow([real_time,
                                                tx, ty, tz,
                                                rx, ry, rz,
                                                ])
            
            elif not target_ok:
                pub_data.target = False
                pub_data.time_s = init_t
                pub_data.frame_id = 0
                pub_data.target_id = 0
                pub_data.tx = 0.0
                pub_data.ty = 0.0
                pub_data.tz = 0.0
                pub_data.rx = 0.0
                pub_data.ry = 0.0
                pub_data.rz = 0.0

                self.pnp_pub.publish(pub_data)

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