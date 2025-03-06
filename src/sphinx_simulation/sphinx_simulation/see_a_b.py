
import rclpy
from rclpy.node import Node
import numpy as np
import os

class SeeAandB(Node):
    def __init__(self):
        super().__init__('see_a_b')
        self.watch()

    def watch(self):
        # Define the data directory
        data_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'npy')

        # Load the matrices from the .npy files
        A_x = np.load(os.path.join(data_dir, 'A_x.npy'))
        B_x = np.load(os.path.join(data_dir, 'B_x.npy'))

        A_y = np.load(os.path.join(data_dir, 'A_y.npy'))
        B_y = np.load(os.path.join(data_dir, 'B_y.npy'))

        A_z = np.load(os.path.join(data_dir, 'A_z.npy'))
        B_z = np.load(os.path.join(data_dir, 'B_z.npy'))

        A_yaw = np.load(os.path.join(data_dir, 'A_yaw.npy'))
        B_yaw = np.load(os.path.join(data_dir, 'B_yaw.npy'))

        # Print the matrices
        print("A_x matrix:")
        print(A_x)
        print("B_x matrix:")
        print(B_x)

        print("A_y matrix:")
        print(A_y)
        print("B_y matrix:")
        print(B_y)

        print("A_z matrix:")
        print(A_z)
        print("B_z matrix:")
        print(B_z)

        print("A_yaw matrix:")
        print(A_yaw)
        print("B_yaw matrix:")
        print(B_yaw)


def main(args = None):
    rclpy.init(args=args)
    node = SeeAandB()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()