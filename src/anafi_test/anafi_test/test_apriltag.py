import cv2
print(cv2.__version__)

import numpy as np

# Load the ArUco dictionary
aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250)

# Verify that the dictionary is loaded correctly
if aruco_dict is not None:
    print("ArUco dictionary loaded successfully!")
else:
    print("Failed to load ArUco dictionary.")