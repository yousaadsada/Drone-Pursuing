The code includes both PID and MPC algorithms for guiding the drone to the reference point. 
Additionally, it utilizes YOLOv8 to estimate the 3D pose of the target drone, enabling precise pursuit during flight.

Requirements
Operating System: Ubuntu 22.04
ROS Version: ROS2 Humble
CUDA: 11.5 or above (for GPU acceleration)
Python: 3.8 or above

Dependencies Installation

1.Install ROS2 Humble

Official Guide: https://docs.ros.org/en/humble/Installation.html

2.Install Olympe SDK (for Parrot Anafi)

Using pip:

pip install olympe

3️⃣ Install CUDA (Recommended: 11.5 or above)
Official Guide: https://developer.nvidia.com/cuda-toolkit

4️⃣ Install PyTorch (with CUDA support)
Using pip:
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

5️⃣ Install CasADi (for MPC)
Using pip:
pip install casadi

6️⃣ Install Additional Python Libraries
Using pip:
pip install numpy opencv-python scipy matplotlib yolov8

🚀 How to Run the Program
1. Build the workspace
colcon build --packages-select tracking_parrot
source install/setup.bash

2. Train the 2D BBox model
Run the following command from the root directory of the repository:
python src/tracking_parrot/tracking_parrot/train_drone_yolo_2d/train.py

3. Train the 3D BBox model
Run the following command from the root directory of the repository:
python src/tracking_parrot/tracking_parrot/train_drone_yolo_3d/train.py

4.Run the Launch File to Start the Drone Pursuiting Process
ros2 launch tracking_parrot pursuer_launch_yolo.py

https://github.com/user-attachments/assets/75198e3d-f4f3-42eb-8cb0-6d54497a4115

