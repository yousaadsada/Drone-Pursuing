https://github.com/user-attachments/assets/75198e3d-f4f3-42eb-8cb0-6d54497a4115

# 🚁 **Drone Pursuit with PID, MPC, and YOLOv8**

The code includes both PID and MPC algorithms for guiding the drone to the reference point. Additionally, it utilizes YOLOv8 to estimate the 3D pose of the target drone, enabling precise pursuit during flight.

---

## **Requirements**
- **Operating System:** Ubuntu 22.04  
- **ROS Version:** ROS2 Humble  
- **CUDA:** 11.5 or above (for GPU acceleration)  
- **Python:** 3.8 or above  

---

## **Installation**
```bash
pip install -r requirements.txt
```

## **Train YOLO Model**
```bash
python src/tracking_parrot/tracking_parrot/train_drone_yolo_2d/train.py
python src/tracking_parrot/tracking_parrot/train_drone_yolo_3d/train.py
```

## **Run Drone Pursuit**
```bash
colcon build --packages-select tracking_parrot
source install/setup.bash
ros2 launch tracking_parrot pursuer_launch_yolo.py
```

