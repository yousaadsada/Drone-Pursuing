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

> It is recommended to use a `conda` virtual environment.

```bash
conda create -n drone_env python=3.9
conda activate drone_env
pip install -r requirements.txt
pip install -e .


https://github.com/user-attachments/assets/75198e3d-f4f3-42eb-8cb0-6d54497a4115

