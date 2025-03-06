import torch

device = torch.device('cuda')
if device:
    print("GPU")
else:
    print("CPU")
print(torch.__version__)
print(torch.cuda.is_available())



try:
    model = torch.load('/home/yousa/anafi_main_ros2/output/model_final.pth')
    print("model_final.pth loaded successfully!")
except Exception as e:
    print(f"Failed to load model_final.pth: {e}")

try:
    model = torch.load('/home/yousa/anafi_simulation/data/training_model/frame_kp_vertices.pt')
    print("frame_kp_vertices.pt loaded successfully!")
except Exception as e:
    print(f"Failed to load frame_kp_vertices.pt: {e}")