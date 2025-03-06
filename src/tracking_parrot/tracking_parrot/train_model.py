import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import json

class KeypointDataset(Dataset):
    def __init__(self, image_dir, keypoint_dir, transform=None):
        self.image_dir = image_dir
        self.keypoint_dir = keypoint_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        keypoint_file = image_file.replace('.jpg', '_keypoints.json')
        keypoint_path = os.path.join(self.keypoint_dir, keypoint_file)

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(keypoint_path, 'r') as f:
            keypoint_data = json.load(f)
            keypoints = np.array(keypoint_data['keypoints'], dtype=np.float32)

        if self.transform:
            image_rgb = self.transform(image_rgb)

        return image_rgb, torch.tensor(keypoints, dtype=torch.float32)
    
class LargeKeypointNet(nn.Module):
    def __init__(self):
        super(LargeKeypointNet, self).__init__()
        self.base_model = models.resnet50(pretrained = True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 16)

    def forward(self, x):
        return self.base_model(x)
    

def train_model():
    image_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'keypoints', 'image')
    keypoint_dir = os.path.join(os.path.expanduser("~"), 'anafi_simulation', 'data', 'keypoints', 'point')

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])

    dataset = KeypointDataset(image_dir, keypoint_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = LargeKeypointNet()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.01)

    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, keypoints in dataloader:
            images = images.to(device)
            keypoints = keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

    model_save_path = 'data/model_dir/keypoint_model.pth'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    train_model()