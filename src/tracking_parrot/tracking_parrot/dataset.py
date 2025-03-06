import torch
import os
import cv2
import json
from torch.utils.data import Dataset

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_ids = sorted(os.listdir(image_dir))

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id)
        ann_path = os.path.join(self.annotation_dir, image_id.replace('.jpg', '_keypoints.json'))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img / 255.0).permute(2, 0, 1).float()
        
        with open(ann_path) as f:
            ann = json.load(f)
        
        keypoints = torch.tensor(ann['keypoints'], dtype=torch.float32).view(-1, 3)
        labels = torch.ones((1,), dtype=torch.int64)
        boxes = torch.tensor([[0, 0, img.shape[2], img.shape[1]]], dtype = torch.float32)

        target = {
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints.unsqueeze(0)
        }

        return img, target
    
    def __len__(self):
        return len(self.image_ids)