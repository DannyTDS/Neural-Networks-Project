import glob
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TrainSet(Dataset):
    def __init__(self, data_dir):
        # Feb 25: for unsupervised learning in the validation step, we sample at a larger interval

        self.imgs, self.training_names = [], []
        
        trans = transforms.Compose([transforms.ToTensor()])

        for name in sorted(glob.glob(os.path.join(data_dir, "*.png"))):
            theta = float(name.split("/")[-1].split("_")[2])
            phi = float(name.split("/")[-1].split("_")[3].replace(".png", ""))

            if theta % 60 > 0 or phi % 60 > 0:
                continue
            
            self.training_names.append(name)
        
            # img = np.asarray(Image.open(name).convert('RGB')) / 255.
            # self.imgs.append(trans(Image.fromarray(np.uint8(255 * img))))
            # shape = [C, H, W]
            self.imgs.append(trans(Image.open(name).convert("RGB")))
        
        self.hw = self.imgs[0].shape[1:]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return self.imgs[idx], idx