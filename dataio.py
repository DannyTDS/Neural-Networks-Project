import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TrainSet(Dataset):
    # sample from the GT with 1 interval.
    def __init__(self, data_dir):
        
        self.transform = transforms.ToTensor()

        self.imgs, self.masks = [], []
        training_names = []
        
        # sample images with 1 interval, put sampled ones into training set
        for name in sorted(glob.glob(f"{data_dir}/*.png")):
            theta = float(name.split("/")[-1].split("_")[2])
            phi = float(name.split("/")[-1].split("_")[3].replace(".png", ""))

            if theta % 30 > 0 or phi % 30 > 0:
                continue
            else:
                training_names.append(name)
        
            # read in all the image files
            img = np.asarray(Image.open(name).convert('RGB')) / 255.
            self.imgs.append(self.transform(Image.fromarray(np.uint8(255 * img))))
        
        self.hw = self.imgs[0].shape[1:]

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return self.imgs[idx], idx