import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def infiniteloop(dataloader):
    while True:
        for *x, y in iter(dataloader):
            yield *x, y


class TrainSet(Dataset):
    # TODO:
    # sample from the GT with 1 interval.
    # time step <-> phi <-> angle (fix two)
    def __init__(self, args,root, size=None):
        
        transforms_list = []
        if size is not None:
            transforms_list.append(transforms.Resize(size))
        transforms_list.append(transforms.ToTensor())
        self.transform = transforms.Compose(transforms_list)

        self.imgs, self.masks = [], []
        training_names = []
        
        # sample images with 1 interval, put sampled ones into training set
        for name in sorted(glob.glob(f"{root}/*.png")):
            theta = name.split("/")[-1].split("_")[2]
            phi = name.split("/")[-1].split("_")[3].replace(".png", "")

            if theta % 30 > 0 or phi % 30 > 0:
                continue
            else:
                training_names.append(name)
        
            # read in all the image files
            img = np.asarray(Image.open(name).convert('RGB')) / 255.
            self.imgs.append(self.transform(Image.fromarray(np.uint8(255 * img))))
        
        self.hw = self.imgs[0].shape[1:]
    
    def parser(self):

if __name__ =="__main__":
