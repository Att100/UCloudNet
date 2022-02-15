import paddle
from paddle.io import Dataset
from PIL import Image
import pandas as pd
import random
import os
import numpy as np

RANDOM_SEED = 999

def read_metadata(path, type='all', split='train') -> list:
    random.seed(RANDOM_SEED)
    data = pd.read_csv(os.path.join(path, 'metadata.csv'))
    names = data['Name'].to_list()
    augmented = []
    for name in names:
        if type != 'all':
            if type == 'd':
                if name[0] != 'd':
                    continue
            elif type == 'n':
                if name[0] != 'n':
                    continue
        augmented.append(name)
        for i in range(1, 5+1):
            augmented.append(name.split(".")[0]+"_"+str(i)+".jpg")
    random.shuffle(augmented)
    length = len(augmented)
    if split == 'train':
        return augmented[:int(length*0.8)]
    else:
        return augmented[int(length*0.8):]

class SWINySEG(Dataset):
    def __init__(self, path="./dataset/SWINySEG", type='all', split='train'):
        super().__init__()

        self.path = path
        self.split = split
        self.names = read_metadata(path, type, split)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, 'images', self.names[idx]))
        gt = Image.open(os.path.join(
            self.path, 'GTmaps', self.names[idx].split(".")[0]+".png"))
        img = img.resize((304, 304))
        gt = gt.resize((304, 304))
        # to numpy array and normalize
        img_arr = np.array(img).transpose(2, 0, 1) / 255
        gt_arr = np.array(gt) / 255
        img_tensor = paddle.to_tensor(img_arr).astype('float32')
        gt_tensor = paddle.to_tensor(gt_arr).astype('float32')
        return img_tensor, gt_tensor

    def __len__(self):
        return len(self.names)