import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
from torchvision.io import read_image
from .config import cfg
from .utilities.SATMAE_transform import build_transform as SATMAE_transform
import code

def gt_mapper(gt,dataset_type):

    pass

class Dataset_landtype(Dataset):
    def __init__(self,
                 dataset_type,                              # Provide choice for dataset name
                 split="train",                             # Provide choice for split: train/val/test
                 is_train = True,                           # Flag set True if it is train dataloader
                 sat_input_size = 224)                      # Input size of satellite image
        

        self.csv_file = pd.read_csv(cfg.DataRoot,dataset_type,split+".csv")      
        self.sat_transform = SATMAE_transform(is_train=is_train, input_size=sat_input_size)
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.csv_file)
    def __getitem__(self,idx):
        sample = self.csv_file.iloc[idx]
        sat_img = read_image(os.path.join(cfg.DataRoot,self.dataset_type,sample.image_id+".jpg"))
        sat_img = np.array(torch.permute(sat_img,[1,2,0]))
        sat_img = self.sat_transform(sat_img)
        out_dict['sat']= sat_img
        out_dict['gt']= gt_mapper(sample.gt,self.dataset_type)
 
        return out_dict


if __name__ == '__main__':
    loader = torch.utils.data.DataLoader(Dataset_landtype(dataset_type,                           # Provide choice for dataset name
                 split="train",                             # Provide choice for split: train/val/test
                 is_train = True,                           # Flag set True if it is train dataloader
                 sat_input_size = 224),
                num_workers=2, batch_size=2, shuffle=True, drop_last=False,pin_memory=True))                     # Input size of satellite image
   
    batch = next(iter(loader))
    print(type(batch['sat']),type(batch['gt']))
    print(type(batch['sat'].shape),type(batch['gt'].shape))