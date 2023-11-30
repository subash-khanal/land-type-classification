import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
from torchvision.io import read_image
from PIL import Image
from config import cfg, dataconfig 
from utilities.SATMAE_transform import build_transform as SATMAE_transform


def gt_mapper(dataset_type):
    classes = dataconfig[dataset_type+"_classes"]
    class_map = dict(map(lambda i,j : (i,j) ,classes,list(range(len(classes)))))
    print("The class map of the dataset is",class_map)
    return class_map
    

class Dataset_landtype(Dataset):
    def __init__(self,
                 dataset_type,                              # Provide choice for dataset name
                 split="train",                             # Provide choice for split: train/val/test
                 is_train = True,                           # Flag set True if it is train dataloader. Useful to determine train/ non-train augmentation scheme
                 sat_input_size = 224):                     # Input size of satellite image
        
        self.csv_file = pd.read_csv(os.path.join(dataconfig[dataset_type],dataset_type+"_"+split+".csv"))      
        self.sat_transform = SATMAE_transform(is_train=is_train, input_size=sat_input_size)
        self.dataset_type = dataset_type
        self.class_map = gt_mapper(dataset_type=self.dataset_type)

    def __len__(self):
        return len(self.csv_file)
    def __getitem__(self,idx):
        out_dict = {'sat':None,'gt':None}
        sample = self.csv_file.iloc[idx]
        image_path = os.path.join(dataconfig[self.dataset_type],sample['class'],sample['id'])
        if self.dataset_type != "UCMerced_LandUse":
            sat_img = read_image(image_path)
            sat_img = np.array(torch.permute(sat_img,[1,2,0]))
        else:
            sat_img = np.array(Image.open(image_path))
            # exec(os.environ.get('DEBUG'))
        
        sat_img = self.sat_transform(sat_img)
        out_dict['sat']= sat_img
        out_dict['gt']= torch.tensor(self.class_map[sample['class']]).long()
 
        return out_dict


if __name__ == '__main__':
    datasets = ["NWPU_RESISC45", "RSSCN7", "UCMerced_LandUse"]
   
    loader = torch.utils.data.DataLoader(Dataset_landtype(datasets[2],                            # Provide choice for dataset name from options: ["NWPU_RESISC45", "RSSCN7", "EuroSAT_RGB", "UCMerced_LandUse"]
                 split="train",                                                                   # Provide choice for split: train/val/test
                 is_train = True,                                                                 # Flag set True if it is train dataloader
                 sat_input_size = 224),                                                           # Input size of satellite image
                 num_workers=2, batch_size=8, shuffle=True, drop_last=False,pin_memory=True)      
   
    batch = next(iter(loader))
    print(type(batch['sat']),type(batch['gt']))
    print(batch['sat'].shape,batch['gt'].shape)