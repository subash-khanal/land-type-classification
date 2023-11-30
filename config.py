from yacs.config import CfgNode as CN
import os
cfg = CN()

cfg.DataRoot = '/storage1/fs1/jacobsn/Active/user_k.subash/data/'
cfg.pretrained_models_path = '/storage1/fs1/jacobsn/Active/user_k.subash/checkpoints/'
cfg.satmae_pretrained_ckpt = '/storage1/fs1/jacobsn/Active/user_k.subash/checkpoints/SATMAE/finetune-vit-base-e7.pth'
cfg.sentinel_ckpt_path = '/storage1/fs1/jacobsn/Active/user_k.subash/projects/geoclap/logs/best_ckpts/geoclap_sentinel_best_model.ckpt'
cfg.googleEarth_ckpt_path = '/storage1/fs1/jacobsn/Active/user_k.subash/projects/geoclap/logs/best_ckpts/geoclap_soundingEarth_best_model.ckpt'
cfg.log_dir = '/storage1/fs1/jacobsn/Active/user_k.subash/projects/land-type-classification/logs'

#datasets

NWPU_RESISC45 = "/storage1/fs1/jacobsn/Active/user_k.subash/data/NWPU_RESISC45"
NWPU_RESISC45_classes = ["Airfield", "Anchorage", "Beach", "Dense-Residential", "Farm",  "Flyover",  "Forest",  "Game-Space",  "Parking-Space" , "River" , "Sparse-Residential",  "Storage-Cisterns"] #images in "*.jpg" format.

RSSCN7 = "/storage1/fs1/jacobsn/Active/user_k.subash/data/RSSCN7"
RSSCN7_classes = ["aGrass",  "bField",  "cIndustry",  "dRiverLake",  "eForest",  "fResident",  "gParking"]#images in "*.jpg" format.

UCMerced_LandUse = "/storage1/fs1/jacobsn/Active/user_k.subash/data/UCMerced_LandUse/Images"
UCMerced_LandUse_classes = ['buildings', 'tenniscourt', 'harbor', 'airplane', 'denseresidential', 'intersection', 'river', 'chaparral', 'beach', 'forest', 'agricultural', 'mobilehomepark', 'baseballdiamond', 'parkinglot', 'golfcourse', 'storagetanks', 'mediumresidential', 'freeway', 'sparseresidential', 'runway', 'overpass'] #images in "*.tif" format.


dataconfig = {'NWPU_RESISC45': NWPU_RESISC45,
              'NWPU_RESISC45_classes': NWPU_RESISC45_classes,
              'RSSCN7': RSSCN7,
              'RSSCN7_classes': RSSCN7_classes,
              'UCMerced_LandUse': UCMerced_LandUse,
              'UCMerced_LandUse_classes': UCMerced_LandUse_classes
              }