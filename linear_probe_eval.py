import numpy as np
import torch
import random
import os
from argparse import ArgumentParser
from .train import GeoCLAP_landClassifer
from .dataloader import gt_mapper
import torch.nn as nn
from torchmetrics.classification import Accuracy
from .config import cfg, dataconfig 
from dataloader import Dataset_landtype
from tqdm import tqdm
import pandas as pd

results_path = "/home/k.subash/land-type-classification/logs"

def set_seed(seed: int = 56) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class Evaluate(object):
    def __init__(self, dataset_type, ckpt_path,device):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.device = device
        self.dataset_type = dataset_type
        self.class_map = gt_mapper(dataset_type)
        self.inv_class_map = {v: k for k, v in self.class_map.items()}
        self.lc_model, self.hparams = self.get_geoclap()
        self.softmax = nn.Softmax(dim=1)
        self.num_classes = len(dataconfig[self.dataset_type+"_classes"])
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
    
    def get_geoclap(self):
        #load geoclap model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path)
        hparams = pretrained_ckpt['hyper_parameters']
        pretrained_weights = pretrained_ckpt['state_dict']
        model = GeoCLAP_landClassifer(hparams).to(self.device)
        model.load_state_dict(pretrained_weights)
        lc_model = model.eval()
        #set all requires grad to false
        for params in lc_model.parameters():
            params.requires_grad=False
        return lc_model, hparams 

    def get_embeds(self, batch):
        embeds = {'sat_embeddings':None, 'sat_prediction':None}
        embeds['sat_embeddings']  = self.lc_model.linear_layer(self.lc_model.sat_encoder(batch['sat'].to(self.device)))
        embeds['sat_prediction'] = self.softmax(embeds['sat_embeddings'])
        return embeds
    
    def test_dataloader(self):
        dataset = Dataset_landtype(self.dataset_type,                  # Provide choice for dataset name from options: ["NWPU_RESISC45", "RSSCN7", "UCMerced_LandUse"]
                 split="test",                                         # Provide choice for split: train/val/test
                 is_train = False,                                     # Flag set True if it is train dataloader
                 sat_input_size = 224)                                 # Input size of satellite image
        
        loader = torch.utils.data.DataLoader(dataset,
                 num_workers=self.hparams['num_workers'], batch_size=self.hparams['test_batch_size'], shuffle=False, drop_last=False,pin_memory=True)      

        return loader

    
    @torch.no_grad()
    def get_test_acc(self):
        
        test_dataloader = self.test_dataloader()
        preds = []
        gts = []
        for i,batch in tqdm(enumerate(test_dataloader)):
            print("batch no:",str(i))
            embeds = self.get_embeds(batch=batch)
            gts.append(batch['gt'])
            preds.append(embeds['sat_prediction'])
        preds = torch.cat(preds)
        gts = torch.cat(gts)
        test_acc = self.accuracy(preds.to(self.device), gts.to(self.device))
        df = pd.DataFrame(columns=['ground_truth','prediction'])
        preds = [self.inv_class_map[torch.argmax(preds[i]).detach().cpu().item()] for i in range(len(preds))]
        gts = [self.inv_class_map[gts[i].detach().cpu().item()] for i in range(len(gts))]
        df['ground_truth'] = gts
        df['prediction'] = preds
        df.to_csv(os.path.join(results_path,self.dataset_type+"_using_"+self.hparams['sat_type']+"_trained_ckpt.csv"))
        return test_acc

if __name__ == '__main__':
    set_seed(56)
    parser = ArgumentParser(description='')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--sat_type', type=str, default='SoundingEarth')                 #Options: [SoundingEarth, sentinel]
    parser.add_argument('--lc_ckpt_path',type=str, default='none')
    parser.add_argument('--dataset_type', type=str, default='NWPU_RESISC45')              #Options: ["NWPU_RESISC45", "RSSCN7", "UCMerced_LandUse"]
    args = parser.parse_args()

    
    
    eval_obj = Evaluate(dataset_type=args.dataset_type, 
                        ckpt_path=args.lc_ckpt_path,
                        device=device)
    
    print(eval_obj.get_test_acc())