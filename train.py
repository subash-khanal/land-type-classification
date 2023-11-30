#This script evaluates the sat_enocder component from the provided GeoCLAP checkpoint by performing linear probing

import pytorch_lightning as pl
from .config import cfg
from .geoclap_train import GeoCLAP
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
from itertools import chain
from dataloader import Dataset_landtype
import torch
import numpy as np
import torch
import os
import random
from torchmetrics.classification import Accuracy
from .config import cfg, dataconfig 
from argparse import ArgumentParser

class GeoCLAP_landClassifer(pl.LightningModule):
    def __init__(self,hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.dataset_type = self.hparams.dataset_type
        self.num_classes = len(dataconfig[self.dataset_type+"_classes"])
        self.sat_type = self.hparams.sat_type
        if self.sat_type == "sentinel":
            self.ckpt_path = cfg.sentinel_ckpt_path
        else:
            self.ckpt_path = cfg.googleEarth_ckpt_path

        self.geoclap_model, self.orig_hparams = self.get_geoclap()
        self.valid_end_list =[]

        self.sat_encoder = self.geoclap_model.sat_encoder.eval()
        self.linear_layer = nn.Linear(self.orig_hparams['fc_dim'],self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    
    def get_geoclap(self):
        #load geoclap model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path)
        hparams = pretrained_ckpt['hyper_parameters']
        pretrained_weights = pretrained_ckpt['state_dict']
        model = GeoCLAP(hparams).to(self.device)
        model.load_state_dict(pretrained_weights)
        geoclap = model.eval()
        #set all requires grad to false
        for params in geoclap.parameters():
            params.requires_grad=False
        
        return geoclap, hparams

    def get_embeds(self, batch):
        embeds = {'sat_embeddings':None, 'sat_prediction':None}
        embeds['sat_embeddings']  = self.linear_layer(self.sat_encoder(batch['sat'].to(self.device)))
        embeds['sat_prediction'] = self.softmax(embeds['sat_embeddings'])
        return embeds
    
    def forward(self,batch):
        embeds = self.get_embeds(batch)
        return embeds

    def shared_step(self, batch):
        out_dict = {'embeds':None, 'loss':None}
        gt_targets = batch['gt'].to(self.device)
        embeds = self(batch)
        out_dict['embeds'] = embeds
        loss = self.loss_fn(embeds['sat_prediction'], gt_targets)
        out_dict['loss'] = loss
        return out_dict

    def training_step(self, batch, batch_idx):
        gt_targets = batch['gt']
        out_dict = self.shared_step(batch)
        acc = self.accuracy(out_dict['embeds']['sat_prediction'],gt_targets)
        # exec(os.environ.get('DEBUG'))
        self.log('train_loss', out_dict['loss'], sync_dist=True, batch_size=self.hparams.train_batch_size)
        self.log('train_acc', acc, sync_dist=True, batch_size=self.hparams.train_batch_size)
        return out_dict['loss']
    
    def validation_step(self, batch, batch_idx):
        gt_targets = batch['gt']
        out_dict = self.shared_step(batch)
        acc = self.accuracy(out_dict['embeds']['sat_prediction'],gt_targets)
        self.log('val_loss', out_dict['loss'], sync_dist=True, batch_size=self.hparams.val_batch_size)
        final_output = {'val_loss':out_dict['loss'],
                        'gt':gt_targets,
                        'preds':out_dict['embeds']['sat_prediction']}
        self.valid_end_list.append(final_output)
        return final_output
    
    def on_validation_epoch_end(self):
        outputs = self.valid_end_list
        preds = []
        gts = []
        for output in outputs:
            preds.append(output['preds'])
            gts.append(output['gt'])
        preds = torch.cat(preds)
        gts = torch.cat(gts)
        # exec(os.environ.get('DEBUG'))
        val_acc = self.accuracy(preds, gts)
        self.log('val_acc', val_acc)
        self.valid_end_list = []
        return val_acc
    

    def train_dataloader(self):
        
        dataset = Dataset_landtype(self.dataset_type,                  # Provide choice for dataset name from options: ["NWPU_RESISC45", "RSSCN7", "UCMerced_LandUse"]
                 split="train",                                        # Provide choice for split: train/val/test
                 is_train = True,                                      # Flag set True if it is train dataloader
                 sat_input_size = 224)                                 # Input size of satellite image
        
        loader = torch.utils.data.DataLoader(dataset,
                 num_workers=self.hparams.num_workers, batch_size=self.hparams.train_batch_size, shuffle=True, drop_last=False,pin_memory=True)      

        return loader

    def val_dataloader(self):

        dataset = Dataset_landtype(self.dataset_type,                  # Provide choice for dataset name from options: ["NWPU_RESISC45", "RSSCN7", "UCMerced_LandUse"]
                 split="val",                                          # Provide choice for split: train/val/test
                 is_train = False,                                     # Flag set True if it is train dataloader
                 sat_input_size = 224)                                 # Input size of satellite image
        
        loader = torch.utils.data.DataLoader(dataset,
                 num_workers=self.hparams.num_workers, batch_size=self.hparams.val_batch_size, shuffle=True, drop_last=False,pin_memory=True)      

        return loader
    

    def test_dataloader(self):
        dataset = Dataset_landtype(self.dataset_type,                  # Provide choice for dataset name from options: ["NWPU_RESISC45", "RSSCN7", "UCMerced_LandUse"]
                 split="test",                                         # Provide choice for split: train/val/test
                 is_train = False,                                     # Flag set True if it is train dataloader
                 sat_input_size = 224)                                 # Input size of satellite image
        
        loader = torch.utils.data.DataLoader(dataset,
                 num_workers=self.hparams.num_workers, batch_size=self.hparams.test_batch_size, shuffle=True, drop_last=False,pin_memory=True)      

        return loader

    def configure_optimizers(self):
        print(f'Initializing Learning rate {self.hparams.learning_rate}')
        params = chain(self.linear_layer.parameters())

        self.optim = torch.optim.AdamW(params=params,
                    lr=self.hparams.learning_rate,
                    weight_decay=0.0,
                    betas=(0.9,0.98),
                    eps=1e-6
                    )
        
        self.warm_up_iterations = 1000
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer = self.optim,
            T_0 = self.warm_up_iterations
        )
        return {'optimizer': self.optim,
        'lr_scheduler': {
            'name':'train/lr',
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }
        }

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


if __name__ == '__main__':
    set_seed(56)
    parser = ArgumentParser(description='')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--mode', type=str, default='dev')                          #Options: dev, train
    parser.add_argument('--sat_type', type=str, default='SoundingEarth')            #Options: [SoundingEarth, sentinel]
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--project_name', type=str, default='GeoCLAPforLC')
    parser.add_argument('--run_name', type=str, default='debug')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_true')
    parser.add_argument('--accelerator',type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--val_check_interval', type=int, default=1.0)
    parser.add_argument('--fc_dim', type=int, default = 512)
    parser.add_argument('--sat_input_size', type=int, default= 224)
    parser.add_argument('--lc_ckpt_path',type=str, default='none')
    parser.add_argument('--lc_ckpt_mode',type=str, default='soft')
    parser.add_argument('--dataset_type', type=str, default='NWPU_RESISC45')              #Options: ["NWPU_RESISC45", "RSSCN7", "UCMerced_LandUse"]
    args = parser.parse_args()

    #set learning rate logger
    print('Starting Training')
    print(args)
    #initliaze model
    lc_model = GeoCLAP_landClassifer(args)
    #initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='step')
    wb_logger = WandbLogger(save_dir=cfg.log_dir,project=args.project_name, name=args.run_name, mode=args.wandb_mode)
    ckpt_monitors = ((
            ModelCheckpoint(monitor='val_acc', mode='max', filename='{epoch}-{step}-{val_loss:.3f}', save_top_k = -1, every_n_epochs = 1,save_last=True,save_on_train_epoch_end=True)
        ))

    if args.mode == 'dev': 
        print('Development Test Run')
        trainer = pl.Trainer(precision=16,fast_dev_run=6, max_epochs=4, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=4,
        accelerator=args.accelerator, devices=args.devices, callbacks=[ckpt_monitors, lr_logger])
    elif args.mode == 'train':
        print('Training Run')
        trainer = pl.Trainer(precision=16, max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=0, 
        accelerator=args.accelerator, devices=args.devices, callbacks=[ckpt_monitors, lr_logger], 
        val_check_interval=args.val_check_interval, log_every_n_steps=25)
    else:
        raise ValueError('Invalid value for mode')
    
    if args.lc_ckpt_path.lower()=='none'.lower():
        trainer.fit(lc_model)
    else:
        if args.lc_ckpt_path.lower()=='hard':
            print('Hard Checkpoint Reload')
            trainer.fit(lc_model, ckpt_path=args.lc_ckpt_path)
        elif args.lc_ckpt_mode.lower()=='soft':
            print('Soft Checkpoint Reload')
            checkpoint = torch.load(args.lc_ckpt_path)
            lc_model.load_state_dict(checkpoint['state_dict'])
            trainer.fit(lc_model)