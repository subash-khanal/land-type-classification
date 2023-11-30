#This script evaluates the sat_enocder component from the provided GeoCLAP checkpoint by performing zero-shot classification on land-cover types

import pytorch_lightning as pl
from ..config import cfg
from .geoclap_train import GeoCLAP, l2normalize
import torch.nn as nn
from .utilities.landtype_text_prompts import get_class_prompts
from .utilities import clap_data_processor
from .lc_dataloader import lc_dataset
import torch

class ZS_Evaluation(pl.LightningModule):
    def __init__(self,hparams):
        super().__init__()
        self.ckpt_path = self.hparams.geoclap_ckpt_path
        self.dataset_type = self.hparams.dataset_type
        self.sat_type = self.hparams.sat_type
        self.geoclap_model, self.orig_hparams = self.get_geoclap()

        self.sat_encoder = self.geoclap_model.sat_encoder.eval()
        self.text_encoder = self.geoclap_model.text_encoder.eval()
        self.class_prompts = self.get_class_prompts(data_type=self.hparams.dataset_type)
        self.text_embeds = self.get_text_embeddings()

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
        
        return geoclap

    def get_text_embeddings(self):
        text_prompts = self.class_prompts
        processed_text = clap_data_processor.get_text_clap(text_prompts)
        text_input = {}
        for key in processed_text.keys():
            text_input[key] = processed_text[key].to(self.device)
        text_embeds = self.text_encoder(text_input)
        return text_embeds

    def get_embeds(self, batch):
        embeds = {'sat_embeddings':None, 'text_embeddings':None}
        embeds['sat_embeddings']  = l2normalize(self.sat_encoder(batch['sat'].to(self.device)))
        embeds['text_embeddings'] = l2normalize(self.text_embeds)
    
        return embeds
    
    def get_dataloader(self):

        pass
    
    def compute_lc_acc_metrics(sat_embeddings, text_embeddings):

        pass
    
    def get_lc_metrics(self):
        set_seed(56)
        geoclap, hparams  = self.get_geoclap()
        print(hparams)
        test_dataloader = self.get_dataloader()
        sat_embeddings = []
        text_embeddings = []
        for i,batch in tqdm(enumerate(test_dataloader)):
            print("batch no:",str(i))
            embeds = self.get_embeddings(batch=batch,model=geoclap,hparams=hparams)
            sat_embeddings.append(embeds['sat_embeddings'])
            text_embeddings.append(embeds['text_embeddings'])
        sat_embeddings = torch.cat(sat_embeddings,axis=0).to(self.device)
        text_embeddings = torch.cat(text_embeddings,axis=0).to(self.device)
        print(sat_embeddings.shape, text_embeddings.shape)
        results =  self.compute_lc_acc_metrics(sat_embeddings, text_embeddings)
        return results


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--sat_type', type=str, default='sentinel')
    parser.add_argument('--dataset_type', type=str, default='')
    args = parser.parse_args()

    if args.sat_type == "sentinel":
        args['geoclap_ckpt_path'] = cfg.sentinel_ckpt_path
    else:
        args['geoclap_ckpt_path'] = cfg.googleEarth_ckpt_path

    #params
    set_seed(56)
    #configure evaluation
    evaluation = ZS_Evaluation(args)
    results = evaluation.get_lc_metrics()
    print("LC results:",results)