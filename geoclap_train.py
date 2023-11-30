import sys
import pytorch_lightning as pl
import torch.nn as nn
import torch
import os
from .config import cfg
from .models import SATMAE,AudioCLAP
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "300"

def l2normalize(batch_embeddings):
    return batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

#computer cross entropy for the similarity matrix both rowwise and columnwise
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    modality1_loss = contrastive_loss(similarity)
    modality2_loss = contrastive_loss(similarity.t())
    return (modality1_loss + modality2_loss) / 2.0

def get_loss(modality1_embeddings, modality2_embeddings,logit_scale):
    #similarity between moadality1 and modality2
    logits_per_modality1 = torch.matmul(modality1_embeddings,modality2_embeddings.t())*logit_scale
    #compute cross_entropy loss between the cross-modal similarities and hard gt
    loss_mod1mod2 = clip_loss(logits_per_modality1)

    return loss_mod1mod2, logits_per_modality1


class GeoCLAP(pl.LightningModule):
    def __init__(self, hparams):

        #save paramaters
        super().__init__()
        self.save_hyperparameters(hparams)
         #set path attributes
        self.valid_end_list =[]
        #Data modality: Satellite Image
        if self.hparams.sat_encoder == 'SatMAE':
            self.sat_encoder = SATMAE.SatMAE(pretrained_models_path=cfg.pretrained_models_path,device=self.device,fc_dim = self.hparams.fc_dim,metadata_type=self.hparams.metadata_type).to(self.device)     
        else:
            raise NotImplementedError("Only implemented Satellite image encoder is SATMAE")

        #Data modality: Audio and/or text
        if self.hparams.audio_encoder == 'AudioCLAP': #accepts either audio or text
            if 'audio' in self.hparams.data_type:
                if not self.hparams.saved_audio_embeds: # if frozen embeddings are NOT already saved 
                    self.audio_encoder = AudioCLAP.AudioCLAP_audiomodel(freeze=self.hparams.freeze_audio_model)
            if 'text' in self.hparams.data_type:
                if not self.hparams.saved_text_embeds: # if frozen embeddings are NOT already saved 
                    self.text_encoder = AudioCLAP.AudioCLAP_textmodel(freeze=self.hparams.freeze_text_model)
        
        if not self.hparams.saved_audio_embeds:
            if 'audio' in self.hparams.data_type and self.hparams.freeze_audio_model:
                self.audio_encoder.eval()
                for params in self.audio_encoder.parameters():
                    params.requires_grad=False

        if not self.hparams.saved_text_embeds:
            if 'text' in self.hparams.data_type and self.hparams.freeze_text_model:
                self.text_encoder.eval()
                for params in self.text_encoder.parameters():
                    params.requires_grad=False
            
        #trainable satellite image encoder to get embeddings for satellite image
        self.sat_encoder.train()

        self.temp_layer = AudioCLAP.temp_layer(self.hparams)
                
        self.temp_clip = self.hparams.temp_clip

    def get_embeds(self,batch):
        embeds = {'sat_embeddings':None, 'audio_embeddings':None, 'text_embeddings':None}
        if self.hparams.metadata_type == 'lat_long':
            embeds['sat_embeddings']  = l2normalize(self.sat_encoder(batch['sat'].to(self.device),batch['lat_long'].to(self.device)))
        else:
            embeds['sat_embeddings']  = l2normalize(self.sat_encoder(batch['sat'].to(self.device)))

        if self.hparams.data_type == 'sat_audio':
            output = {}
            if self.hparams.saved_audio_embeds:
                output['audio_embeddings'] = batch['audio'].to(self.device)
            else:
                batch_audio = {}
                for key in batch['audio'].keys():
                    batch_audio[key] = batch['audio'][key].to(self.device)

                output['audio_embeddings'] = self.audio_encoder(batch_audio)
            
            embeds['audio_embeddings'] = l2normalize(output['audio_embeddings'])

        if self.hparams.data_type == 'sat_audio_text':
            output = {}
            if self.hparams.saved_audio_embeds:
                output['audio_embeddings'] = batch['audio'].to(self.device)
            else:
                batch_audio = {}
                for key in batch['audio'].keys():
                    batch_audio[key] = batch['audio'][key].to(self.device) 
                output['audio_embeddings'] = self.audio_encoder(batch_audio)
            
            if self.hparams.saved_text_embeds:
                output['text_embeddings'] = batch['text'].to(self.device)
            else:
                batch_text = {}
                for key in batch['text'].keys():
                    batch_text[key] = batch['text'][key].to(self.device)
                output['text_embeddings'] = self.text_encoder(batch_text)
            
            embeds['audio_embeddings'], embeds['text_embeddings'] = l2normalize(output['audio_embeddings']), l2normalize(output['text_embeddings'])

        return embeds
    def forward(self, batch):
        embeds = self.get_embeds(batch)
        return embeds
    
    #clamp the temperature parameter
    def on_before_zero_grad(self, *args, **kwargs):
        self.temp_layer.logit_scale_ia.data = torch.clamp(self.temp_layer.logit_scale_ia.data, min=1.0, max=np.log(self.hparams.temp_clip))
        if self.hparams.data_type == 'sat_audio_text':
            self.temp_layer.logit_scale_it.data = torch.clamp(self.temp_layer.logit_scale_it.data, min=1.0, max=np.log(self.hparams.temp_clip))
            if not self.hparams.freeze_text_model:
                self.temp_layer.logit_scale_at.data = torch.clamp(self.temp_layer.logit_scale_at.data, min=1.0, max=np.log(self.hparams.temp_clip))
    
    def shared_step(self, batch):
        embeds = self(batch)

        audio_embeddings = embeds['audio_embeddings']
        sat_embeddings = embeds['sat_embeddings']
        text_embeddings = embeds['text_embeddings']

        #Calculate loss
        logit_scale_ia = self.temp_layer.logit_scale_ia.exp()
        loss_ia, logits_per_satImage_audio = get_loss(modality1_embeddings = sat_embeddings, 
                                                    modality2_embeddings=audio_embeddings,
                                                    logit_scale=logit_scale_ia)
            
        if self.hparams.data_type == 'sat_audio_text':
            logit_scale_it = self.temp_layer.logit_scale_it.exp()
            loss_SatText, logits_per_satImage_text = get_loss(modality1_embeddings = sat_embeddings, 
                                                            modality2_embeddings=text_embeddings,
                                                            logit_scale=logit_scale_it)
            if not self.hparams.freeze_text_model:
                logit_scale_at = self.temp_layer.logit_scale_at.exp() 
                loss_AudioText, logits_per_Audio_text = get_loss(modality1_embeddings = audio_embeddings, 
                                                            modality2_embeddings=text_embeddings,
                                                            logit_scale=logit_scale_at)     
                loss = (loss_ia + loss_SatText + loss_AudioText)/3
        
                return {'loss':loss,
                        'loss_ia':loss_ia,
                        'loss_it':loss_SatText,
                        'loss_at':loss_AudioText,
                        'logits_per_satImage_audio': logits_per_satImage_audio,
                        'normalized_audio_embeddings': audio_embeddings,
                        'normalized_satellite_embeddings': sat_embeddings
                        }
            else:
                loss = (1-self.hparams.text_loss_weight)*loss_ia + self.hparams.text_loss_weight*loss_SatText
                return {'loss':loss,
                        'loss_ia':loss_ia,
                        'loss_it':loss_SatText,
                        'logits_per_satImage_audio': logits_per_satImage_audio,
                        'normalized_audio_embeddings': audio_embeddings,
                        'normalized_satellite_embeddings': sat_embeddings
                        }
            
        else:
            return {'loss':loss_ia,
                'logits_per_satImage_audio': logits_per_satImage_audio,
                'normalized_audio_embeddings': audio_embeddings,
                'normalized_satellite_embeddings': sat_embeddings
                }
        
    
    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        if self.hparams.data_type == 'sat_audio':
            self.log('loss', outputs['loss'], sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('logit_scale_ia',self.temp_layer.logit_scale_ia.data,sync_dist=True, batch_size=self.hparams.train_batch_size)
        if self.hparams.data_type == 'sat_audio_text':
            self.log('loss', outputs['loss'], sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('loss_ia', outputs['loss_ia'], sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('loss_it', outputs['loss_it'], sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('logit_scale_ia',self.temp_layer.logit_scale_ia.data,sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('logit_scale_it',self.temp_layer.logit_scale_it.data,sync_dist=True, batch_size=self.hparams.train_batch_size)
            if not self.hparams.freeze_text_model: 
                self.log('loss_at', outputs['loss_at'], sync_dist=True, batch_size=self.hparams.train_batch_size)
                self.log('logit_scale_at',self.temp_layer.logit_scale_at.data,sync_dist=True, batch_size=self.hparams.train_batch_size)
        
        return outputs['loss']
        
    
    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        val_loss = outputs['loss']

        self.log('val_loss', val_loss, sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
        final_output = {'val_loss':outputs['loss'],'normalized_satellite_embeddings':outputs['normalized_satellite_embeddings'], 'normalized_audio_embeddings':outputs['normalized_audio_embeddings']}
        self.valid_end_list.append(final_output)
        return final_output