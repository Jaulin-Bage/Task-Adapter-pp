import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
import math
import clip
from einops import rearrange
from module_adapter import clip_vit_base_patch16_adapter
from module_sem_adapter import clip_encode_text_adapter
from utils import *

class MetaTemplate(nn.Module):
    def __init__(self, n_way, n_support, n_query):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query
    
    @abstractmethod
    def forward(self,x):
        pass
    
    def distribute_backbone(self,devices):
        self.feature.cuda(0)
        self.feature = torch.nn.DataParallel(self.feature,device_ids = devices)
    

params = argparse.Namespace(**read_yaml())
class TaskAdapter(MetaTemplate):
    def __init__(self, n_way, n_support, n_query, dataset,adapter_depth,text_depth):
        super().__init__(n_way, n_support, n_query)


        if dataset == 'hmdb51':
            self.cls_txt = clip.tokenize(hmdb_cls).cuda()
            self.cls_name = hmdb_c
            self.entxt = read_yml(dataset)#content
        elif dataset == 'ucf101':
            self.cls_txt = clip.tokenize(ucf_cls).cuda()
            self.cls_name = ucf_c
            self.entxt = read_yml(dataset)#content
        elif dataset == 'kinetics':
            self.cls_txt = clip.tokenize(kinetics_cls).cuda()
            self.cls_name = kinetics_c
            self.entxt = read_yml(dataset)#content
        elif 'something' in dataset:
            self.cls_txt = clip.tokenize(smsm_cls).cuda()
            self.cls_name = smsm_c
            self.entxt = read_yml(dataset)#content
        
        self.feature = clip_vit_base_patch16_adapter(embed_dim=512, adapter_layers=adapter_depth)
        self.text = clip_encode_text_adapter(adapter_layer=text_depth)

        self.ca=CrossAttention()
    
    
    def forward(self, x, label):
        label_idx = list(label[:,0].numpy())
        N,C,H,W = x.shape
        x = x.reshape(self.n_way*(self.n_query+self.n_support), 8 , C, H, W)
        x = x.permute(0, 2, 1, 3, 4) # B C T H W

        x = x.reshape(self.n_way, (self.n_query+self.n_support), C , 8, H, W)
    
        support_images = x[:, :self.n_support].reshape(self.n_way*self.n_support, C, 8, H, W)
        target_images = x[:,self.n_support:].reshape(self.n_way*self.n_query, C, 8, H, W)

        z_support  = self.feature(support_images).reshape(self.n_way*self.n_support, 8, -1)
        z_query = self.feature(target_images).reshape(self.n_way*self.n_query, 8, -1)
        
        z_proto = z_support.reshape(self.n_way, self.n_support, 8, -1).mean(1)
        z_query = z_query.reshape(self.n_way*self.n_query, 8, -1 )
        
        q_reshape=z_query.permute(1,0,2)
        q_aft_tm=[]

        for frame in range(7):
            q_aft_tm.append(self.ca(q_reshape[frame],q_reshape[frame+1],q_reshape[frame+1])+q_reshape[frame+1]) 
        
        q_aft_tm=torch.stack(q_aft_tm,dim=0)


        if True:
            label_name = [self.cls_name[i] for i in label_idx]
            enhtxt=[]
            tmp_prompt = []
            for i in label_name:
                
                for j in self.entxt[i]['sub_act_en_li']:
                    one_prompt = f'A video of action about {i}: {j}'
                    tmp_prompt.append(one_prompt)
                enhtxt.append(tmp_prompt)
                tmp_prompt=[]

            enh_embedding = []
            for i in range(self.n_way):
                enh_embedding.append(self.text(clip.tokenize(enhtxt[i]).to('cuda')))

            enh_embedding = torch.stack(enh_embedding,dim=0).permute(1,0,2)
            cos_score = 0
        for i in range(3):
            cos_score += (cosine_similarity(enh_embedding[i],q_aft_tm[2*i])+cosine_similarity(enh_embedding[i],q_aft_tm[2*i+1])+cosine_similarity(enh_embedding[i],q_aft_tm[2*i+2]))

        cos_score = cos_score/9


        
        sem_dists = cos_score
        dists = cosine_similarity(z_query.mean(1), z_proto.mean(1)).squeeze()

        return sem_dists, dists


class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads = 1, dim_head = 512, dropout = 0.1):
        super().__init__()
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.ln_1=nn.LayerNorm(dim) if params.dataset=='somethingotam'  else nn.Identity()
        self.attend = nn.Softmax(dim = -1)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q,k,v):
        
        dots = q@k.transpose(0,1) * self.scale

        attn = self.attend(dots)

        out = attn@v
        
        return self.ln_1(out)

