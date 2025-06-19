# modified from: https://github.com/openai/CLIP/blob/a9b1bf5920416aaeaec965c25dd9e8f98c864f16/clip/model.py

from typing import Tuple
from collections import OrderedDict
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

CLIP_VIT_B16_PATH = '/home/hanpeiheng/.cache/clip/ViT-B-16.pt'
CLIP_VIT_L14_PATH = '/home/hanpeiheng/.cache/clip/ViT-L-14.pt'


class Adapter(nn.Module):

    def __init__(self, in_channels, adapter_channels, kernel_size):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, adapter_channels)
        self.conv = nn.Conv3d(
            adapter_channels, adapter_channels,
            kernel_size=kernel_size,
            stride=(1, 1, 1),
            padding=tuple(x // 2 for x in kernel_size),
            groups=adapter_channels,
        )
        self.fc2 = nn.Linear(adapter_channels, in_channels)
        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, T):
        BT, L, C = x.size()
        B = BT // T
        Ca = self.conv.in_channels
        H = W = round(math.sqrt(L - 1))
        assert L - 1 == H * W
        x_id = x
        x = x[:, 1:, :]
        x = self.fc1(x)
        x = x.view(B, T, H, W, Ca).permute(0, 4, 1, 2, 3).contiguous()

        x = self.conv(x)

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(BT, L - 1, Ca)
        x = self.fc2(x)
        x_id[:, 1:, :] += x
        return x_id


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Aim_adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x



class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 adapter_width: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 use_adapter: bool,
                 ) -> None:
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

        adapter_class = functools.partial(
            Adapter,
            in_channels=d_model,
            adapter_channels=adapter_width,
            kernel_size=adapter_kernel_size,
        )
        self.use_adapter = use_adapter
        self.t_adapter = Aim_adapter(d_model, skip_connect=False)
        self.task_adapter = Aim_adapter(d_model, skip_connect=False)
        self.s_adapter = Aim_adapter(d_model)
        self.MLP_adapter = Aim_adapter(d_model, skip_connect=False)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.size()
        H = self.attn.num_heads

        qkv = F.linear(x, weight=self.attn.in_proj_weight, bias=self.attn.in_proj_bias)
        qkv = qkv.view(B, L, H * 3, -1).permute(0, 2, 1, 3)
        q, k, v = qkv.split([H, H, H], dim=1)

        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = torch.softmax((q@k.transpose(-2, -1) * scale_factor), dim=-1)
        attention_map = q@k.transpose(-2, -1) * scale_factor
        out =  attn_weight @ v
        
        out = out.permute(0, 2, 1, 3).flatten(-2)
        out = self.attn.out_proj(out)

        return out
        

    def forward(self,
                x: torch.Tensor,
                num_frames: int
                ) -> torch.Tensor:
        
        bt,l,d = x.shape# BT, L, D

        if self.use_adapter:
            xt = rearrange(x, '(b t) l d -> (b l) t d', t=8) # t is number of frames
            xt = self.attention(self.ln_1(xt)) 
            xt = rearrange(xt, '(b l) t d -> (b t) l d', l=l)
            x = x + self.t_adapter(xt) #temporal adaptation
           
            x = x + self.s_adapter(self.attention(self.ln_1(x))) # spatial adaptation

            xb = rearrange(x, '(b t) l d -> (t l) b d', t=8) 
            xb = self.attention(self.ln_1(xb))
            xb = rearrange(xb, '(t l) b d -> (b t) l d', l=l)
            x = x + self.task_adapter(xb)# task adaptation
            
            
            xn = self.ln_2(x)
            x = x + self.mlp(xn) +  self.MLP_adapter(xn) # joint adaptation
        else:
            x = x + self.attention(self.ln_1(x))
            xn = self.ln_2(x)
            x = x + self.mlp(xn)

        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                adapter_width=adapter_width,
                adapter_kernel_size=adapter_kernel_size,
                use_adapter= i >= layers - adapter_layers, # Last M layers
            )
            for i in range(layers)
        ]
        )

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x, num_frames)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 embed_dim: int,
                 adapter_width: int,
                 adapter_layers: int,
                 adapter_kernel_size: Tuple[int, int, int],
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(
                (input_resolution // patch_size) ** 2 + 1, width
            )
        )
        self.temporal_embedding = nn.Parameter(torch.zeros(1, 8, width))#8 is num frames
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads,
            adapter_width, adapter_layers, adapter_kernel_size)

        self.ln_post = LayerNorm(width)

    
        
        self.dropout = nn.Dropout(0.5)
        self.proj = nn.Parameter(scale * torch.randn(width, embed_dim))

        
        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 's_adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 't_adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)


        for n, p in self.named_parameters():
            if 'adapter' not in n and 'temporal_embedding' not in n: # only finetune adapter and temporal embedding
                p.requires_grad_(False)
 

    def forward(self, x: torch.Tensor):
        B, T = x.size(0), x.size(2)
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)# n*t, c, h, w
        x = self.conv1(x)
        # shape = [*, width, grid ** 2]
        # after the conv, the image will be devided into sevral patchs. n*t, 768, patch_size, patch_size
        spatial_size = tuple(x.size()[2:])
        # ps, ps
        x = x.flatten(-2).permute(0, 2, 1)
        # n*t, patch_size^2, 768
        x = torch.cat([
            self.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1), x
            ], dim=1)  # [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # n*t, patch_size^2+1, 768
        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=8)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)

        x = self.ln_pre(x)

        x = x.view(B, T, x.size(1), x.size(2)).flatten(0, 1) # BT, L, D

        x = self.transformer(x, T)

        x = x.contiguous().view(B, T, spatial_size[0] * spatial_size[1] + 1, x.size(-1))
        x = x[:, :, 0, :]

        x = self.ln_post(x)
        x = x@ self.proj


        return x



def clip_vit_base_patch16_adapter(**kwargs):
    model = VisionTransformer(
        input_resolution=224,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        adapter_width=384,
        adapter_kernel_size=(3, 1, 1),
        **kwargs,
    )
    assert CLIP_VIT_B16_PATH is not None, \
        'Please set CLIP_VIT_B16_PATH in configs.py'
    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')
    model.load_state_dict(checkpoint.visual.state_dict(), strict=False)
    return model