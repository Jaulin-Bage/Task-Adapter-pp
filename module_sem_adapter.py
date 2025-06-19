from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import argparse
from utils import read_yaml
params = argparse.Namespace(**read_yaml())
CLIP_VIT_B16_PATH = '/home/hanpeiheng/.cache/clip/ViT-B-16.pt'
CLIP_VIT_L14_PATH = '/home/hanpeiheng/.cache/clip/ViT-L-14.pt'

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        # xs=rearrange(x,'a t c -> (a t) c')
        xs = self.D_fc1(x)
        # xs = rearrange(xs,'(a t) c -> a t c',t=197)
        xs = self.act(xs)
        # xs=rearrange(xs,'a t c -> (a t) c')
        xs = self.D_fc2(xs)
        # xs = rearrange(xs,'(a t) c -> a t c',t=197)

        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 use_adapter: bool,
                 attn_mask
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
        self.attn_mask=attn_mask
        self.use_adapter = use_adapter
        self.sub3_adapter = adapter(d_model, skip_connect=False)

   
    def attention(self, x: torch.Tensor,use_mask=True):
        self.attn_mask = self.attn_mask.to(device=x.device,dtype=torch.float16) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0] if use_mask else self.attn(x, x, x, need_weights=False, attn_mask=None)[0]

    def forward(self,
                x: torch.Tensor
                ) :

        if self.use_adapter:
            # 77 5 512
            x = x + self.attention(self.ln_1(x))

            x3 = rearrange(x, 'l s d -> s l d')
            x3 = self.attention(self.ln_1(x3),use_mask=False)
            x3 = rearrange(x3, 's l d -> l s d')
            x = x + self.sub3_adapter(x3)

            x = x + self.mlp(self.ln_2(x))
            
        else: 
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, depth: int,mask):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(d_model=width, n_head=heads, use_adapter= i >= layers - depth,attn_mask=mask) for i in range(layers)])
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _,block in enumerate(self.resblocks):
            x = block(x)
            
        return x

def clip_encode_text_adapter(adapter_layer):
    model = encode_text(adapter_layer=adapter_layer,
                        transformer_layers=12,
                        transformer_heads=8,
                        transformer_width=512,
                        vocab_size= 49408,
                        )

    checkpoint = torch.jit.load(CLIP_VIT_B16_PATH, map_location='cpu')

    new_ck = {k:v for k,v in checkpoint.state_dict().items() if 'visual'  not in k}
    new_ck.pop("input_resolution")
    new_ck.pop("context_length")
    new_ck.pop("vocab_size")
    convert_weights(model)
    model.load_state_dict(new_ck, strict=False)
    # print(list(model.named_modules()))
    return model

class encode_text(nn.Module):
    def __init__(self,vocab_size,transformer_width, transformer_layers, transformer_heads, adapter_layer):
        super().__init__()

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            depth=adapter_layer,
            mask=self.build_attention_mask()
        )
        self.ln_final=LayerNorm(512)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, 512))
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        for n, m in self.transformer.named_modules():
            if 'adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
        for n, p in self.named_parameters():
            if 'adapter' not in n : # only finetune adapter
                p.requires_grad_(False)
                
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(77, 77)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    def token_cut(cls):
        head=cls[:1]
        context=cls[1:6]
        body=cls[6:]
        return head,context,body
    def forward(self, text:torch.Tensor):

        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x.to(torch.float16))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)