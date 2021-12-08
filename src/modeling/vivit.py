import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .vit import Transformer

  
class ViViT(nn.Module):
    def __init__(self, cfg,):
                #  image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                #  emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert cfg.MODEL.VIVIT.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert cfg.DATA.IMG_SIZE % cfg.MODEL.VIVIT.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (cfg.DATA.IMG_SIZE // cfg.MODEL.VIVIT.patch_size) ** 2
        patch_dim = cfg.MODEL.VIVIT.in_channels * cfg.MODEL.VIVIT.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = cfg.MODEL.VIVIT.patch_size, p2 = cfg.MODEL.VIVIT.patch_size),
            nn.Linear(patch_dim, cfg.MODEL.VIVIT.dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, cfg.DATA.NUM_SLICES, num_patches + 1, cfg.MODEL.VIVIT.dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, cfg.MODEL.VIVIT.dim))
        self.space_transformer = Transformer(cfg.MODEL.VIVIT.dim, cfg.MODEL.VIVIT.depth, cfg.MODEL.VIVIT.heads, cfg.MODEL.VIVIT.dim_head, cfg.MODEL.VIVIT.dim*cfg.MODEL.VIVIT.scale_dim, cfg.MODEL.VIVIT.dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, cfg.MODEL.VIVIT.dim))
        self.temporal_transformer = Transformer(cfg.MODEL.VIVIT.dim, cfg.MODEL.VIVIT.depth, cfg.MODEL.VIVIT.heads, cfg.MODEL.VIVIT.dim_head, cfg.MODEL.VIVIT.dim*cfg.MODEL.VIVIT.scale_dim, cfg.MODEL.VIVIT.dropout)

        self.dropout = nn.Dropout(cfg.MODEL.VIVIT.emb_dropout)
        self.pool = cfg.MODEL.VIVIT.pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(cfg.MODEL.VIVIT.dim),
            nn.Linear(cfg.MODEL.VIVIT.dim, cfg.MODEL.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)