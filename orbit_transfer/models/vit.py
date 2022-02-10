from .group_equivariant_layers import *
import torch
from torch import nn
from einops import rearrange, repeat
from vit_pytorch.vit import Transformer
from vit_pytorch.vit import ViT as OrigViT


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3
    ):
        super().__init__()
        assert (
            image_size % patch_size == 0
        ), "image dimensions must be divisible by the patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


def vit_builder(seed: int, config):
    if config.orig_vit:
        model = OrigViT(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            channels=config.channels,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            dim_head=config.mlp_dim,
        )
    else:
        model = ViT(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_classes=config.num_classes,
            channels=config.channels,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
        )
    return model
