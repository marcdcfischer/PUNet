import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict
from argparse import Namespace, ArgumentParser
from monai.networks.nets import SwinUNETR
import math
from monai.networks.layers.utils import get_act_layer, get_norm_layer


# Monai UNETR model
class MonaiSwinUNETR(nn.Module):
    def __init__(self,
                 conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf
        self.img_size = self.conf.patch_size_teacher
        self.norm_name = 'instance'  # atm hardcoded.

        self.net = SwinUNETR(img_size=self.img_size[:-1],
                             in_channels=self.conf.in_channels,
                             out_channels=32,
                             feature_size=24,
                             norm_name=self.norm_name,
                             use_checkpoint=True,
                             spatial_dims=2)

        self.norm_seg = get_norm_layer(name='batch', spatial_dims=2, channels=32)
        self.conv_seg = nn.Conv2d(in_channels=32,
                                  out_channels=self.conf.out_channels,
                                  kernel_size=(1, 1))

        self.norm_emb = get_norm_layer(name='batch', spatial_dims=2, channels=32)
        self.conv_emb = nn.Conv2d(in_channels=32,
                                  out_channels=32,
                                  kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):

        # pad students to teacher size
        paddings = (0, 0, 0, 0, 0, 0)
        if any([x_ < y_ for x_, y_ in zip(x.shape[2:], self.img_size)]):
            x_size = x.shape[2:]
            paddings = [math.floor((self.img_size[0] - x_size[0]) / 2), math.ceil((self.img_size[0] - x_size[0]) / 2),
                        math.floor((self.img_size[1] - x_size[1]) / 2), math.ceil((self.img_size[1] - x_size[1]) / 2),
                        math.floor((self.img_size[2] - x_size[2]) / 2), math.ceil((self.img_size[2] - x_size[2]) / 2)]
            paddings[-1] = 0 if self.img_size[2] == 1 else paddings[-1]  # don't pad depth singleton dim.
            x = F.pad(x, tuple(reversed(paddings)), mode='constant', value=min(x.min(), -1.))  # F.pad needs reverse order (starting from last)

        assert x.shape[-1] == 1  # atm only 2D case allowed.
        x = self.net(x[..., 0])

        x_seg = self.conv_seg(F.leaky_relu(self.norm_seg(x))).unsqueeze(-1)
        x_emb = self.conv_emb(F.leaky_relu(self.norm_emb(x))).unsqueeze(-1)

        # crop students back to teacher size
        if any([x_ > 0 for x_ in paddings]):
            x_seg = x_seg[...,
                          paddings[0]: self.img_size[0] - paddings[1],
                          paddings[2]: self.img_size[1] - paddings[3],
                          paddings[4]: self.img_size[2] - paddings[5]]  # F.pad needs reverse order (starting from last)

            x_emb = x_emb[...,
                          paddings[0]: self.img_size[0] - paddings[1],
                          paddings[2]: self.img_size[1] - paddings[3],
                          paddings[4]: self.img_size[2] - paddings[5]]  # F.pad needs reverse order (starting from last)

        dict_out = {'dense': {'embedded_latents': x_seg},
                    'patched':  {'embedded_latents': x_emb}}

        return dict_out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        return parser
