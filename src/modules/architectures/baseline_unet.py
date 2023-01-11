import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Dict
from argparse import Namespace, ArgumentParser
from monai.networks.nets import UNet
from monai.networks.layers.utils import get_act_layer, get_norm_layer


# Monai UNet model
class MonaiUNet(nn.Module):
    def __init__(self,
                 conf: Union[Dict, Namespace]):
        super().__init__()
        self.conf = conf

        self.net = UNet(spatial_dims=2,
                        in_channels=self.conf.in_channels,
                        out_channels=32,
                        channels=(32, 64, 128, 256, 384),
                        strides=(1, 2, 2, 2),
                        kernel_size=3,
                        up_kernel_size=3,
                        num_res_units=2,
                        act='PRELU',
                        norm='BATCH',
                        dropout=0.0,
                        bias=True)

        self.norm_seg = get_norm_layer(name='batch', spatial_dims=2, channels=32)
        self.conv_seg = nn.Conv2d(in_channels=32,
                                  out_channels=self.conf.out_channels,
                                  kernel_size=(1, 1))

        self.norm_emb = get_norm_layer(name='batch', spatial_dims=2, channels=32)
        self.conv_emb = nn.Conv2d(in_channels=32,
                                  out_channels=32,
                                  kernel_size=(1, 1))

    def forward(self, x: torch.Tensor):

        assert x.shape[-1] == 1  # atm only 2D case allowed.
        x = self.net(x[..., 0])
        x_seg = self.conv_seg(F.leaky_relu(self.norm_seg(x))).unsqueeze(-1)
        x_emb = self.conv_emb(F.leaky_relu(self.norm_emb(x))).unsqueeze(-1)

        dict_out = {'dense': {'embedded_latents': x_seg},
                    'patched':  {'embedded_latents': x_emb}}

        return dict_out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        return parser
