import os
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from imagenet_dali import ClassificationDALIDataModule
from args import parse_args
import pytorch_lightning as pl
from torchvision.models import vgg16, densenet121, inception_v3, mobilenet_v2
from torchvision.ops import Conv2dNormActivation

PRETRAINED = False
LAYERS = 10


def sub_net1():
    u_net = mobilenet_v2(pretrained=PRETRAINED)
    sub_net_list = []
    for i in range(LAYERS):
        sub_net_list.append(u_net.features[i])
    sub_net_list[0] = Conv2dNormActivation(in_channels=3, out_channels=32, kernel_size=1, stride=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
    sub_net_list[2].conv[1] = Conv2dNormActivation(in_channels=96, out_channels=96, kernel_size=3, stride=1,groups=96, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
    sub_net_list[4].conv[1] = Conv2dNormActivation(in_channels=144, out_channels=144, kernel_size=3, stride=1,groups=144, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
    return nn.Sequential(*sub_net_list)


def sub_net2():
    u_net = mobilenet_v2(pretrained=PRETRAINED)
    sub_net_list = []
    for i in range(LAYERS):
        sub_net_list.append(u_net.features[i])
    sub_net_list[0] = Conv2dNormActivation(in_channels=3, out_channels=32, kernel_size=3, stride=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)
    return nn.Sequential(*sub_net_list)
