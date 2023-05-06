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
from torchvision.models import vgg16,densenet121,inception_v3,mobilenetv2,vgg16_bn
PRETRAINED=False
LAYERS=24

def sub_net1():
    u_net = vgg16_bn(pretrained=PRETRAINED)
    sub_net_list=[]
    for i in range(LAYERS):
        sub_net_list.append(u_net.features[i])
    # maxpool 4,9,16
    for i in [6,13,23]:
        sub_net_list[i]=nn.MaxPool2d(kernel_size=2, stride=1)
    return nn.Sequential(*sub_net_list)

def sub_net2():
    u_net = vgg16_bn(pretrained=PRETRAINED)
    sub_net_list=[]
    for i in range(LAYERS):
        sub_net_list.append(u_net.features[i])
    # # maxpool 4,9,16
    for i in [23]:
        sub_net_list[i]=nn.MaxPool2d(kernel_size=2, stride=1)
    return nn.Sequential(*sub_net_list)


