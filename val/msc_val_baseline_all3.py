import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import wandb
import torchmetrics

import torch.nn.functional as F
from tqdm import tqdm
from torchprofile import profile_macs
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    from torchvision.models import inception_v3,vgg16_bn, mobilenet_v2

    model = mobilenet_v2(pretrained=True)
    model.eval()

    from torchvision.models import alexnet

    # model = alexnet(pretrained=True)

    resolutions = list(range(32, 225, 16))
    input = torch.rand([4, 3, 224, 224])
    output = model(input)
    a = 0
