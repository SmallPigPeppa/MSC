import os
import torchmetrics
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from args import parse_args
import pytorch_lightning as pl
from imagenet_dali import ClassificationDALIDataModule
from torchvision.models import resnet50
from torchprofile import profile_macs


def unified_net():
    u_net = resnet50(pretrained=False)
    u_net.conv1 = nn.Identity()
    u_net.bn1 = nn.Identity()
    u_net.relu = nn.Identity()
    u_net.maxpool = nn.Identity()
    u_net.layer1 = nn.Identity()
    return u_net


class ResNet50_L2(LightningModule):
    def __init__(self):
        super().__init__()
        self.large_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=False).layer1
        )
        self.mid_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=False).layer1
        )
        self.small_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), resnet50(pretrained=False).layer1
        )
        self.unified_net = unified_net()
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)
        self.unified_size = (56, 56)

    def forward(self, imgs):
        small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')

        z1 = self.small_net(small_imgs)
        z2 = self.mid_net(mid_imgs)
        z3 = self.large_net(large_imgs)

        z1 = F.interpolate(z1, size=self.unified_size, mode='bilinear')
        z2 = F.interpolate(z2, size=self.unified_size, mode='bilinear')

        y1 = self.unified_net(z1)
        y2 = self.unified_net(z2)
        y3 = self.unified_net(z3)

        return z1, z2, z3, y1, y2, y3

    def forward_32(self, imgs):
        # small_imgs = F.interpolate(imgs, size=self.small_size, mode='bilinear')
        z1 = self.small_net(imgs)
        z1 = F.interpolate(z1, size=self.unified_size, mode='bilinear')
        y1 = self.unified_net(z1)

        return y1

    def forward_128(self, imgs):
        # mid_imgs = F.interpolate(imgs, size=self.mid_size, mode='bilinear')
        z2 = self.mid_net(imgs)
        z2 = F.interpolate(z2, size=self.unified_size, mode='bilinear')
        y2 = self.unified_net(z2)

        return y2

    def forward_224(self, imgs):
        # large_imgs = F.interpolate(imgs, size=self.large_size, mode='bilinear')
        z3 = self.large_net(imgs)
        y3 = self.unified_net(z3)

        return y3


class ResNet50(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = ResNet50_L2()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.metrics_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model.forward_224(x)

    def share_step(self, batch, batch_idx):
        x, y = batch
        z1, z2, z3, y_hat1, y_hat2, y_hat3 = self(x)

        ce_loss1 = self.ce_loss(y_hat1, y)
        ce_loss2 = self.ce_loss(y_hat2, y)
        ce_loss3 = self.ce_loss(y_hat3, y)

        si_loss1 = self.mse_loss(z1, z2)
        si_loss2 = self.mse_loss(z1, z3)
        si_loss3 = self.mse_loss(z2, z3)

        if si_loss1 < 0.01:
            si_loss1 = 0

        if si_loss2 < 0.01:
            si_loss2 = 0

        if si_loss3 < 0.01:
            si_loss3 = 0

        total_loss = si_loss1 + si_loss2 + si_loss3 + ce_loss1 + ce_loss2 + ce_loss3

        acc1 = self.metrics_acc(y_hat1, y)
        acc2 = self.metrics_acc(y_hat2, y)
        acc3 = self.metrics_acc(y_hat3, y)

        result_dict = {
            "si_loss1": si_loss1,
            "si_loss2": si_loss2,
            "si_loss3": si_loss3,
            "ce_loss1": ce_loss1,
            "ce_loss2": ce_loss2,
            "ce_loss3": ce_loss3,
            "total_loss": total_loss,
            "acc1": acc1,
            "acc2": acc2,
            "acc3": acc3,
        }
        return result_dict

    def training_step(self, batch, batch_idx):
        result_dict = self.share_step(batch, batch_idx)
        train_result_dict = {f'train_{k}': v for k, v in result_dict.items()}
        self.log_dict(train_result_dict, on_epoch=True)
        return result_dict['total_loss']

    def validation_step(self, batch, batch_idx):
        result_dict = self.share_step(batch, batch_idx)
        val_result_dict = {f'val_{k}': v for k, v in result_dict.items()}
        self.log_dict(val_result_dict, on_epoch=True)
        return val_result_dict

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay, momentum=0.9)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=self.args.max_epochs,
            warmup_start_lr=0.01 * self.args.learning_rate,
            eta_min=0.01 * self.args.learning_rate,
        )
        return [optimizer], [scheduler]


# if __name__=="__main__":
#     a=torch.rand(8,3,224,224)
#     model=ResNet18_L2()
#     b=model(a)
#     for bi in b:
#         print(bi.shape)

if __name__ == "__main__":
    args = parse_args()
    # wandb_logger = WandbLogger(name=args.run_name, project=args.project, entity=args.entity, offline=args.offline)
    model = ResNet50(args)
    inputs = torch.rand([8, 3, 224, 224])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    model = model.to(device)
    model.eval()
    macs = profile_macs(model, inputs)
    flops = macs / 1e9
    print(f"FLOPs:{flops}")
