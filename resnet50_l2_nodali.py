import os
import torchmetrics
import torch
from torch import nn
import torchvision
from torchvision.transforms.functional import InterpolationMode
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from args import parse_args
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import transforms, datasets

PRETRAINED = False


def unified_net():
    u_net = torchvision.models.resnet50(pretrained=PRETRAINED)
    u_net.conv1 = nn.Identity()
    u_net.bn1 = nn.Identity()
    u_net.relu = nn.Identity()
    u_net.maxpool = nn.Identity()
    u_net.layer1 = nn.Identity()
    return u_net


class MultiScaleNet(nn.Module):
    def __init__(self):
        super(MultiScaleNet, self).__init__()
        self.large_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=PRETRAINED).layer1
        )
        self.mid_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), resnet50(pretrained=PRETRAINED).layer1
        )
        self.small_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), resnet50(pretrained=PRETRAINED).layer1
        )
        self.unified_net = unified_net()
        self.unified_size = (56, 56)

    def forward(self, x):
        x1, x2, x3 = x
        z1 = self.small_net(x1)
        z2 = self.mid_net(x2)
        z3 = self.large_net(x3)

        z1 = F.interpolate(z1, size=self.unified_size, mode='bilinear')
        z2 = F.interpolate(z2, size=self.unified_size, mode='bilinear')

        y1 = self.unified_net(z1)
        y2 = self.unified_net(z2)
        y3 = self.unified_net(z3)

        return z1, z2, z3, y1, y2, y3


class ResNet50(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.max_epochs = args.max_epochs
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.weight_decay = args.weight_decay
        self.dataset_path = args.dataset_path
        self.model = MultiScaleNet()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def share_step(self, batch, batch_idx):
        x, y = batch
        print(len(batch))
        print(x.shape)

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

        total_loss = si_loss1 + si_loss2 + si_loss3 + ce_loss3

        acc1 = self.train_acc(y_hat1, y)
        acc2 = self.train_acc(y_hat2, y)
        acc3 = self.train_acc(y_hat3, y)

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
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=0.9
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=args.max_epochs,
            warmup_start_lr=0.01 * self.learning_rate,
            eta_min=0.01 * self.learning_rate,
        )
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        size = 224
        hflip_prob = 0.5
        interpolation = InterpolationMode.BILINEAR

        def process_image(img, sizes=[224, 128, 32], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            resized_imgs = [transforms.Resize(s, interpolation=interpolation)(img) for s in sizes]
            tensor_imgs = [transforms.ToTensor()(resized_img) for resized_img in resized_imgs]
            normalized_imgs = [transforms.Normalize(mean, std)(tensor_img) for tensor_img in tensor_imgs]
            return normalized_imgs

        transform = transforms.Compose([
            transforms.RandomResizedCrop(size, interpolation=interpolation),
            transforms.RandomHorizontalFlip(hflip_prob),
            transforms.Lambda(process_image)
        ])

        dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, "train"), transform=transform)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        size = 224
        interpolation = InterpolationMode.BILINEAR

        def process_image(img, sizes=[224, 128, 32], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            resized_imgs = [transforms.Resize(s, interpolation=interpolation)(img) for s in sizes]
            tensor_imgs = [transforms.ToTensor()(resized_img) for resized_img in resized_imgs]
            normalized_imgs = [transforms.Normalize(mean, std)(tensor_img) for tensor_img in tensor_imgs]
            return normalized_imgs

        transform = transforms.Compose([
            transforms.Resize(256, interpolation=interpolation),
            transforms.CenterCrop(size),
            transforms.Lambda(process_image),
        ])

        dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, "val"), transform=transform)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)


if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(19)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc3", mode="min", dirpath=args.checkpoint_dir, save_top_k=1)
    wandb_logger = WandbLogger(name=args.run_name, project=args.project, entity=args.entity, offline=args.offline)
    model = ResNet50(args)

    if args.resume_from_checkpoint is not None:
        trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                             callbacks=[checkpoint_callback, lr_monitor],
                                             resume_from_checkpoint=args.resume_from_checkpoint, precision=16,
                                             gradient_clip_val=0.5,
                                             check_val_every_n_epoch=args.eval_every)
    else:
        trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                             callbacks=[checkpoint_callback, lr_monitor], precision=16,
                                             gradient_clip_val=0.5,
                                             check_val_every_n_epoch=args.eval_every)

    trainer.fit(model)
