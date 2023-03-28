import os
import torchmetrics
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
from args import parse_args
import torch.nn.functional as F


class MultiScaleNet(nn.Module):
    def __init__(self):
        super(MultiScaleNet, self).__init__()
        self.u_net = torchvision.models.resnet50(pretrained=False)
        self.small_size = (32, 32)
        self.mid_size = (128, 128)
        self.large_size = (224, 224)

    def forward(self, x):
        small_imgs = F.interpolate(x, size=self.small_size, mode='bilinear')
        mid_imgs = F.interpolate(x, size=self.mid_size, mode='bilinear')
        large_imgs = F.interpolate(x, size=self.large_size, mode='bilinear')

        small_imgs = F.interpolate(small_imgs, size=self.large_size, mode='bilinear')
        mid_imgs = F.interpolate(mid_imgs, size=self.large_size, mode='bilinear')

        y3 = self.u_net(large_imgs)

        with torch.no_grad():
            y1 = self.u_net(small_imgs)
            y2 = self.u_net(mid_imgs)


        return y1, y2, y3


class ResNet50(LightningModule):
    def __init__(self, max_epochs: int, learning_rate: float, batch_size: int, weight_decay: float, dataset_path: str):
        super().__init__()
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dataset_path = dataset_path
        self.model = MultiScaleNet()
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat1, y_hat2, y_hat3 = self(x)
        loss = self.criterion(y_hat3, y)
        self.log("train_loss", loss)
        self.log("train_acc1", self.train_acc(y_hat1, y), on_epoch=True)
        self.log("train_acc2", self.train_acc(y_hat2, y), on_epoch=True)
        self.log("train_acc3", self.train_acc(y_hat3, y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat1, y_hat2, y_hat3 = self(x)
        loss = self.criterion(y_hat3, y)
        self.log("val_loss", loss)
        self.log("val_acc1", self.val_acc(y_hat1, y), on_epoch=True)
        self.log("val_acc2", self.val_acc(y_hat2, y), on_epoch=True)
        self.log("val_acc3", self.val_acc(y_hat3, y), on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,momentum=0.9)
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
        hflip_prob = 0.5
        interpolation = InterpolationMode.BILINEAR
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=interpolation),
            transforms.RandomHorizontalFlip(hflip_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, "train"), transform=transform)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        interpolation = InterpolationMode.BILINEAR
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=interpolation),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.ImageFolder(os.path.join(self.dataset_path, "val"), transform=transform)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)


if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(19)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=args.checkpoint_dir, save_top_k=1)
    wandb_logger = WandbLogger(name=args.run_name, project=args.project, entity=args.entity, offline=args.offline)
    model = ResNet50(args.max_epochs, args.learning_rate, args.batch_size, args.weight_decay, args.dataset_path)

    if args.resume_from_checkpoint is not None:
        trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                             callbacks=[checkpoint_callback, lr_monitor],
                                             resume_from_checkpoint=args.resume_from_checkpoint, precision=16,gradient_clip_val=1.0,
                                             check_val_every_n_epoch=args.eval_every)
    else:
        trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                             callbacks=[checkpoint_callback, lr_monitor], precision=16,gradient_clip_val=1.0,
                                             check_val_every_n_epoch=args.eval_every)

    trainer.fit(model)

