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
from torchvision.models import resnet18


class ResNet18(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = resnet18(pretrained=False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.metrics_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.ce_loss(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.metrics_acc(y_hat, y), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x1 = F.interpolate(x, size=32, mode='bilinear')
        x1 = F.interpolate(x1, size=224, mode='bilinear')

        x2 = F.interpolate(x, size=128, mode='bilinear')
        x2 = F.interpolate(x2, size=224, mode='bilinear')

        x3 = x

        y_hat1 = self(x1)
        y_hat2 = self(x2)
        y_hat3 = self(x3)

        loss1 = self.ce_loss(y_hat1, y)
        loss2 = self.ce_loss(y_hat2, y)
        loss3 = self.ce_loss(y_hat3, y)
        self.log("val_loss1", loss1)
        self.log("val_loss2", loss2)
        self.log("val_loss3", loss3)
        self.log("val_acc1", self.metrics_acc(y_hat1, y), on_epoch=True)
        self.log("val_acc2", self.metrics_acc(y_hat2, y), on_epoch=True)
        self.log("val_acc3", self.metrics_acc(y_hat3, y), on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            momentum=0.9
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=self.args.max_epochs,
            warmup_start_lr=0.01 * self.args.learning_rate,
            eta_min=0.01 * self.args.learning_rate,
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(19)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=args.checkpoint_dir, save_top_k=1)
    wandb_logger = WandbLogger(name=args.run_name, project=args.project, entity=args.entity, offline=args.offline)
    model = ResNet18(args)

    if args.resume_from_checkpoint is not None:
        trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                             callbacks=[checkpoint_callback, lr_monitor],
                                             resume_from_checkpoint=args.resume_from_checkpoint, precision=16,
                                             gradient_clip_val=1.0,
                                             check_val_every_n_epoch=args.eval_every)
    else:
        trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                             callbacks=[checkpoint_callback, lr_monitor], precision=16,
                                             gradient_clip_val=1.0,
                                             check_val_every_n_epoch=args.eval_every)

    try:
        from pytorch_lightning.loops import FitLoop


        class WorkaroundFitLoop(FitLoop):
            @property
            def prefetch_batches(self) -> int:
                return 1


        trainer.fit_loop = WorkaroundFitLoop(
            trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs
        )
    except:
        pass

    dali_datamodule = ClassificationDALIDataModule(
        train_data_path=os.path.join(args.dataset_path, 'train'),
        val_data_path=os.path.join(args.dataset_path, 'val'),
        num_workers=args.num_workers,
        batch_size=args.batch_size)

    trainer.fit(model, datamodule=dali_datamodule)
