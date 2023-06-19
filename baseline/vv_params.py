import os
import torchmetrics
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from args import parse_args
import pytorch_lightning as pl
from imagenet_dali import ClassificationDALIDataModule
from torchvision.models import vgg16_bn, densenet121, inception_v3, mobilenet_v2, resnet50

PRETRAINED = False


class MSC(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = resnet50(pretrained=PRETRAINED)
        # self.model = densenet121(pretrained=PRETRAINED)
        # self.model = mobilenet_v2(pretrained=PRETRAINED)
        # self.model = vgg16_bn(pretrained=PRETRAINED)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.metrics_acc = torchmetrics.Accuracy()
        # trunc

    def forward(self, x):
        return self.model(x)

    def share_step(self, batch, batch_idx):
        x, y = batch
        y_hat3 = self(x)
        ce_loss3 = self.ce_loss(y_hat3, y)

        total_loss = ce_loss3

        acc3 = self.metrics_acc(y_hat3, y)

        result_dict = {
            "ce_loss3": ce_loss3,
            "total_loss": total_loss,
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
#     model=vgg16()
#     model=densenet121()
#     a=torch.rand(8,3,224,224)
#     model=DenseNet121_L2()
#
#     b=model(a)
#     for bi in b:
#         print(bi.shape)

if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(19)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, save_last=True)
    wandb_logger = WandbLogger(name=f"{args.run_name}", project=args.project, entity=args.entity,
                               offline=args.offline)
    model = MSC(args)

    if args.resume_from_checkpoint is not None:
        trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                             callbacks=[checkpoint_callback, lr_monitor],
                                             resume_from_checkpoint=args.resume_from_checkpoint, precision=16,
                                             # gradient_clip_val=1.0,
                                             check_val_every_n_epoch=args.eval_every)
    else:
        trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                             callbacks=[checkpoint_callback, lr_monitor], precision=16,
                                             # gradient_clip_val=1.0,
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
