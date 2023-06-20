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
from torchvision.models import resnet50, densenet121, vgg16_bn, mobilenet_v2
from torch.utils.data import DataLoader
from transfer_dataset import *

PRETRAINED = False


class MSC(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model == 'resnet50':
            self.model = resnet50(pretrained=PRETRAINED)
        elif args.model == 'densenet121':
            self.model = densenet121(pretrained=PRETRAINED)
        elif args.model == 'vgg16':
            self.model = vgg16_bn(pretrained=PRETRAINED)
        elif args.model == 'mobilenetv2':
            self.model = mobilenet_v2(pretrained=PRETRAINED)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.metrics_acc = torchmetrics.Accuracy()

    def initial_classifier(self):
        if self.args.model == 'resnet50':
            num_features = self.model.fc.weight.shape[1]
            num_classes = args.num_classes
            self.model.fc = nn.Identity()
            self.classifier = nn.Linear(num_features, num_classes)
        elif self.args.model == 'densenet121':
            num_features = self.model.classifier.weight.shape[1]
            num_classes = args.num_classes
            self.model.classifier = nn.Identity()
            self.classifier = nn.Linear(num_features, num_classes)
        elif self.args.model == 'vgg16':
            dropout = 0.5
            num_classes = args.num_classes
            self.model.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, num_classes),
            )
        elif self.args.model == 'mobilenetv2':
            dropout = 0.2
            num_classes = args.num_classes
            num_features = self.model.last_channel
            self.model.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_features, num_classes),
            )

    def forward(self, x):
        x = F.interpolate(x, size=224, mode='bilinear')
        with torch.no_grad():
            z = self.model(x)
        y = self.classifier(z)
        return y

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
    wandb_logger = WandbLogger(name=f"{args.run_name}-{args.model}-{args.dataset}", project=args.project,
                               entity=args.entity,
                               offline=args.offline)

    if args.dataset == 'cifar10':
        dataset_train, dataset_test = get_cifar10(data_path=args.dataset_path)
        args.num_classes = 10
    if args.dataset == 'cifar100':
        dataset_train, dataset_test = get_cifar100(data_path=args.dataset_path)
        args.num_classes = 100
    if args.dataset == 'stl10':
        dataset_train, dataset_test = get_cifar10(data_path=args.dataset_path)
        args.num_classes = 10

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = MSC.load_from_checkpoint(args.checkpoint_path, args=args)
    model.initial_classifier()
    trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                         callbacks=[checkpoint_callback, lr_monitor], precision=16,
                                         # gradient_clip_val=1.0,
                                         check_val_every_n_epoch=args.eval_every)

    trainer.fit(model, train_dataloader, val_dataloader)
