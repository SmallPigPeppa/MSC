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
from torch.utils.data import DataLoader
from transfer_dataset import *
# from mstrain.resnet50 import ResNet50_L2
# from mstrain.densenet121 import DenseNet121_L2
# from mstrain.vgg16_bn import VGG16_L2
# from mstrain.mobilenetv2 import MobileNetV2_L3
from resnet50_l2_last import ResNet50_L2
from densenet121_l2_seprate_trans import DenseNet121_L2
from vgg16_l3 import VGG16_L2
from mobilenetv2_l3 import MobileNetV2_L3

PRETRAINED = False


class MSC(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model == 'resnet50':
            self.model = ResNet50_L2()
        elif args.model == 'densenet121':
            self.model = DenseNet121_L2()
        elif args.model == 'vgg16':
            self.model = VGG16_L2()
        elif args.model == 'mobilenetv2':
            self.model = MobileNetV2_L3()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.metrics_acc = torchmetrics.Accuracy()

    def initial_classifier(self):
        if self.args.model == 'resnet50':
            num_features = self.model.unified_net.fc.weight.shape[1]
            num_classes = args.num_classes
            self.model.unified_net.fc = nn.Identity()
            self.classifier = nn.Linear(num_features, num_classes)
        elif self.args.model == 'densenet121':
            num_features = self.model.unified_net.classifier.weight.shape[1]
            num_classes = args.num_classes
            self.model.unified_net.classifier = nn.Identity()
            self.classifier = nn.Linear(num_features, num_classes)
        elif self.args.model == 'vgg16':
            dropout = 0.5
            num_features = 512 * 7 * 7
            num_classes = args.num_classes
            self.model.unified_net.classifier = nn.Identity()
            self.classifier = nn.Linear(num_features, num_classes)
            # self.classifier = nn.Sequential(
            #     nn.Linear(512 * 7 * 7, 4096),
            #     nn.ReLU(True),
            #     nn.Dropout(p=dropout),
            #     nn.Linear(4096, 4096),
            #     nn.ReLU(True),
            #     nn.Dropout(p=dropout),
            #     nn.Linear(4096, num_classes),
            # )
        elif self.args.model == 'mobilenetv2':
            dropout = 0.2
            num_classes = args.num_classes
            num_features = self.model.unified_net.last_channel
            self.model.unified_net.classifier = nn.Identity()
            self.classifier = nn.Linear(num_features, num_classes)
            # self.classifier = nn.Sequential(
            #     nn.Dropout(p=dropout),
            #     nn.Linear(num_features, num_classes),
            # )

    def forward(self, x):
        with torch.no_grad():
            if args.imagesize == 224:
                print('############################# 224 #############################')
                z = self.model.forward_224(x)
            elif args.imagesize in [128, 96, 64]:
                print('############################# 128 #############################')
                z = self.model.forward_128(x)
            elif args.imagesize in [32, 28]:
                print('############################# 32 #############################')
                z = self.model.forward_32(x)
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
        args.imagesize = 32
    if args.dataset == 'cifar100':
        dataset_train, dataset_test = get_cifar100(data_path=args.dataset_path)
        args.num_classes = 100
        args.imagesize = 32
    if args.dataset == 'stl10':
        dataset_train, dataset_test = get_stl10(data_path=args.dataset_path)
        args.num_classes = 10
        args.imagesize = 96
    if args.dataset == 'caltech':
        dataset_train, dataset_test = get_caltech101(data_path=args.dataset_path)
        args.num_classes = 101
        args.imagesize = 224
    if args.dataset == 'fashion':
        dataset_train, dataset_test = get_fashion_mnist(data_path=args.dataset_path)
        args.num_classes = 10
        args.imagesize = 28
    if args.dataset == 'flowers':
        dataset_train, dataset_test = get_flowers(data_path=args.dataset_path)
        args.num_classes = 102
        args.imagesize = 224
    if args.dataset == 'pets':
        dataset_train, dataset_test = get_pets(data_path=args.dataset_path)
        args.num_classes = 37
        args.imagesize = 224
    if args.dataset == 'cars':
        dataset_train, dataset_test = get_cars(data_path=args.dataset_path)
        args.num_classes = 196
        args.imagesize = 224
    if args.dataset == 'aircraft':
        dataset_train, dataset_test = get_aircraft(data_path=args.dataset_path)
        args.num_classes = 102
        args.imagesize = 64
    if args.dataset == 'rafdb':
        dataset_train, dataset_test = get_rafdb(data_path=args.dataset_path)
        args.num_classes = 7
        args.imagesize = 32
    if args.dataset=='dtd':
        dataset_train, dataset_test = get_dtd(data_path=args.dataset_path)
        args.num_classes = 47
        args.imagesize = 224
    if args.dataset=='sun397':
        dataset_train, dataset_test = get_sun397(data_path=args.dataset_path)
        args.num_classes = 397
        args.imagesize = 224

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = MSC.load_from_checkpoint(args.checkpoint_path, args=args)
    model.initial_classifier()
    # if args.model == 'vgg16':
    trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
                                         callbacks=[checkpoint_callback, lr_monitor], precision=16,
                                         gradient_clip_val=1.0,
                                         check_val_every_n_epoch=args.eval_every)
    # else:
    #     trainer = Trainer.from_argparse_args(args, gpus=args.num_gpus, accelerator="ddp", logger=wandb_logger,
    #                                          callbacks=[checkpoint_callback, lr_monitor], precision=16,
    #                                          # gradient_clip_val=1.0,
    #                                      check_val_every_n_epoch=args.eval_every)

    trainer.fit(model, train_dataloader, val_dataloader)
