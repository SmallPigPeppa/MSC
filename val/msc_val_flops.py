import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import wandb
import torchmetrics
# from resnet50_l2_norm import ResNet50
from resnet50_l2_last import ResNet50
import torch.nn.functional as F
from tqdm import tqdm
from torchprofile import profile_macs
from pytorch_lightning.loggers import WandbLogger


def test_resolutions(model, dataset_path, resolutions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    accuracy = torchmetrics.Accuracy().to(device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=8, pin_memory=True)
    res_list = []
    acc1_list = []
    acc2_list = []
    acc3_list = []
    acc_best_list = []
    flops_list = []
    for res in resolutions:
        correct1 = 0
        correct2 = 0
        correct3 = 0
        total = 0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader)):
                inputs, targets = batch
                inputs = F.interpolate(inputs, size=int(res), mode='bilinear')
                # inputs = F.interpolate(inputs, size=int(224), mode='bilinear')
                inputs, targets = inputs.to(device), targets.to(device)
                z1, z2, z3, y_hat1, y_hat2, y_hat3 = model(inputs)
                acc1 = accuracy(y_hat1, targets)
                correct1 += acc1.item() * inputs.size(0)
                acc2 = accuracy(y_hat2, targets)
                correct2 += acc2.item() * inputs.size(0)
                acc3 = accuracy(y_hat3, targets)
                correct3 += acc3.item() * inputs.size(0)
                total += inputs.size(0)

                if idx == 0:
                    macs = profile_macs(model, inputs)
                    flops = macs / 1e9
                    # GFLOPs
                    flops_list.append(flops)

        mean_acc1 = correct1 / total * 100.
        mean_acc2 = correct2 / total * 100.
        mean_acc3 = correct3 / total * 100.

        acc1_list.append(mean_acc1)
        acc2_list.append(mean_acc2)
        acc3_list.append(mean_acc3)
        acc_best_list.append(max(mean_acc1, mean_acc2, mean_acc3))
        res_list.append(res)
        print(
            f"Resolution: {res}, Accuracy1: {mean_acc1}, Accuracy2: {mean_acc2}, Accuracy3: {mean_acc3}, FLOPs:{flops}")

    return res_list, acc1_list, acc2_list, acc3_list, acc_best_list, flops_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        default='checkpoints/resnet50-baseline-s-clip-multi/epoch=89-step=112680.ckpt',
                        help="Path to the trained model checkpoint")

    parser.add_argument("--run_name", type=str, default="resnet50-baseline", help="Name of the Weights & Biases run")
    parser.add_argument("--project", type=str, default="Multi-Scale-CNN", help="Name of the Weights & Biases project")
    parser.add_argument("--entity", type=str, default='pigpeppa',
                        help="Name of the Weights & Biases entity (team or user)")
    parser.add_argument("--offline", action="store_true", help="Run Weights & Biases logger in offline mode")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--max_epochs", type=int, default=90, help="Maximum number of training epochs")
    parser.add_argument("--dataset_path", type=str, default="/mnt/mmtech01/dataset/lzy/ILSVRC2012",
                        help="Path to the ImageNet dataset")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate the model every N epochs")
    parser.add_argument("--trunc", type=float, default=0.01, help="trunc for the si loss")

    args = parser.parse_args()

    wandb_logger = WandbLogger(name=args.run_name, project=args.project, entity=args.entity)
    # wandb_table = wandb.Table(columns=["Resolution", "Accuracy"])

    model = ResNet50.load_from_checkpoint(args.checkpoint_path, args=args)

    resolutions = list(range(32, 225, 16))
    res_list, acc1_list, acc2_list, acc3_list, acc_best_list, flops_list = test_resolutions(model, args.dataset_path,
                                                                                            resolutions)
    columns = ['size'] + [str(i) for i in res_list]
    acc_table = [['acc1'] + acc1_list, ['acc2'] + acc2_list, ['acc3'] + acc3_list,
                 ['acc_best'] + acc_best_list]

    # wandb.log({"Resolution": res_list, "Accuracy1": acc1_list, "Accuracy2": acc2_list, "Accuracy3": acc3_list,
    #            "FLOPs": flops_list})
    wandb_logger.log_table(key="acc", columns=columns, data=acc_table)
    wandb.finish()
