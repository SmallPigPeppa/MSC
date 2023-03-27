import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import wandb
import torchmetrics
from resnet50_baseline_train import ResNet50
import torch.nn.functional as F
def test_resolutions(model, dataset_path, resolutions, wandb_table):
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)
    for res in resolutions:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                inputs = F.interpolate(inputs, size=int(res), mode='bilinear')
                inputs = F.interpolate(inputs, size=int(224), mode='bilinear')
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                acc = accuracy(outputs, targets)
                correct += acc.item() * inputs.size(0)
                total += inputs.size(0)

        mean_acc = correct / total
        wandb_table.add_data(res, mean_acc)
        print(f"Resolution: {res}, Accuracy: {mean_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,default='checkpoints/resnet50-baseline/epoch=89-step=112680.ckpt', help="Path to the trained model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="/mnt/mmtech01/dataset/lzy/ILSVRC2012", help="Path to the ImageNet dataset")
    parser.add_argument("--project", type=str, default="Multi-scale-CNN val", help="Name of the Weights & Biases project")
    parser.add_argument("--entity", type=str, default="pigpeppa", help="Name of the Weights & Biases entity (team or user)")
    parser.add_argument("--run_name", type=str, default="resnet50", help="Name of the Weights & Biases run")

    args = parser.parse_args()

    wandb.init(name=args.run_name,project=args.project, entity=args.entity)
    wandb_table = wandb.Table(columns=["Resolution", "Accuracy"])

    model = ResNet50.load_from_checkpoint(args.checkpoint_path)

    resolutions = list(range(32, 224, 16))
    test_resolutions(model, args.dataset_path, resolutions, wandb_table)

    wandb.log({"Accuracy per Resolution": wandb_table})
    wandb.finish()