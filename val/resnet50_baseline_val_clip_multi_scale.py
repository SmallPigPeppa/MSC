import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import wandb
import torchmetrics
from resnet50_baseline_train_clip_multi import ResNet50
import torch.nn.functional as F
from tqdm import tqdm


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
    dataset = torchvision.datasets.ImageFolder(os.path.join(dataset_path, ""), transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=8, pin_memory=True)
    res_list = []
    acc_list = []
    for res in resolutions:
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                inputs, targets = batch
                # inputs = F.interpolate(inputs, size=int(res), mode='bilinear')
                # inputs = F.interpolate(inputs, size=int(224), mode='bilinear')
                inputs, targets = inputs.to(device), targets.to(device)
                _,_,outputs = model(inputs)
                acc = accuracy(outputs, targets)
                correct += acc.item() * inputs.size(0)
                total += inputs.size(0)

        mean_acc = correct / total
        res_list.append(res)
        acc_list.append(mean_acc)
        print(f"Resolution: {res}, Accuracy: {mean_acc}")

    return res_list, acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
                        default='checkpoints/resnet50-baseline-s-clip-multi/epoch=89-step=112680.ckpt',
                        help="Path to the trained model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="/mnt/mmtech01/dataset/lzy/ILSVRC2012",
                        help="Path to the ImageNet dataset")
    parser.add_argument("--project", type=str, default="Multi-scale-CNN val",
                        help="Name of the Weights & Biases project")
    parser.add_argument("--entity", type=str, default="pigpeppa",
                        help="Name of the Weights & Biases entity (team or user)")
    parser.add_argument("--run_name", type=str, default="resnet50-debug", help="Name of the Weights & Biases run")

    args = parser.parse_args()

    wandb.init(name=args.run_name, project=args.project, entity=args.entity)
    wandb_table = wandb.Table(columns=["Resolution", "Accuracy"])

    model = ResNet50.load_from_checkpoint(args.checkpoint_path,max_epochs=90, learning_rate=0.5, batch_size=128, weight_decay=5e-4, dataset_path='no')

    resolutions = list(range(224, 225, 16))
    res_list, acc_list = test_resolutions(model, args.dataset_path, resolutions)

    wandb.log({"Resolution": res_list, "Accuracy": acc_list})
    wandb.finish()
