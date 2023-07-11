import torch
from torchprofile import profile_macs
import torchvision


def test_resolutions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(pretrained=False)
    model = model.to(device)
    model.eval()
    inputs = torch.rand(256, 3, 32, 32).to(device)
    macs = profile_macs(model, inputs)
    flops = macs / 1e9
    print(f"FLOPs:{flops}")

    return 0


if __name__ == "__main__":
    test_resolutions()
