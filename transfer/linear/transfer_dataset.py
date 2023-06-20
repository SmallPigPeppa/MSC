from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Subset
import torch


def split_dataset(dataset, num_test=30):
    # 获取所有类别的标签
    targets = np.array(dataset.y)
    classes, class_counts = np.unique(targets, return_counts=True)

    train_indices, test_indices = [], []

    for c in classes:
        indices = np.where(targets == c)[0]
        np.random.shuffle(indices)  # 随机打乱
        test_indices.extend(indices[:num_test])
        train_indices.extend(indices[num_test:])

    dataset_train = Subset(dataset, train_indices)
    dataset_test = Subset(dataset, test_indices)

    return dataset_train, dataset_test


def get_statistics(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader.dataset)
    std /= len(loader.dataset)

    return mean, std


def get_cifar10(data_path):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_cifar100(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_stl10(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.STL10(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.STL10(root=data_path, split='test', download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_caltech101(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset = datasets.Caltech101(root=data_path, target_type='category', download=True, transform=None)  # 先不进行预处理
    dataset_train, dataset_test = split_dataset(dataset)  # 划分训练集和测试集
    dataset_train.dataset.transform = transform_train  # 对训练集进行预处理
    dataset_test.dataset.transform = transform_test  # 对测试集进行预处理

    return dataset_train, dataset_test


def to_3_channels(x):
    return torch.stack([x] * 3)


def get_fashion_mnist(data_path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(to_3_channels),
                                    transforms.Normalize((0.2860, 0.2860, 0.2860), (0.3205, 0.3205, 0.3205))])
    dataset_train = datasets.FashionMNIST(data_path, download=True, train=True, transform=transform)
    dataset_test = datasets.FashionMNIST(data_path, download=True, train=False, transform=transform)
    # mean_train, std_train = get_statistics(dataset_train)
    # mean_test, std_test = get_statistics(dataset_test)
    #
    # print(f'Train dataset: mean={mean_train}, std={std_train}')
    # print(f'Test dataset: mean={mean_test}, std={std_test}')

    return dataset_train, dataset_test


def get_flowers(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5123, 0.4131, 0.3406), (0.2523, 0.2079, 0.2196)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.5123, 0.4131, 0.3406), (0.2523, 0.2079, 0.2196)),
        ]
    )
    dataset_train = datasets.Flowers102(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.Flowers102(root=data_path, split='test', download=True, transform=transform_test)

    # mean_train, std_train = get_statistics(dataset_train)
    # mean_test, std_test = get_statistics(dataset_test)
    #
    # print(f'Train dataset: mean={mean_train}, std={std_train}')
    # print(f'Test dataset: mean={mean_test}, std={std_test}')

    return dataset_train, dataset_test

def get_pets(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.5123, 0.4131, 0.3406), (0.2523, 0.2079, 0.2196)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            # transforms.Normalize((0.5123, 0.4131, 0.3406), (0.2523, 0.2079, 0.2196)),
        ]
    )
    dataset_train = datasets.OxfordIIITPet(root=data_path, split='trainval', download=True, target_types='category',
                                           transform=transform_train)
    dataset_test = datasets.OxfordIIITPet(root=data_path, split='test', download=True, target_types='category',
                                          transform=transform_test)
    mean_train, std_train = get_statistics(dataset_train)
    mean_test, std_test = get_statistics(dataset_test)

    print(f'Train dataset: mean={mean_train}, std={std_train}')
    print(f'Test dataset: mean={mean_test}, std={std_test}')
    return dataset_train, dataset_test


def get_cars(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.StanfordCars(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.StanfordCars(root=data_path, split='test', download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_aircraft(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.FGVCAircraft(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.FGVCAircraft(root=data_path, split='test', download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_sun397(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.SUN397(root=data_path, download=True, transform=transform_train)
    dataset_test = datasets.SUN397(root=data_path, download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_pcam(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.PCAM(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.PCAM(root=data_path, split='train', download=True, transform=transform_train)
    return dataset_train, dataset_test


def get_pcam(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.PCAM(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.PCAM(root=data_path, split='train', download=True, transform=transform_test)
    return dataset_train, dataset_test


def get_dtd(data_path):
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=96, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        ]
    )
    dataset_train = datasets.DTD(root=data_path, split='train', download=True, transform=transform_train)
    dataset_test = datasets.DTD(root=data_path, split='train', download=True, transform=transform_test)
    return dataset_train, dataset_test







if __name__ == '__main__':
    # a, b = get_caltech101(data_path='/Users/lwz/torch_ds')
    # c, d = len(a), len(b)
    a, b = get_stl10(data_path='/Users/lwz/torch_ds')
    c, d = len(a), len(b)
    # # a, b = get_cars(data_path='/Users/lwz/torch_ds')
    # a, b = get_aircraft(data_path='/Users/lwz/torch_ds')
    c = 0

"""
Caltech101：这个数据集包含101个类别的图像，每个类别大约有40到800张图像。图像的大小不同，但大部分的图像的分辨率都低于224x224。

Caltech256：这个数据集与Caltech101类似，但包含256个类别。同样，图像的大小各不相同，但大部分图像的分辨率都低于224x224。

CelebA：这是一个大规模的人脸属性数据集，包含202,599张人脸图像，每张图像有40个属性注释。虽然原始图像的大小为218x178，小于224x224，但在预处理时通常会将其裁剪和调整为64x64或更小的大小。

Country211：这是一个由OpenAI提供的数据集，包含211个国家的图像。这个数据集的详细信息我不太确定，因为这个数据集在我知识库的更新日期（2021年9月）之后发布。

DTD：Describable Textures Dataset (DTD) 是一个纹理描述数据集，包含了5640个图像，这些图像被分为47个类别。每个图像的大小是120x120，小于224x224。

EuroSAT：这是一个卫星图像数据集，包含10个类别，每个类别有3000张图像。图像的分辨率为64x64。

FGVCAircraft：这个数据集包含了100个类别的飞机图像。虽然每个图像的大小不同，但许多图像的分辨率都低于224x224。

GTSRB：German Traffic Sign Recognition Benchmark (GTSRB) 数据集，这个数据集包含了约39000张训练图像和约12000张测试图像，这些图像都是交通标志。图像的尺寸不一，但大多数都小于224x224。

PCAM：这是一个病理图像分类的数据集，包含了327680张图像，每张图像的分辨率是96x96。

RenderedSST2：这个数据集的具体信息我不太确定，因为这个数据集在我知识库的更新日期（2021年9月）之后发布。

SEMEION：这是一个手写数字识别的数据集，包含了1593个32x32的二值图像。

StanfordCars：这个数据集包含8144张训练图像和8041张测试图像，共有196个类别的汽车图像。每张图像的大小都不同，但许多图像的分辨率都低于224x224。

SUN397：这是一个场景识别的数据集，包含397个类别，每个类别有100张图像。图像的尺寸不一，但大多数都小于224x224。
"""

#
# def get_Caltech101():
#     pass
#
#
# def get_Caltech256():
#     pass
#
#
# def get_CelebA():
#     pass
#
#
# def get_EuroSAT():
#     pass
#
#
# def get_FGVCAircraft():
#     pass
#
#
# def get_PCAM():
#     pass
#
#
# def get_StanfordCars():
#     pass
#
#
# def get_SUN397():
#     pass
