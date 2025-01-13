import math
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

NUM_WORKERS = 16


def get_imnet1k_dataloader(root, batch_size=128, resize_big=False):
    if True:
        dataset_val, _ = build_dataset(root, is_train=False, input_size=224, resize_big=resize_big)

        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size,
            shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, drop_last=False
        )

        return None, val_loader


def build_dataset(root, is_train, input_size=224, resize_big=False):
    transform = build_transform(is_train, input_size, resize_big)

    root = os.path.join(root)
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, input_size=224, resize_big=False):
    resize_im = input_size > 32

    t = []
    if resize_im:
        if resize_big:
            t.append(transforms.Resize(int((256 / 224) * input_size)))
            t.append(transforms.CenterCrop(input_size))
        else:
            t.append(transforms.Resize(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)