import math
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

NUM_WORKERS = 16


def get_imnet1k_dataloader(root, batch_size=128, augmentation=False, val_only=False):
    if True:
        dataset_val, _ = build_dataset(root, is_train=False, input_size=224, augmentation=False)

        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size,
            shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, drop_last=False
        )

        return None, val_loader
    
    else:
        dataset_train, nb_classes = build_dataset(root, is_train=True, input_size=224, augmentation=augmentation)
        dataset_val, _ = build_dataset(root, is_train=False, input_size=224, augmentation=False)

        train_loader = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=batch_size,
            shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, drop_last=False
        )

        return train_loader, val_loader


def build_dataset(root, is_train, input_size=224, augmentation=False):
    transform = build_transform(is_train, input_size, augmentation)

    root = os.path.join(root)
    dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = 1000

    return dataset, nb_classes


def build_transform(is_train, input_size=224, augmentation=False):
    resize_im = input_size > 32
    # if False:
    #     if augmentation == "noaug":
    #         transform = transforms.Compose([
    #             transforms.RandomResizedCrop(input_size, scale=(0.08, 1.0)),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
    #         return transform
    #     else:
    #         # this should always dispatch to transforms_imagenet_train
    #         transform = create_transform(
    #             input_size=input_size,
    #             is_training=True,
    #             color_jitter=0.4,
    #             auto_augment=augmentation,
    #             interpolation='bicubic',
    #             re_prob=0.25,
    #             re_mode='pixel',
    #             re_count=1,
    #         )
    #         if not resize_im:
    #             # replace RandomResizedCropAndInterpolation with
    #             # RandomCrop
    #             transform.transforms[0] = transforms.RandomCrop(
    #                 input_size, padding=4)
    #         return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)