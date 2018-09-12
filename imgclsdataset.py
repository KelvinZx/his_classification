import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
import PIL
from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np
import torch.nn as nn

class GeneralDataset(nn.Module):
    def __init__(self, data_path, shape=None):
        super(GeneralDataset, self).__init__()
        self.type_datapath = data_path

        self.type_data = ImageFolder(root=self.type_datapath, transform=self.transform_img())
        if shape is not None:
            self.shape = shape

    def __len__(self):
        a = 1
        return a

    def __getitem__(self, item):
        return self.type_data

    def __call__(self):
        return self.type_data

    def transform_img(self):
        TRANSFORM_IMG = transforms.Compose([
            ImageTransform(),
            lambda x:PIL.Image.fromarray(x),
            #transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
             #                    std=[0.229, 0.224, 0.225])
        ])
        return TRANSFORM_IMG


class ImageTransform:
    """
    Use imgaug library to do image augmentation.
    """
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)