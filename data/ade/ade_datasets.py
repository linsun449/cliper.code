import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

PARENT_PATH = os.path.split(os.path.realpath(__file__))[0]
ADE_CAT_NAME = open(os.path.join(PARENT_PATH, "cls_ade20k.txt")).read().splitlines()
ADE_PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]]

def load_img_list(dataset_path):
    img_gt_name_list = open(dataset_path).readlines()
    img_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]
    name_list = []
    for img in img_list:
        dat = img.split(" ")
        name_list.append(dat[0])
    return name_list


class ADESegDataset(Dataset):
    def __init__(self, img_name_list_path, ade_root, used_dir="validation", img_transform=None, label_transform=None):
        self.name_list = load_img_list(img_name_list_path)
        self.label_root = os.path.join(ade_root, 'annotations', used_dir)
        self.ade_root = os.path.join(ade_root, "images", used_dir)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.categories = ADE_CAT_NAME[1:]
        self.category_number = len(self.categories)
        self.background = re.split(r"[\s,]+", ADE_CAT_NAME[0])
        self.palette = np.array(ADE_PALETTE)

    def __getitem__(self, idx):
        name = self.name_list[idx]

        img = Image.open(os.path.join(self.ade_root, f"{name}.jpg"))
        mask = Image.open(os.path.join(self.label_root, f"{name}.png"))

        ori_img = torch.tensor(np.array(img.convert('RGB'))).permute(2, 0, 1) / 255.

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)
        else:
            mask = torch.tensor(np.array(mask))

        label = F.one_hot(torch.where(mask == 255, 0, mask.long()).unique(), self.category_number + 1).sum(0)[1:]

        return ori_img, img, mask, label, name

    def __len__(self):
        return len(self.name_list)
