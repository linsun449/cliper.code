import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

PARENT_PATH = os.path.split(os.path.realpath(__file__))[0]
COCO_CAT_NAME = open(os.path.join(PARENT_PATH, "cls_coco81.txt")).read().splitlines()
COCO_STUFF_CAT_NAME = open(os.path.join(PARENT_PATH, "cls_coco_stuff172.txt")).read().splitlines()
OBJECT_PALETTE = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
                 [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
                 [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192],
                 [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
                 [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
                 [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
                 [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
                 [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
                 [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128],
                 [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
                 [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96],
                 [64, 160, 0],
                 [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]]
STUFF_PALETTE = palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                 [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                 [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                 [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                 [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                 [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                 [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                 [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                 [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                 [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                 [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                 [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                 [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                 [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                 [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                 [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                 [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                 [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                 [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                 [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                 [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                 [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                 [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                 [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                 [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                 [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                 [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                 [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                 [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                 [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                 [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                 [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                 [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                 [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                 [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                 [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                 [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                 [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                 [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                 [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                 [64, 192, 96], [64, 160, 64], [64, 64, 0]]


def load_img_list(dataset_path):
    img_gt_name_list = open(dataset_path).readlines()
    img_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]
    name_list = []
    for img in img_list:
        dat = img.split(" ")
        name_list.append(dat[0])
    return name_list


class COCOSegDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, used_dir="train2014", img_transform=None, label_transform=None):
        self.name_list = load_img_list(img_name_list_path)
        self.label_root = os.path.join(coco_root, 'coco_seg_anno')
        self.coco_root = os.path.join(coco_root, used_dir)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.categories = COCO_CAT_NAME[1:]
        self.category_number = len(self.categories)
        self.background = re.split(r"[\s,]+", COCO_CAT_NAME[0])
        self.palette = np.array(OBJECT_PALETTE)

    def __getitem__(self, idx):
        name = self.name_list[idx]

        img = Image.open(os.path.join(self.coco_root, f"{name}.jpg"))
        mask = Image.open(os.path.join(self.label_root, f"{name[-12:]}.png"))

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


class COCOStuffSegDataset(COCOSegDataset):
    def __init__(self, img_name_list_path, coco_root, used_dir="train2014", img_transform=None, label_transform=None):
        super().__init__(img_name_list_path, coco_root, used_dir, img_transform, label_transform)
        self.label_root = os.path.join(coco_root, 'stuff_anno164', used_dir)
        self.categories = COCO_STUFF_CAT_NAME[1:]
        self.category_number = len(self.categories)
        self.background = re.split(r"[\s,]+", COCO_STUFF_CAT_NAME[0])
        self.palette = np.array(STUFF_PALETTE)

    def __getitem__(self, idx):
        ori_img, img, mask, label, name = super().__getitem__(idx)
        cand = mask.unique()
        cand = cand[cand != 255]
        label = F.one_hot(cand.long(), self.category_number).sum(0)
        mask[mask != 255] = mask[mask != 255] + 1 # add background but not use to keep with dataset containing background
        return ori_img, img, mask, label, name
