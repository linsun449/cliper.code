import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes

PARENT_PATH = os.path.split(os.path.realpath(__file__))[0]
CITY_CAT_NAME = open(os.path.join(PARENT_PATH, "cls_city.txt")).read().splitlines()
CITY_PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                [107, 142, 35], [152, 251, 152], [70, 130, 180],
                [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]


def load_img_list(dataset_path):
    img_gt_name_list = open(dataset_path).readlines()
    img_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]
    name_list = []
    for img in img_list:
        dat = img.split(" ")
        name_list.append(dat[0])
    return name_list


class CitySegDataset(Dataset):
    def __init__(self, img_name_list_path, city_root, used_dir="val", img_transform=None, label_transform=None):
        self.name_list = load_img_list(img_name_list_path)
        self.label_root = os.path.join(city_root, 'gtFine', used_dir)
        self.city_root = os.path.join(city_root, "leftImg8bit", used_dir)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.categories = CITY_CAT_NAME[1:]
        self.category_number = len(self.categories)
        self.background = re.split(r"[\s,]+", CITY_CAT_NAME[0])
        self.palette = np.array(CITY_PALETTE)

    def __getitem__(self, idx):
        name = self.name_list[idx // 2]

        img = Image.open(os.path.join(self.city_root, f"{name}_leftImg8bit.png"))
        mask = Image.open(os.path.join(self.label_root, f"{name}_gtFine_labelTrainIds.png"))

        ori_img = torch.tensor(np.array(img.convert('RGB'))).permute(2, 0, 1) / 255.

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)
        else:
            mask = torch.tensor(np.array(mask))
        mh, _ = mask.shape[-2], mask.shape[-1]
        ih, _ = img.shape[-2], img.shape[-1]
        if idx % 2 == 0:
            mask = mask[:, mh:]
            ori_img = ori_img[..., mh:]
            img = img[..., ih:]
        else:
            mask = mask[:, :mh]
            ori_img = ori_img[..., :mh]
            img = img[..., :ih]
        mask[mask != 255] = mask[mask != 255] + 1
        label = F.one_hot(torch.where(mask == 255, 0, mask.long()).unique(), self.category_number + 1).sum(0)[1:]

        return ori_img, img, mask, label, name

    def __len__(self):
        return len(self.name_list) * 2


if __name__ == "__main__":
    import os
    path = '/data/datasets/CityScapes/leftImg8bit/val'
    des = 'city_val.txt'
    folders = os.listdir(path)
    with open(des, 'w') as f:
        for folder in folders:
            files = os.listdir(f'{path}/{folder}')
            for file in files:
                f.write(f"{folder}/{file.split('_leftImg8bit')[0]}\n")
