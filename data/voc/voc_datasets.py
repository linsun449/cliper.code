import os.path
import re
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

PARENT_PATH = os.path.split(os.path.realpath(__file__))[0]

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255

VOC2012_CAT_NAME = open(os.path.join(PARENT_PATH, "cls_voc21.txt")).read().splitlines()
VOC_PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
               [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
               [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
               [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
               [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
               [0, 64, 128]]

VOC_CONTEXT_NAME = open(os.path.join(PARENT_PATH, "cls_context60.txt")).read().splitlines()
CONTEXT_PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
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
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]

def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]


def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    img_name_list = [int(img_name_list[i].replace("_", "")) for i in range(len(img_name_list))]

    return img_name_list


class VOC12SegmentationDataset(Dataset):

    def __init__(self, img_name_list_path, label_dir, voc_root, img_transform=None, label_transform=None):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc_root = voc_root
        self.label_dir = label_dir
        self.img_transform = img_transform
        self.label_transform = label_transform

        if self.voc_root.__contains__('VOC2012'):
            self.background = re.split(r"[\s,]+", VOC2012_CAT_NAME[0])
            self.categories = VOC2012_CAT_NAME[1:]
            self.category_number = len(self.categories)
            self.palette = np.array(VOC_PALETTE)
        else:
            self.background = re.split(r"[\s,]+", VOC_CONTEXT_NAME[0])
            self.categories = VOC_CONTEXT_NAME[1:]
            self.category_number = len(self.categories)
            self.palette = np.array(CONTEXT_PALETTE)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = Image.open(get_img_path(name_str, self.voc_root))
        mask = Image.open(os.path.join(self.label_dir, name_str + '.png'))

        ori_img = torch.tensor(np.array(img.convert('RGB'))).permute(2, 0, 1) / 255.
        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            mask = self.label_transform(mask)
        else:
            mask = torch.tensor(np.array(mask))

        label = F.one_hot(torch.where(mask == IGNORE, 0, mask.long()).unique(), self.category_number + 1).sum(0)[1:]

        return ori_img, img, mask, label, name_str
