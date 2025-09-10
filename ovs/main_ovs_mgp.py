import os

import clip
import open_clip
import pytorch_lightning as pl
import torchvision.transforms as ttf
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler

from data.ade.ade_datasets import ADESegDataset
from data.cityscapes.city_datasets import CitySegDataset
from data.coco14.coco_datasets import COCOSegDataset, COCOStuffSegDataset
from data.voc.voc_datasets import VOC12SegmentationDataset
from ovs.pipeline import Pipeline

from data.load import load_dataset
from util.miou import ShowSegmentResult, cam_to_label


class CLIPerModule(pl.LightningModule):
    def __init__(self, cfg):
        super(CLIPerModule, self).__init__()
        self.fg_text_features = None
        self.bg_text_features = None
        self.util_iou = None
        self.cfg = cfg
        self.pipe = None

    def on_test_start(self):
        self.pipe = Pipeline(self.cfg, self.device)
        self.util_iou = ShowSegmentResult(num_classes=self.cfg.category_number + 1,
                                          ignore_labels=self.cfg.ignore_labels)

        self.bg_text_features = self.pipe.cliper.classifier(self.dataset.background, self.cfg.semantic_templates)
        self.fg_text_features = self.pipe.cliper.classifier(self.dataset.categories, self.cfg.semantic_templates)

    def test_step(self, batch, batch_idx):
        ori_img, img, mask, label, name = batch
        bg_text_features, fg_text_features = self.bg_text_features, self.fg_text_features
        ori_height, ori_width = mask.shape[1], mask.shape[2]
        img, fg_text_features, bg_text_features = img.to(self.device), fg_text_features.to(
            self.device), bg_text_features.to(self.device)

        pred_mask, final_score = self.pipe(ori_img, img, fg_text_features, bg_text_features)

        pred_mask = F.interpolate(pred_mask[None], size=(ori_height, ori_width), mode='bilinear')[0]
        pred_mask = pred_mask / (pred_mask.amax((-1, -2), keepdim=True) + 1e-5) * final_score[..., None, None]
        _, arg_mask = cam_to_label(pred_mask[None].clone(), cls_label=final_score[None],
                                   bkg_thre=self.cfg.bkg_thre, cls_thre=self.cfg.score_threshold,
                                   is_normalize=self.cfg.is_normalize)

        self.util_iou.add_prediction(mask[0].cpu().numpy(), arg_mask[0].cpu().numpy())
        return torch.tensor(0.0)  # 返回值会被用于计算梯度等，这里我们不需要它

    def on_test_end(self):
        self.synchronize_hist()
        iou = self.util_iou.calculate()
        if not self.trainer.is_global_zero: return
        print(f"mIoU: {iou['mIoU']}")
        file = (f"{self.cfg.log_path}/{self.cfg.model_name[4]}-{self.cfg.dataset_name}-"
                f"{self.cfg.refinement}-{self.cfg.ignore_labels}.txt")
        with open(file, "a+") as f:
            f.write(str(self.cfg))
            f.write(f"mIoU: {iou['mIoU']}\n\n")

    def synchronize_hist(self):
        """同步所有设备的混淆矩阵"""
        # 获取当前设备上的hist并转换为tensor
        hist = self.util_iou.hist
        tensor_hist = torch.tensor(hist, dtype=torch.float32).cuda()

        # 使用all_reduce同步所有设备上的hist
        dist.all_reduce(tensor_hist, op=dist.ReduceOp.SUM)

        # 将合并后的结果重新存储在hist中
        self.util_iou.hist = tensor_hist.cpu().numpy()


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset):
        super(DataModule, self).__init__()
        self.dataset = dataset

    def setup(self, stage=None):
        pass

    def test_dataloader(self):
        sampler = DistributedSampler(self.dataset, num_replicas=torch.distributed.get_world_size(),
                                     rank=torch.distributed.get_rank(), shuffle=False)
        return DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=1,
                          pin_memory=False, sampler=sampler, persistent_workers=False)


def classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]  # format with class
            texts = clip.tokenize(texts).to(model.device)  # tokenize
            class_embeddings = model.model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(model.device)
    return zeroshot_weights.t()


def load_dataset(cfg, preprocess):
    if cfg.dataset_name == "voc21":
        dataset = VOC12SegmentationDataset(img_name_list_path=cfg.img_name_list_path,
                                           label_dir=cfg.label_dir,
                                           voc_root=cfg.voc12_root,
                                           img_transform=preprocess)

    elif cfg.dataset_name == "context":
        dataset = VOC12SegmentationDataset(img_name_list_path=cfg.img_name_list_path,
                                           label_dir=cfg.label_dir,
                                           voc_root=cfg.voc12_root,
                                           img_transform=preprocess)
    elif cfg.dataset_name == "coco5k":
        dataset = COCOSegDataset(img_name_list_path=cfg.img_name_list_path,
                                 coco_root=cfg.coco_root,
                                 used_dir=cfg.used_dir,
                                 img_transform=preprocess)
    elif cfg.dataset_name == "coco_stuff164":
        dataset = COCOStuffSegDataset(img_name_list_path=cfg.img_name_list_path,
                                      coco_root=cfg.coco_root,
                                      used_dir=cfg.used_dir,
                                      img_transform=preprocess)
    elif cfg.dataset_name == "ade150":
        dataset = ADESegDataset(img_name_list_path=cfg.img_name_list_path,
                                ade_root=cfg.ade_root,
                                used_dir=cfg.used_dir,
                                img_transform=preprocess)
    elif cfg.dataset_name == "cityscapes":
        dataset = CitySegDataset(img_name_list_path=cfg.img_name_list_path,
                                 city_root=cfg.city_root,
                                 used_dir=cfg.used_dir,
                                 img_transform=preprocess)
    else:
        raise NotImplementedError("Unknown dataset")
    return dataset


def main():
    from util.args import parse_args
    cfg = parse_args()
    cfg.local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    cfg.semantic_templates = [line.strip() for line in list(open(cfg.semantic_templates))]
    cliper = CLIPerModule(cfg)

    if cfg.model_name in ["ViT-B/16", "ViT-L/14"]:
        if cfg.model_name == "ViT-B/16":
            model_path = "/home/sunlin/.cache/clip/ViT-B-16.pt"
        elif cfg.model_name == "ViT-L/14":
            model_path = "/home/sunlin/.cache/clip/ViT-L-14.pt"
        _, preprocess = clip.load(model_path, device='cpu')
    elif cfg.model_name in ["ViT-H/14"]:
        # model_name = 'hf-hub:laion/CLIP--laion2B-s32B-b79K'
        model_name = "ViT-H-14"
        _, _, preprocess = open_clip.create_model_and_transforms(model_name,
                                                                 pretrained="/data/cache/open_clip_model.safetensors",
                                                                 device="cpu",
                                                                 pretrained_hf=False,
                                                                 cache_dir="/home/sunlin/.cache/clip",
                                                                 precision='fp16')
    patch_size = int(cfg.model_name.split("/")[-1])

    def _resize(image):
        ori_width, ori_height = image.size
        ratio = cfg.size / min(ori_width, ori_height)
        ori_width, ori_height = ori_width * ratio, ori_height * ratio
        # ori_width, ori_height = 224, 224
        h, w = (int(ori_height / patch_size + 0.5) * patch_size,
                int(ori_width / patch_size + 0.5) * patch_size)
        resized_image = image.resize((w, h), Image.BICUBIC)
        return resized_image

    preprocess = ttf.Compose([_resize] + preprocess.transforms[2:])
    dataset = load_dataset(cfg, preprocess)
    cliper.cfg.category_number = dataset.category_number
    cliper.dataset = dataset
    datamodule = DataModule(dataset)
    trainer = pl.Trainer(
        accelerator="gpu",
        default_root_dir="outputs",
        devices=-1,
        log_every_n_steps=1,
    )

    # 训练/验证模型
    trainer.test(cliper, datamodule)


if __name__ == "__main__":
    import torch

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.distributed.init_process_group('nccl')
    main()
