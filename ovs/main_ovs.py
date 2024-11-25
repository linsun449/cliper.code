import logging
import os.path
import time

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.load import load_dataset
from ovs.pipeline import Pipeline
from util.args import parse_args
from util.imutils import save_heatmap
from util.miou import ShowSegmentResult, cam_to_label

cfg = parse_args()
print(f"config:\n{cfg}")
os.makedirs(cfg.log_path, exist_ok=True)
logging.basicConfig(filename=f"{cfg.log_path}/{cfg.model_name[4]}-{cfg.dataset_name}-{cfg.refinement}-{cfg.ignore_labels}.txt",
                    level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info(f"{cfg}")

cfg.semantic_templates = [line.strip() for line in list(open(cfg.semantic_templates))]

if hasattr(cfg, "save_path"):
    os.makedirs(cfg.save_path, exist_ok=True)

pipe = Pipeline(cfg)

dataset, bg_text_features, fg_text_features = load_dataset(cfg, pipe.cliper)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

util_iou = ShowSegmentResult(num_classes=dataset.category_number + 1, ignore_labels=cfg.ignore_labels)

consumption_time = 0
for i, (ori_img, img, mask, label, name) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing image"):
    ori_height, ori_width = mask.shape[1], mask.shape[2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img, fg_text_features, bg_text_features = img.to(device), fg_text_features.to(device), bg_text_features.to(device)

    torch.cuda.synchronize()
    start_time = time.time()
    pred_mask, final_score = pipe(ori_img, img, fg_text_features, bg_text_features)

    pred_mask = F.interpolate(pred_mask[None], size=(ori_height, ori_width), mode='bilinear')[0]
    pred_mask = pred_mask / (pred_mask.amax((-1, -2), keepdim=True) + 1e-5) * final_score[..., None, None]
    _, arg_mask = cam_to_label(pred_mask[None].clone(), cls_label=final_score[None],
                               bkg_thre=cfg.bkg_thre, cls_thre=cfg.score_threshold, is_normalize=cfg.is_normalize)

    torch.cuda.synchronize()
    end_time = time.time()
    consumption_time += end_time - start_time
    saved_pred = pred_mask / pred_mask.amax(dim=(-1, -2), keepdim=True) * 255
    if hasattr(cfg, "save_path"):
        for l in label[0].nonzero()[:, 0]:
            save_heatmap(saved_pred[l].cpu().numpy(), f"{cfg.save_path}/{name[0]}-{l.item()}.png")
        cv2.imwrite(f"{cfg.save_path}/{name[0]}.png", dataset.palette[arg_mask[0].cpu().numpy()])
        saved_mask = mask.clone()
        saved_mask[saved_mask == 255] = 0
        cv2.imwrite(f"{cfg.save_path}/{name[0]}_gt.png", dataset.palette[saved_mask[0].cpu().numpy()])
    util_iou.add_prediction(mask[0].cpu().numpy(), arg_mask[0].cpu().numpy())

logging.info(f"consumption time:{consumption_time}")
iou = util_iou.calculate()
logging.info(f"mIou:{iou['mIoU']}")
print(f"mIou:{iou['mIoU']}")
