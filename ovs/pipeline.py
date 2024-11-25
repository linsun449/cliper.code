from functools import reduce

import torch
import torch.nn.functional as F
from torch import nn

from diffusion_model.stable_diffusion import diffusion
from modified_clip.model import CLIPer
from modified_clip.open_model import OpenCLIPer


class Pipeline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cliper, self.attn_refine = None, None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cfg.model_name in ["ViT-B/16", "ViT-L/14"]:
            self.cliper = CLIPer(model_name=cfg.model_name, logit_scale=cfg.logit_scale, attn_type=cfg.attn_type,
                                 fuse_feature=cfg.fuse_feature, size=cfg.size, device=self.device)
        elif cfg.model_name == "ViT-H/14":
            self.cliper = OpenCLIPer(model_name=cfg.model_name, logit_scale=cfg.logit_scale, attn_type=cfg.attn_type,
                                     fuse_feature=cfg.fuse_feature, size=cfg.size, device=self.device)
        else:
            raise NotImplementedError("Unknown Model")

        if hasattr(cfg, "refinement") and cfg.refinement in ["SFSA", "mean", "selection"]:
            self.attn_refine = diffusion(attention_layers_to_use=cfg.attention_layers_to_use,
                                         model=cfg.sd_version, time_step=cfg.time_step,
                                         device=self.device, dtype=torch.float16)

        self.cfg = cfg

    def refinement(self, ori_img, pred_mask):
        # refinement pipeline
        if self.attn_refine is not None:
            pred_mask = F.interpolate(pred_mask[None], size=(64, 64), mode='bilinear',
                                      align_corners=False)[0].flatten(-2).float()
            cross_att = pred_mask.transpose(0, 1)

            self.attn_refine(ori_img.to(self.device), "")
            self_att = torch.cat(
                [self.attn_refine.attention_maps[idx][0] for idx in self.cfg.attention_layers_to_use]).float()
            self_att /= torch.amax(self_att, dim=-2, keepdim=True) + 1e-5
            self_att = torch.where(self_att < 0.1, 0, self_att)
            self_att /= self_att.sum(dim=-1, keepdim=True) + 1e-5
            if self.cfg.refinement == "mean":
                self_att = self_att.mean(0)
            elif self.cfg.refinement == "selection":
                self_att = self_att[self.cfg.attention_idx]
            else:
                self_att = reduce(torch.matmul, self_att, torch.eye(self_att.shape[-1], device=self_att.device))
            pred_mask = (self_att @ cross_att).transpose(0, 1).reshape(-1, 64, 64)
        return pred_mask

    def forward(self, ori_img, img, classify_fg_text_features, classify_bg_text_features):
        segment_results = self.cliper(img, classify_fg_text_features, classify_bg_text_features)
        seg = segment_results["seg"]
        final_score = seg.amax(dim=0)
        seg = seg.transpose(0, 1).reshape(-1, self.cliper.img_h, self.cliper.img_w)
        pred_mask = self.refinement(ori_img, seg)
        final_score = pred_mask.amax(dim=(-1, -2)) * 0.5 + final_score * 0.5
        return pred_mask, final_score
