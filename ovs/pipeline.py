from functools import reduce

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as ttf

from diffusion_model.stable_diffusion import diffusion
from modified_clip.model import CLIPer


class Pipeline(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cliper, self.attn_refine, self.vfm = None, None, None
        self.device = device
        if cfg.model_name in ["ViT-B/16", "ViT-L/14"]:
            self.cliper = CLIPer(cfg=cfg, device=self.device)
        else:
            raise NotImplementedError("Unknown Model")

        if hasattr(cfg, "refinement") and cfg.refinement in ["SFSA", "mean", "selection"]:
            self.attn_refine = diffusion(attention_layers_to_use=cfg.attention_layers_to_use,
                                         model=cfg.sd_version, time_step=cfg.time_step,
                                         device=self.device, dtype=torch.float16)
        elif hasattr(cfg, "refinement") and cfg.refinement in ["dino-b8"]:
            torch.hub.set_dir("/data/cache/dino")
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').to(self.device)
            self.vfm_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif hasattr(cfg, "refinement") and cfg.refinement in ["dino-s8"]:
            torch.hub.set_dir("/data/cache/dino")
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').to(self.device)
            self.vfm_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif hasattr(cfg, "refinement") and cfg.refinement in ["dino-s16"]:
            torch.hub.set_dir("/data/cache/dino")
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(self.device)
            self.vfm_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif hasattr(cfg, "refinement") and cfg.refinement in ["dino-b16"]:
            torch.hub.set_dir("/data/cache/dino")
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').to(self.device)
            self.vfm_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif hasattr(cfg, "refinement") and cfg.refinement in ["dinov2-s14"]:
            torch.hub.set_dir("/data/cache/dino")
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
            self.vfm_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif hasattr(cfg, "refinement") and cfg.refinement in ["dinov2-b14"]:
            torch.hub.set_dir("/data/cache/dino")
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device)
            self.vfm_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif hasattr(cfg, "refinement") and cfg.refinement in ["dinov2-l14"]:
            torch.hub.set_dir("/data/cache/dino")
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(self.device)
            self.vfm_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        elif hasattr(cfg, "refinement") and cfg.refinement in ["dinov2-g14"]:
            torch.hub.set_dir("/data/cache/dino")
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(self.device)
            self.vfm_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.cfg = cfg 

    @torch.no_grad()
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
            self_att = torch.where(self_att < self.cfg.attention_thr, 0, self_att)
            self_att /= self_att.sum(dim=-1, keepdim=True) + 1e-5
            if self.cfg.refinement == "mean":
                self_att = self_att.mean(0)
            elif self.cfg.refinement == "selection":
                self_att = self_att[self.cfg.attention_idx]
            else:
                self_att = reduce(torch.matmul, self_att, torch.eye(self_att.shape[-1], device=self_att.device))
            pred_mask = (self_att @ cross_att).transpose(0, 1).reshape(-1, 64, 64)
        elif self.vfm is not None:

            patch_size = self.vfm.patch_embed.patch_size
            if not isinstance(patch_size, int):
                patch_size = patch_size[0]

            h, w = ori_img.shape[2:]
            ori_img = F.interpolate(ori_img,
                                    size=(h // patch_size * patch_size, w // patch_size * patch_size),
                                    # size=(512, 512),
                                    mode="bilinear", align_corners=False)

            imgs_norm = self.vfm_transform(ori_img).to(self.device)
            fh, fw = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
            if "v2" not in self.cfg.refinement:
                feat = feat[:, 1:, :]
            ex_feats = feat.reshape(feat.shape[0], fh, fw, -1).permute(0, 3, 1, 2)
            q_k = F.normalize(ex_feats.flatten(2, 3), dim=1)
            similarity = torch.einsum("b c m, b c n -> b m n", q_k, q_k)

            similarity = (similarity - 1.2 * torch.mean(similarity)) * 3
            similarity[similarity < 0.0] = float('-inf')
            self_attn = F.softmax(similarity, dim=-1)[0]
            cross_att = F.interpolate(pred_mask[None], size=(fh, fw), mode='bilinear',
                                      align_corners=False)[0].flatten(-2).float().transpose(0, 1)
            pred_mask = (self_attn @ cross_att).transpose(0, 1).reshape(-1, fh, fw)
        return pred_mask

    def forward(self, ori_img, img, classify_fg_text_features, classify_bg_text_features):
        segment_results = self.cliper(img, classify_fg_text_features, classify_bg_text_features)
        seg = segment_results["seg"]
        final_score = seg.amax(dim=(-1, -2))
        pred_mask = self.refinement(ori_img, seg)
        final_score = pred_mask.amax(dim=(-1, -2)) * 0.5 + final_score * 0.5
        torch.cuda.empty_cache()
        return pred_mask, final_score
