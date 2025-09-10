import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as ttf
from PIL import Image


class CLIPer(nn.Module):
    def __init__(self, cfg, device):
        super(CLIPer, self).__init__()
        self.cfg = cfg
        self.device = device
        self.attn_type = cfg.attn_type
        self.fuse_feature = cfg.fuse_feature
        self.size = cfg.size
        self.select_layer = cfg.select_layer
        if cfg.model_name == "ViT-B/16":
            model_path = "/home/sunlin/.cache/clip/ViT-B-16.pt"
        elif cfg.model_name == "ViT-L/14":
            model_path = "/home/sunlin/.cache/clip//ViT-L-14.pt"
        else:
            NotImplementedError(f"Error: model name {cfg.model_name} not implemented")
        model, preprocess = clip.load(model_path, device=device)
        self.model = model.eval()

        self.preprocess = ttf.Compose([self._resize] + preprocess.transforms[2:])
        self.patch_size = int(cfg.model_name.split("/")[-1])
        self.logit_scale = nn.Parameter(torch.tensor(cfg.logit_scale), requires_grad=False)

        self.layers = model.visual.transformer.layers
        self.modify()

        self.img_h, self.img_w = None, None
        self.attn = None
        self.img_part_features = None
        self.image_feature = []

        if self.attn_type == "proxy":
            torch.hub.set_dir("/data/cache/dino")
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8').to(self.device)
            self.vfm_transform = ttf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def custom_attn(self, attn_layer, x, attn_mask=None):

        num_heads = attn_layer.num_heads
        _, bsz, embed_dim = x.size()
        head_dim = embed_dim // num_heads
        scale = head_dim ** -0.5

        q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        if self.attn_type == "fused-attn" and attn_mask is not None:
            # sum to 1
            attn_mask /= torch.sum(attn_mask, dim=-2, keepdim=True)
            attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)
            attn_mask = (attn_mask + attn_mask.transpose(-2, -1)) / 2
            attn_mask -= attn_mask.mean(-2, keepdim=True)
            attn_mask = torch.clamp(attn_mask, 0)
            attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)

            attn_mask = attn_mask.flatten(0, 1)
            attn_weights = torch.repeat_interleave(attn_mask, dim=0, repeats=v.shape[0] // attn_mask.shape[0])
        elif self.attn_type == "proxy":
            attn_mask = F.pad(self.self_attn, (1, 0, 1, 0), mode='constant', value=0)[None]
            attn_weights = torch.repeat_interleave(attn_mask, dim=0,
                                                   repeats=v.shape[0] // attn_mask.shape[0])
        elif self.attn_type == "q-q":
            attn_weights = torch.bmm(q * scale, q.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif self.attn_type == "k-k":
            attn_weights = torch.bmm(k * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif self.attn_type == "v-v":
            attn_weights = torch.bmm(v * scale, v.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        elif self.attn_type == "vanilla":
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
        else:
            identity = torch.eye(v.shape[-2], dtype=v.dtype, device=v.device)[None]
            attn_weights = torch.repeat_interleave(identity, dim=0, repeats=v.shape[0])

        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
        attn_output = attn_layer.out_proj(attn_output)

        return attn_output, attn_weights

    def forward_visual(self, x: torch.Tensor):
        model_visual = self.model.visual
        h, w = x.shape[-2], x.shape[-1]
        positional_embedding_new = self.upsample_pos_emb(model_visual.positional_embedding,
                                                         (h // self.patch_size, w // self.patch_size))
        x = model_visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([model_visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                              dtype=x.dtype, device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + positional_embedding_new.to(x.dtype)
        x = model_visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        return model_visual.transformer(x)

    def forward_transformer(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        model_transformer = self.model.visual.transformer
        model_visual = self.model.visual
        # 计算到倒数第二层
        attn_maps, img_features = 0, torch.zeros([self.layers] + list(x.shape), device=x.device, dtype=x.dtype)
        for i in range(self.layers - 1):
            ln_x = model_transformer.resblocks[i].ln_1(x)
            if self.fuse_feature or (self.select_layer + self.layers) % self.layers == i:
                img_features[i] = ln_x
            ln_x, attn_map = model_transformer.resblocks[i].attn(ln_x, ln_x, ln_x, need_weights=True,
                                                                 attn_mask=attn_mask, average_attn_weights=False)
            attn_maps += attn_map
            x = x + ln_x
            x = x + model_transformer.resblocks[i].mlp(model_transformer.resblocks[i].ln_2(x))

        # 计算最后一层
        model_res = model_transformer.resblocks[-1]
        img_features[-1] = x if self.fuse_feature or self.select_layer == -1 else 0
        for kth, x in enumerate(img_features):
            ln_x, attn = self.custom_attn(model_res.attn, model_res.ln_1(x), attn_mask=attn_maps)
            img_features[kth] = ln_x

        img_features = model_visual.ln_post(img_features.squeeze())
        if model_visual.proj is not None:
            img_features = img_features @ model_visual.proj

        return img_features, attn

    def modify(self):
        model_transformer = self.model.visual.transformer
        model_visual = self.model.visual

        model_transformer.forward = self.forward_transformer
        model_visual.forward = self.forward_visual

    def classify(self, x: torch.Tensor, text_emb: torch.Tensor):
        x = x / x.norm(dim=-1, keepdim=True)
        norm_text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_per_image = self.logit_scale * x @ norm_text_emb.to(x.dtype).t()

        soft_per_image = logit_per_image.softmax(dim=-1)
        return soft_per_image, logit_per_image

    def _resize(self, image):
        ori_width, ori_height = image.size
        ratio = self.size / min(ori_width, ori_height)
        ori_width, ori_height = ori_width * ratio, ori_height * ratio
        # ori_width, ori_height = 224, 224
        h, w = (int(ori_height / self.patch_size + 0.5) * self.patch_size,
                int(ori_width / self.patch_size + 0.5) * self.patch_size)
        resized_image = image.resize((w, h), Image.BICUBIC)
        return resized_image

    @staticmethod
    def upsample_pos_emb(emb, new_size):
        first, emb = emb[:1, :], emb[1:, :]
        n, d = emb.size(0), emb.size(1)
        size = int(np.sqrt(n))
        emb = emb.permute(1, 0).view(1, d, size, size)
        emb = F.interpolate(emb, size=new_size, mode='bilinear')
        emb = emb.view(d, -1).contiguous().permute(1, 0)
        emb = torch.cat([first, emb], 0)
        return emb.half()

    def classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).to(self.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights.t()

    def forward(self, img: torch.Tensor, fg_text_features: torch.Tensor, bg_text_features: torch.Tensor):
        self.img_h, self.img_w = img.shape[2] // self.patch_size, img.shape[3] // self.patch_size

        if self.attn_type == "proxy":
            patch_size = self.vfm.patch_embed.patch_size
            h, w = img.shape[2:]
            ori_img = F.interpolate(img,
                                    size=(h // patch_size * patch_size, w // patch_size * patch_size),
                                    mode="bilinear", align_corners=False)

            imgs_norm = self.vfm_transform(ori_img).to(self.device)
            fh, fw = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0].half()

            ex_feats = feat[:, 1:, :].reshape(feat.shape[0], fh, fw, -1).permute(0, 3, 1, 2)
            ex_feats = F.interpolate(ex_feats, size=(self.img_h, self.img_w), mode='bilinear', align_corners=False)
            q_k = F.normalize(ex_feats.flatten(2, 3), dim=1)
            similarity = torch.einsum("b c m, b c n -> b m n", q_k, q_k)

            similarity = (similarity - torch.mean(similarity))
            similarity[similarity < 0.0] = float('-inf')
            self.self_attn = F.softmax(similarity, dim=-1)[0]

        # 文本特征处理
        text_features = torch.cat([fg_text_features, bg_text_features, fg_text_features.mean(0, True)], dim=0)
        # 图像特征处理
        with torch.no_grad():
            img_feature, attn = self.model.encode_image(img)
            seg = self.classify(img_feature, text_features)[1][:, 1:]
            seg = seg.transpose(-1, -2).reshape(-1, len(text_features), self.img_h, self.img_w)
            seg = seg.softmax(-3)[:, :len(fg_text_features)]
            if self.cfg.sliding_crop_size > 0:
                seg_slide = self.forward_slide(img, text_features, crop_size=self.cfg.sliding_crop_size).to(seg.dtype)
                seg_slide = seg_slide.softmax(-3)[:, :len(fg_text_features)]
                seg = seg_slide * 0.5 + seg * 0.5

            seg_last = seg[self.select_layer]
            seg_last[seg_last < seg_last.amax((-1, -2), keepdim=True) * self.cfg.attention_thr] = 0
            seg_last = seg_last.flatten(-2, -1) @ attn.mean(0)[1:, 1:]
            seg_last = seg_last.unflatten(dim=-1, sizes=(self.img_h, self.img_w))
            seg = seg_last + (seg.mean(0) if self.fuse_feature else 0)

        return {"seg": seg.detach(), "img_part_features": img_feature.clone(),
                "mid_feature": None, "attn_map": attn.mean(0)[1:, 1:].clone()}

    def forward_slide(self, img: torch.Tensor, text_features: torch.Tensor, stride: int = 112,
                      crop_size: int = 224):  # 224

        _, _, h_img, w_img = img.shape
        h_grids = max(h_img - crop_size + stride - 1, 0) // stride + 1
        w_grids = max(w_img - crop_size + stride - 1, 0) // stride + 1
        preds = img.new_zeros((self.layers, len(text_features),
                               h_img // self.patch_size, w_img // self.patch_size)).to(img.dtype)
        attns = img.new_zeros((h_img // self.patch_size, w_img // self.patch_size,
                               (h_img // self.patch_size), (w_img // self.patch_size)))
        count_mat = img.new_zeros((1, h_img // self.patch_size, w_img // self.patch_size)).to(img.dtype)
        count_mat_attns = torch.zeros_like(attns)
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1, x1 = h_idx * stride, w_idx * stride
                y2, x2 = min(y1 + crop_size, h_img), min(x1 + crop_size, w_img)
                y1, x1 = max(y2 - crop_size, 0), max(x2 - crop_size, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                cts_h, cts_w = crop_img.shape[-2] // self.patch_size, crop_img.shape[-1] // self.patch_size
                cts_y1, cts_y2 = y1 // self.patch_size, y2 // self.patch_size
                cts_x1, cts_x2 = x1 // self.patch_size, x2 // self.patch_size
                with torch.no_grad():
                    img_feature, attn = self.model.encode_image(crop_img)
                    # img_feature = torch.einsum("mn, bnt->bmt", attn.mean(0)[1:, 1:], img_feature[:, 1:])
                    img_feature = img_feature[:, 1:]
                    seg = self.classify(img_feature, text_features)[1]
                    # seg = torch.einsum("mn, bnt->bmt", attn.mean(0)[1:, 1:], seg)# [:, 1:]  # [N, L, T]
                seg = seg.transpose(-1, -2).reshape(-1, len(text_features), cts_h, cts_w)
                attns[cts_y1: cts_y2, cts_x1: cts_x2,
                cts_y1: cts_y2, cts_x1: cts_x2] += attn.mean(0)[1:, 1:].reshape(cts_h, cts_w, cts_h, cts_w)
                count_mat_attns[cts_y1: cts_y2, cts_x1: cts_x2, cts_y1: cts_y2, cts_x1: cts_x2] += 1

                preds[..., cts_y1: cts_y2, cts_x1: cts_x2] += seg
                count_mat[..., cts_y1: cts_y2, cts_x1: cts_x2] += 1

        preds /= count_mat
        attns /= count_mat_attns + 1e-5
        attns = attns.flatten(0, 1).flatten(-2, -1)
        attns /= attns.sum(dim=-1, keepdim=True)
        preds = preds.flatten(-2, -1) @ attns
        preds = preds.unflatten(-1, sizes=(h_img // self.patch_size, w_img // self.patch_size))
        return preds
