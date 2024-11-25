import os

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as ttf

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import open_clip


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class OpenCLIPer(nn.Module):
    def __init__(self, model_name="ViT-H/14", attn_type="fused-attn", fuse_feature=True, size=336,
                 logit_scale=100, device="cuda"):
        super(OpenCLIPer, self).__init__()

        self.device = device
        self.attn_type = attn_type
        self.fuse_feature = fuse_feature
        self.size = size
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale), requires_grad=False)
        self.patch_size = int(model_name.split("/")[-1])

        if model_name == "ViT-H/14":
            model_name = 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, device=device,
                                                                               precision='fp16')
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.preprocess = ttf.Compose([self._resize] + self.preprocess.transforms[2:])

        self.layers = self.model.visual.transformer.layers
        self.modify()

        self.img_h, self.img_w = None, None

    def modify(self):

        model_transformer = self.model.visual.transformer
        model_visual = self.model.visual

        def custom_attn(attn_layer, x, attn_mask=None):
            x = x.transpose(0, 1)
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
                attn_mask -= attn_mask.amax(-2, keepdim=True) * 0.2
                attn_mask = torch.clamp(attn_mask, 0)
                attn_mask /= torch.sum(attn_mask, dim=-1, keepdim=True)

                attn_mask = attn_mask.flatten(0, 1)
                attn_weights = torch.repeat_interleave(attn_mask, dim=0, repeats=v.shape[0] // attn_mask.shape[0])
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
            attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim).transpose(0, 1)
            attn_output = attn_layer.out_proj(attn_output)

            return attn_output, attn_weights

        def forward(x: torch.Tensor):
            x = model_visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            x = torch.cat([_expand_token(model_visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
            x = x + self.upsample_pos_emb(model_visual.positional_embedding, (self.img_h, self.img_w))

            x = model_visual.patch_dropout(x)
            x = model_visual.ln_pre(x)
            x = model_visual.transformer(x)
            return x

        def forward_transformer(x: torch.Tensor, attn_mask: torch.Tensor = None):
            # 计算到倒数第二层
            attn_maps, img_features = 0, torch.zeros([self.layers] + list(x.shape), device=x.device, dtype=x.dtype)
            for i, res in enumerate(model_transformer.resblocks[:-1]):
                ln_x = res.ln_1(x)
                if self.fuse_feature:
                    img_features[i] = ln_x
                ln_x, attn_map = res.attn(ln_x, ln_x, ln_x, need_weights=True,
                                          attn_mask=attn_mask, average_attn_weights=False)
                attn_maps += attn_map
                x = x + res.ls_1(ln_x)
                x = x + res.ls_2(res.mlp(res.ln_2(x)))

            # 计算最后一层
            model_res = model_transformer.resblocks[-1]
            img_features[-1] = x
            for kth, x in enumerate(img_features):
                ln_x, attn = custom_attn(model_res.attn, model_res.ln_1(x), attn_mask=attn_maps)
                img_features[kth] = ln_x - img_features[kth]

            img_features = model_visual.ln_post(img_features.squeeze())
            if model_visual.proj is not None:
                img_features = img_features @ model_visual.proj

            return img_features, attn

        model_visual.forward, model_transformer.forward = forward, forward_transformer

    def _resize(self, image):
        ori_width, ori_height = image.size
        ratio = self.size / min(ori_width, ori_height)
        ori_width, ori_height = ori_width * ratio, ori_height * ratio
        h, w = (int(ori_height / self.patch_size + 0.5) * self.patch_size,
                int(ori_width / self.patch_size + 0.5) * self.patch_size)
        resized_image = image.resize((w, h), Image.BICUBIC)
        return resized_image

    def classify(self, x: torch.Tensor, text_emb: torch.Tensor):
        x = x / x.norm(dim=-1, keepdim=True)
        norm_text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_per_image = self.logit_scale * x @ norm_text_emb.to(x.dtype).t()

        soft_per_image = logit_per_image.softmax(dim=-1)
        return soft_per_image, logit_per_image

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
                texts = self.tokenizer(texts).to(self.device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights.t()

    def forward(self, img: torch.Tensor, fg_text_features: torch.Tensor, bg_text_features: torch.Tensor):
        self.img_h, self.img_w = img.shape[2] // self.patch_size, img.shape[3] // self.patch_size

        img = img.to(self.device).half()
        fg_text_features = fg_text_features.to(self.device).half()
        bg_text_features = bg_text_features.to(self.device).half()

        # 文本特征处理
        text_features = torch.cat([fg_text_features, bg_text_features, fg_text_features.mean(0, True)], dim=0)
        # 图像特征处理
        with torch.no_grad():
            img_feature, attn = self.model.encode_image(img)
            seg = self.classify(img_feature, text_features)[0][:, 1:, :len(fg_text_features)]
            seg_last = seg[-1]
            seg_last[seg_last < seg_last.amax(0, keepdim=True) * 0.1] = 0
            seg = attn.mean(0)[1:, 1:] @ seg_last + seg[:-1].mean(0) * 0.2
            seg /= torch.clamp(seg.amax(dim=0, keepdim=True), min=1)

        return {"seg": seg.detach(), "img_part_features": img_feature.clone(),
                "mid_feature": None, "attn_map": attn.mean(0)[1:, 1:].clone()}
