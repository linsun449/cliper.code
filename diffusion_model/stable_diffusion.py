# make sure you're logged in with `huggingface-cli login`
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from diffusers.utils.torch_utils import randn_tensor

from diffusion_model.Processor import AttnProcessorForCallBack, DIFFUSION_LAYERS

from torch import autocast, nn
from diffusers import StableDiffusionPipeline


class diffusion(nn.Module):

    def __init__(self,
                 attention_layers_to_use=None,
                 model="v2.1",
                 time_step=45,
                 dtype=torch.float16,
                 device='cuda:0'):
        super().__init__()
        # stabilityai/stable-diffusion-2-1-base runwayml/stable-diffusion-v1-5 CompVis/stable-diffusion-v1-4
        if model == "v2.1":
            model = "stabilityai/stable-diffusion-2-1-base"
        elif model == "v1.5":
            model = "runwayml/stable-diffusion-v1-5"
        elif model == "v1.4":
            model = "CompVis/stable-diffusion-v1-4"
        else:
            raise ValueError(f"Not supported model {model}")
        self.model = StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype)
        self.setup(device)
        self.dtype = dtype
        self.time_step = time_step
        # 获取注意力图
        self.attention_maps = {}
        if attention_layers_to_use is None:
            attention_layers_to_use = [-1]
        self.layers = attention_layers_to_use
        for layer_idx in attention_layers_to_use:
            attn = eval(f"self.model.unet.{DIFFUSION_LAYERS[layer_idx]}")
            attn.processor = AttnProcessorForCallBack(self, layer_idx)

    def one_step(self, latents, prompts):

        self.model._guidance_scale = 1
        self.model._clip_skip = None
        self.model._joint_attention_kwargs = None
        self.model._interrupt = False

        self.model.scheduler.set_timesteps(50, device=self.device)
        t = self.model.scheduler.timesteps[self.time_step]

        noise = randn_tensor(latents.shape, device=latents.device, dtype=latents.dtype)
        # get latents
        latents = self.model.scheduler.add_noise(latents, noise, t)

        prompt_embeds, _ = self.model.encode_prompt(
            prompts, self.device, 1, do_classifier_free_guidance=False,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=self.model.clip_skip,
        )

        noise_pred = self.model.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

    def generate_image(self, prompts):
        with autocast("cuda"):
            image = self.model(prompts)["images"][0]
        return image

    @property
    def device(self):
        return self.model._execution_device

    def setup(self, device):
        self.model.to(device)

        for param in self.model.vae.parameters():
            param.requires_grad = False
        for param in self.model.unet.parameters():
            param.requires_grad = False
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, img, prompts=""):
        latent = self.model.image_processor.preprocess(img, height=512, width=512).to(self.dtype)
        latent = self.model.vae.encode(latent)[0].mean * self.model.vae.config.scaling_factor
        self.one_step(latent, prompts=prompts)


if __name__ == "__main__":
    iseg = diffusion(attention_layers_to_use=[-2])
    prompt = "two dogs running under the sea. "
    iseg.one_step(torch.randn((1, 4, 64, 64), dtype=torch.float16, device='cuda'), prompts='')
    img = iseg.generate_image(prompt)
    print(iseg.attention_maps[-2].shape)
