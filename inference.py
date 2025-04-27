import os.path

import cv2
import numpy as np
from transformers import CLIPTokenizer
import torch
from torch import nn
from PIL import Image
import torch.nn.functional as F
from diffusers import (AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline)
from transformers import CLIPTextModel
from torchvision.transforms import v2
import argparse


class SDInference(nn.Module):
    def __init__(self, pretrained_model_name_or_path, lora_path, head_path):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.latent_diffusion_model = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", device_map="auto")
        # 加载 LoRA 配置文件
        self.latent_diffusion_model.load_attn_procs(lora_path)
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        for model in [self.text_encoder, self.latent_diffusion_model, self.vae]:
            model.requires_grad_(False)
            model.to(self.device, torch.float16)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.head = nn.Sequential(
            nn.Conv2d(4, 16, 5, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 5, 1, 1),
        )
        self.head.to(self.device, torch.float16)
        head_weight = torch.load(head_path)
        self.head.load_state_dict(head_weight)
        self.transform = v2.Compose(
            [
                v2.Resize(512),
                # v2.RandomHorizontalFlip(),
                v2.ToTensor(),
            ]
        )

    @torch.no_grad()
    def forward(self, image: Image.Image, prompt: str):
        ow, oh = image.size
        x = self.transform(image).unsqueeze(0).to(self.device)
        input_ids = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        bs, channel, height, width = x.shape
        latents = self.vae.encode(x.to(dtype=self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.to(dtype=self.latent_diffusion_model.dtype)
        timesteps = torch.full((bs,), 500, device=self.device)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(input_ids.to(self.device))[0]

        model_pred = self.latent_diffusion_model(latents, timesteps, encoder_hidden_states).sample
        logits = self.head(model_pred)
        pred = F.sigmoid(logits)
        # logits = model_pred
        pred = F.interpolate(pred, size=(oh, ow), mode='bicubic', align_corners=False)
        return pred


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/sdv1.yaml",
        required=False,
        help="Path to config",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    model = SDInference("D:/stable-diffusion-v1-5",
                        "workspace/log_2025-05-04 18_08_56/checkpoints/lora_20.safetensors",
                        "workspace/log_2025-05-04 18_08_56/checkpoints/head_20.safetensors")
    # image_pil = Image.open(r"D:\temp_data\seg_dataset\PascalContext\train\images\2008_000033.jpg")
    image_pil = Image.open("Design006_5tiBmKQ3Kb.jpg")
    pred = model(image_pil, "text")[0][0]
    # pred = pred > 0.5
    mask = np.clip(pred.to("cpu").numpy() * 255, 0, 255).astype(np.uint8)
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.tile(mask[..., None], (1, 1, 3))
    result = np.concatenate([image, mask], axis=1)
    cv2.imwrite("result.png", result)


