import copy
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
from diffusers import (AutoencoderKL, UNet2DConditionModel)
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from peft import LoraConfig
from .utils import str2torch_dtype, cast_training_params


class StableDiffision(torch.nn.Module):

    def __init__(self,
                 config_diffusion: dict,
                 config_vae: dict = None,
                 config_lora: dict = None,
                 prediction_type: str = None,
                 noise_offset: float = None,
                 ):
        super().__init__()
        self.latent_diffusion_model = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.lora = None
        if torch.cuda.device_count() == 1:
            self.device = "cuda"
            self.weight_dtype = torch.float16
        else:
            accelerator = Accelerator()
            self.device = copy.deepcopy(accelerator.device)
            self.weight_dtype = str2torch_dtype(accelerator.mixed_precision, torch.float16)
            del accelerator
        self.cache = {}
        self.head = nn.Sequential(
            nn.Conv2d(4, 16, 5, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 5, 1, 1),
        )
        self.head.to(self.device, torch.float16)
        self.trainable_params = cast_training_params(self.head)

        # 保存配置
        self.config_diffusion = copy.deepcopy(config_diffusion)
        self.config_vae = copy.deepcopy(config_vae)
        self.config_lora = copy.deepcopy(config_lora)
        self.prediction_type = prediction_type
        self.noise_offset = noise_offset

        self.init_diffusion(self.config_diffusion)
        self.init_vae(self.config_vae)
        if self.config_lora:
            self.init_lora(self.config_lora)
        assert len(self.trainable_params) > 0, "No trainable parameters"

    def init_diffusion(self, config):
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.latent_diffusion_model = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", device_map="auto")

        if "unet_dtype" not in config:
            config["unet_dtype"] = "fp16"
        self.latent_diffusion_model.to(self.device, str2torch_dtype(config["unet_dtype"], default=self.weight_dtype))

        if "text_encoder_dtype" not in config:
            config["text_encoder_dtype"] = config["unet_dtype"]
        self.text_encoder.to(self.device, str2torch_dtype(config["text_encoder_dtype"], default=self.weight_dtype))

        # freeze unet
        if "train_unet" not in config:
            config["train_unet"] = False
        if not config["train_unet"]:
            print("freeze unet")
            self.latent_diffusion_model.requires_grad_(False)
        else:
            if config.get("enable_gradient_checkpointing", False):
                self.latent_diffusion_model.enable_gradient_checkpointing()
            self.trainable_params.extend(cast_training_params(self.latent_diffusion_model))

        self.text_encoder.requires_grad_(False)

    def init_vae(self, config):
        if config is None:
            config = {}
        if "pretrained_vae_name_or_path" in config:
            # 单独指定vae预训练模型
            pretrained_vae_name_or_path = config["pretrained_vae_name_or_path"]
        else:
            pretrained_vae_name_or_path = self.config_diffusion["pretrained_model_name_or_path"]
            config["pretrained_vae_name_or_path"] = pretrained_vae_name_or_path
            self.config_vae = config
        self.vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path, subfolder="vae")
        if "vae_dtype" not in config:
            config["vae_dtype"] = None
        self.vae.to(self.device, str2torch_dtype(config["vae_dtype"], default=self.weight_dtype))
        self.vae.requires_grad_(False)

    def init_lora(self, config):
        if "train_unet" in self.config_diffusion and self.config_diffusion["train_unet"]:
            raise ValueError("不要既训练unet又训练lora!")
        if config.get("enable_gradient_checkpointing", False):
            self.latent_diffusion_model.enable_gradient_checkpointing()
        rank = config["rank"]
        unet_lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.latent_diffusion_model.add_adapter(unet_lora_config)
        unet_lora_parameters = cast_training_params(self.latent_diffusion_model)
        # self.latent_diffusion_model.enable_gradient_checkpointing()
        self.lora = unet_lora_parameters
        self.trainable_params.extend(unet_lora_parameters)

    def forward(self, batch):
        latents = self.run_vae(batch["image"])
        mask = batch["mask"]
        bs, channel, height, width = mask.shape
        timesteps = torch.full((bs,), 500, device=self.device)
        # Get the text embedding for conditioning
        input_ids = self.tokenizer(
            batch["prompt"], max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        encoder_hidden_states = self.text_encoder(input_ids.to(self.device))[0]

        model_pred = self.run_diffusion_model(latents, timesteps, encoder_hidden_states)
        logits = self.head(model_pred)
        logits = F.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False)
        loss = self.run_loss(logits, mask)
        miou = self.run_miou(F.sigmoid(logits), mask)
        return loss, {"miou": miou}

    def run_vae(self, pixel_values):
        latents = self.vae.encode(pixel_values.to(dtype=self.vae.dtype)).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.to(dtype=self.latent_diffusion_model.dtype)
        return latents

    def run_diffusion_model(
            self, noisy_latents, timesteps, encoder_hidden_states
    ):
        # Predict the noise residual and compute loss
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.latent_diffusion_model.dtype)

        model_pred = self.latent_diffusion_model(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
        ).sample
        return model_pred

    def run_loss(self, model_pred, target):
        loss = binary_cross_entropy_with_logits(model_pred.float(), target.float())
        return loss

    @torch.no_grad()
    def run_miou(self, model_pred, target):
        pred = model_pred > 0.5
        gt = target > 0.5
        inter = torch.logical_and(pred, gt).int().sum([1, 2, 3])
        union = torch.logical_or(pred, gt).int().sum([1, 2, 3])
        miou = (inter.float() / (union.float() + 1e-3)).mean()
        return miou.cpu().item()
