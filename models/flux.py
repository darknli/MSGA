import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
from diffusers import AutoencoderKL, FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast, T5EncoderModel
from accelerate import Accelerator
from peft import LoraConfig
from .utils import str2torch_dtype, cast_training_params, quantization, flush_vram
from .sdv1 import StableDiffision
from einops import rearrange, repeat


class Flux(StableDiffision):

    def __init__(self,
                 config_diffusion: dict,
                 config_vae: dict = None,
                 config_lora: dict = None,
                 ):
        super().__init__(config_diffusion, config_vae, config_lora)

    def init_diffusion(self, config):
        pretrained_model_name_or_path = config["pretrained_model_name_or_path"]
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder")
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2")
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_2")
        self.latent_diffusion_model = FluxTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="transformer",
        )

        if "unet_dtype" not in config:
            config["unet_dtype"] = "fp16"
        self.latent_diffusion_model.to(self.device, str2torch_dtype(config["unet_dtype"], default=self.weight_dtype))

        if "text_encoder_dtype" not in config:
            config["text_encoder_dtype"] = config["unet_dtype"]
        self.text_encoder_2.to(self.device, str2torch_dtype(config["text_encoder_dtype"], default=self.weight_dtype))
        self.text_encoder.to(self.device, str2torch_dtype(config["text_encoder_dtype"], default=self.weight_dtype))

        if "quantization" not in config:
            config["quantization"] = False
        elif config["quantization"]:
            print("使用量化!")

        if config["quantization"]:
            quantization(self.latent_diffusion_model)

        # freeze unet
        if "train_unet" not in config:
            config["train_unet"] = False
        if not config["train_unet"]:
            print("freeze unet")
            self.latent_diffusion_model.requires_grad_(False)
        else:
            if config.get("enable_gradient_checkpointing", False):
                # self.latent_diffusion_model.enable_gradient_checkpointing()
                # 这里跟其他模型不一样
                self.latent_diffusion_model.gradient_checkpointing = True
            self.trainable_params = cast_training_params(self.latent_diffusion_model)

        if "train_text_encoder" not in config:
            config["train_text_encoder"] = False

        if config["quantization"]:
            quantization(self.text_encoder_2)

        if not config["train_text_encoder"]:
            print("freeze text_encoder")
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            self.text_encoder.eval()
            self.text_encoder_2.eval()
        else:
            for text_encoder in [self.text_encoder, self.text_encoder_2]:
                if hasattr(text_encoder, 'enable_gradient_checkpointing'):
                    text_encoder.enable_gradient_checkpointing()
                if hasattr(text_encoder, "gradient_checkpointing_enable"):
                    text_encoder.gradient_checkpointing_enable()
            self.trainable_params.extend(cast_training_params(self.text_encoder))
            self.trainable_params.extend(cast_training_params(self.text_encoder_2))
        flush_vram()

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

    def init_head(self):
        self.head = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 5, 1, 1),
        )
        self.head.to(self.device, self.weight_dtype)
        self.trainable_params.extend(cast_training_params(self.head))

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
            batch["prompt"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids

        input_ids2 = self.tokenizer_2(
            batch["prompt"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        ).input_ids
        encoder_hidden_states = self.text_encoder(input_ids.to(self.device), output_hidden_states=False)
        pooled_prompt_embeds = encoder_hidden_states.pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device)
        encoder_hidden_states = self.text_encoder_2(input_ids2.to(self.device), output_hidden_states=False)[0]

        model_pred = self.run_diffusion_model(latents, timesteps, encoder_hidden_states, pooled_prompt_embeds)
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
            self, noisy_latents, timesteps, encoder_hidden_states, pooled_encoder_hidden_states
    ):
        # Predict the noise residual and compute loss
        noisy_latents = noisy_latents.to(self.device, dtype=self.latent_diffusion_model.dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype=self.latent_diffusion_model.dtype)
        pooled_encoder_hidden_states = pooled_encoder_hidden_states.to(dtype=self.latent_diffusion_model.dtype)
        guidance = torch.tensor([1.]).to(device=self.device, dtype=self.latent_diffusion_model.dtype)
        timesteps = timesteps.to(self.device)

        bs, c, h, w = noisy_latents.shape
        latent_model_input_packed = rearrange(
            noisy_latents,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=2,
            pw=2
        )
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        # 新版的img_ids和txt_ids不需要batch维，如有问题可升级最新版本
        img_ids = repeat(img_ids, "h w c -> (h w) c").to(
            self.device, dtype=self.latent_diffusion_model.dtype)
        txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(
            self.device, dtype=self.latent_diffusion_model.dtype)

        model_pred = self.latent_diffusion_model(
            hidden_states=latent_model_input_packed,  # [1, 4096, 64]
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
            # todo make sure this doesnt change
            timestep=timesteps / 1000,  # timestep is 1000 scale
            encoder_hidden_states=encoder_hidden_states,
            # [1, 512, 4096]
            pooled_projections=pooled_encoder_hidden_states,  # [1, 768]
            txt_ids=txt_ids,  # [512, 3]
            img_ids=img_ids,  # [4096, 3]
            guidance=guidance,
            return_dict=False,
        )[0]
        noise_pred = rearrange(
            model_pred,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=noisy_latents.shape[2] // 2,
            w=noisy_latents.shape[3] // 2,
            ph=2,
            pw=2,
            c=noisy_latents.shape[1],
        )
        return noise_pred

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
