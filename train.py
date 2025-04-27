import os.path

from models.sdv1 import StableDiffision
from dataset.seg_dataset import BucketDataset
from dataset.misc import ShuffleBucketHook
from torch.utils.data import DataLoader
from torch_frame import LoggerHook, AccelerateTrainer, logger
from torch_frame.hooks import EvalHook
import yaml
import torch
import argparse
import numpy as np


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


def eval_map(model, batch):
    #####################
    # 2. 计算loss #
    #####################
    for k in batch:
        v = batch[k]
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(model.device)
    loss, metric = model(batch)
    metric["loss"] = [loss.cpu().item()]
    metric["miou"] = [metric["miou"]]
    return metric


def collection(data):
    key_list = list(data[0].keys())
    data_batch = {}
    for k in key_list:
        vs = [d[k] for d in data]
        data_batch[k] = torch.stack(vs, 0)
    return data_batch


class EvalLoraHook(EvalHook):
    def save_model(self):
        # 如果当前epoch指标没有更好, 则不保存模型
        if self.save_metric is not None:
            if not self.is_better(self.trainer.metric_storage[self.save_metric]):
                return
            self.cur_best = self.trainer.metric_storage[self.save_metric].avg
            logger.info(f"{self.save_metric} update to {round(self.cur_best, 4)}")
            self.trainer.model.latent_diffusion_model.save_attn_procs(
                self.trainer.ckpt_dir,
                weight_name=f"best_lora.safetensors")
            head_path = os.path.join(self.trainer.ckpt_dir, f"best_head.safetensors")
            torch.save(self.trainer.model.head.state_dict(), head_path)

        if self._max_to_keep is not None and self._max_to_keep >= 1:
            epoch = self.trainer.epoch  # ranged in [0, max_epochs - 1]
            self.trainer.model.latent_diffusion_model.save_attn_procs(
                self.trainer.ckpt_dir,
                weight_name=f"{epoch:02d}_lora.safetensors")
            head_path = os.path.join(self.trainer.ckpt_dir, f"{epoch:02d}_head.safetensors")
            torch.save(self.trainer.model.head.state_dict(), head_path)
            self._recent_checkpoints.append(f"{epoch:02d}")
            if len(self._recent_checkpoints) > self._max_to_keep:
                # delete the oldest checkpoint
                epoch_name = self._recent_checkpoints.pop(0)
                lora_path = os.path.join(self.trainer.ckpt_dir, f"{epoch_name}_lora.safetensors")
                if os.path.exists(lora_path):
                    os.remove(lora_path)
                head_path = os.path.join(self.trainer.ckpt_dir, f"{epoch_name}_head.safetensors")
                if os.path.exists(head_path):
                    os.remove(head_path)


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    config_model = config["model"]
    config_diffusion = config_model["diffusion"]
    config_lora = config_model.get("lora", None)
    config_train = config["train"]
    train_dataset = BucketDataset(
        config_diffusion["pretrained_model_name_or_path"],
        config_train["dataset_path"],
        config_train["train_batch_size"],
        dataset_type="train",
    )
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              persistent_workers=True,
                              batch_size=None,
                              num_workers=config_train["num_workers"],
                              collate_fn=collection
                              )

    valid_dataset = BucketDataset(
        config_diffusion["pretrained_model_name_or_path"],
        config_train["dataset_path"],
        config_train["train_batch_size"],
        dataset_type="valid",
    )
    valid_loader = DataLoader(valid_dataset,
                              shuffle=False,
                              persistent_workers=False,
                              batch_size=None,
                              num_workers=0,
                              collate_fn=collection
                              )

    model = StableDiffision(config_diffusion,
                            config_lora=config_lora)

    if config_train["use_8bit_adam"]:
        print("使用8bit")
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.trainable_params,
        lr=1e-04,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-08,
    )

    hooks = [
        ShuffleBucketHook(train_dataset),
        EvalLoraHook(valid_loader, eval_map, max_to_keep=2, save_metric="miou", max_first=True, prefix="valid"),
        LoggerHook(),
    ]
    scheduler = config_train["lr_scheduler"]

    trainer = AccelerateTrainer(model, optimizer, scheduler, train_loader, config_train["max_epoch"],
                                config_train["workspace"], config_train["max_grad_norm"],
                                mixed_precision=config_train["mixed_precision"], hooks=hooks,
                                gradient_accumulation_steps=config_train["gradient_accumulation_steps"],
                                )
    trainer.log_param(**config)
    trainer.train()


if __name__ == '__main__':
    main()
