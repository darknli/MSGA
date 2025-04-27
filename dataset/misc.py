from torch_frame.hooks import HookBase
import cv2
import torch
import numpy as np


class ShuffleBucketHook(HookBase):
    """Bucket dataset专用hook"""
    def __init__(self, dataset):
        self.dataset = dataset

    def before_epoch(self):
        # 打乱顺序
        self.dataset.build_batch_indices()
        torch.cuda.empty_cache()


def combine_image_and_mask(image, mask, alpha=0.5):
    """
    把图片和语义分割的结果进行合成
    :param image: 原始图片
    :param mask: 语义分割的结果
    :param alpha: 掩码的透明度
    :return: 合成后的图片
    """
    # 确保掩码和图片尺寸一致
    if len(mask.shape) == 3 and mask.shape[2] != 1:
        mask = mask[..., :1]
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask[mask > 128] = 255
    mask[mask <= 128] = 0
    # 生成随机颜色
    color = np.random.randint(0, 256, 3, dtype=np.uint8)

    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # 合并图片和彩色掩码
    combined = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    return combined