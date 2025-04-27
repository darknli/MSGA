import os
from glob import glob

import cv2
import numpy as np


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


def read_dir(root=r"D:\temp_data\seg_dataset\DUTS-TE"):
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    images = glob(os.path.join(image_dir, "*"))
    print(f"发现[DUTS]有{len(images)}个样本")
    for img_path in images:
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        cv2.imshow("image", image)
        cv2.imshow("mask", mask)
        cv2.imshow("merge", combine_image_and_mask(image, mask))
        cv2.waitKey()


if __name__ == '__main__':
    read_dir(r"D:\temp_data\seg_dataset\DUTS-TR")
