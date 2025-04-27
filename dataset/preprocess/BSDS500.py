"""
这个数据集太脏，低优使用

"""

import os
from glob import glob

import cv2
import numpy as np


def read_dir(root=r"D:\temp_data\seg_dataset\BSDS500"):
    images = os.path.join(root, "images")
    labels = os.path.join(root, "labels")
    for image_dir in glob(os.path.join(images, "*")):
        basename = os.path.basename(image_dir)
        label_dir = os.path.join(labels, basename)
        for img_path in glob(os.path.join(image_dir, "*")):
            img_name = os.path.basename(img_path)
            mask_path = os.path.join(label_dir, img_name.replace(".jpg", ".png"))
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            cv2.imshow("image", image)
            indices = np.unique(mask)
            for i in indices:
                print(i)
                mask_cls = (mask == i).astype(np.uint8) * 255
                cv2.imshow("mask", mask_cls)
                cv2.waitKey()


if __name__ == '__main__':
    read_dir()
