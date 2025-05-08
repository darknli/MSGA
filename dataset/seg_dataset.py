import torch
from torch.utils.data import Dataset
from dataset.preprocess.pascal_context import read_pascal_context
from torchvision.transforms import v2
import numpy as np
import random
from PIL import Image


dataset_map = {
    "pascal_context": read_pascal_context,
}


class NormalImageOperator:
    def __init__(self, size):
        self.size = size
        self.transform_resize = v2.Compose(
            [
                v2.Resize(self.size),
                v2.CenterCrop(self.size),
            ]
        )
        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.ToTensor(),
            ]
        )

    def __call__(self, data, target_size=None):
        data_new = {}
        key_arr, value_arr = list(zip(*data.items()))
        if target_size is None:
            value_arr = self.transform_resize(value_arr)
        else:
            w, h = value_arr[0].size
            tw, th = target_size
            ratio = max(tw / w, th / h)
            nw, nh = int(w * ratio + 0.5), int(h * ratio + 0.5)
            value_arr = v2.Resize((nh, nw))(value_arr)
            value_arr = v2.CenterCrop((th, tw))(value_arr)
        value_arr = self.transform(value_arr)
        for key, value in zip(key_arr, value_arr):
            data_new[key] = value
        return data_new


class BucketDataset(Dataset):
    """
    数据集应该是packing_dataset函数处理过的格式
    """

    def __init__(self, tokenizer_name_or_path, datasets, max_batchsize=2, shuffle=False, dataset_type="train"):
        buckets = [(512, 512)]
        w = [(64 * r, 512) for r in range(8, 15)]
        h = [(512, 64 * r) for r in range(8, 15)]
        buckets.extend(w)
        buckets.extend(h)
        self.data = []
        if isinstance(datasets, str):
            datasets = [datasets]
        for dataset_name in datasets:
            self.data.extend(dataset_map[dataset_name](dataset_type=dataset_type))
        assert buckets is not None
        self.buckets = buckets
        self.asp_buckets = np.array([w / h for w, h in buckets])
        self.bid2row = {}
        # 每行数据对应的bucket id
        self.item2bid = []
        self.max_batchsize = max_batchsize
        self.shuffle = shuffle
        self.batch_indices = []
        self.set_bucket()
        self.build_batch_indices()
        self.lambda_func = NormalImageOperator(512)
        self.dataset_type = dataset_type

    def process(self, idx):
        data: dict = self.data[idx]
        bid = self.item2bid[idx]
        w, h = self.buckets[bid]
        image = Image.open(data["image"])
        mask_dict = torch.load(data["mask"], weights_only=False)
        if self.dataset_type == "train":
            mask_cls, mask = random.choice(list(mask_dict.items()))
        else:
            mask_cls, mask = list(mask_dict.items())[0]
        item = {"image": image, "mask": Image.fromarray(mask)}
        data_images = self.lambda_func(item, (w, h))
        output = {"prompt": mask_cls}
        output.update(data_images)
        return output

    def set_bucket(self):
        self.item2bid = []
        for idx in range(len(self.data)):
            image = Image.open(self.data[idx]["image"])
            width, height = image.size
            ratio = width / height
            # 找到最合适的那个bucket
            bid = np.argmin(np.abs(self.asp_buckets - ratio))
            self.item2bid.append(bid)
            if bid not in self.bid2row:
                self.bid2row[bid] = []
            self.bid2row[bid].append(idx)

    def build_batch_indices(self):
        print("build_batch...")
        if len(self.batch_indices) > 0 and not self.shuffle:
            return
        # 只有一个batch没有什么打乱的意义
        elif len(self.batch_indices) == 1:
            return
        self.batch_indices = []
        for bid, bucket in self.bid2row.items():
            if self.shuffle:
                random.shuffle(bucket)
            for start_idx in range(0, len(bucket), self.max_batchsize):
                end_idx = min(start_idx + self.max_batchsize, len(bucket))
                batch = bucket[start_idx:end_idx]
                self.batch_indices.append(batch)
        if self.shuffle:
            random.shuffle(self.batch_indices)

    def __getitem__(self, bid):
        assert self.batch_indices
        data = [self.process(idx) for idx in self.batch_indices[bid]]
        return data

    def __len__(self):
        return len(self.batch_indices)