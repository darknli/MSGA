import random
import shutil
import scipy.io
import os
from glob import glob
import torch
import cv2
import numpy as np


index2cls = [
    'empty', 'accordion', 'aeroplane', 'air conditioner', 'antenna', 'artillery', 'ashtray', 'atrium',
    'baby carriage', 'bag', 'ball', 'balloon', 'bamboo weaving', 'barrel', 'baseball bat', 'basket',
    'basketball backboard', 'bathtub', 'bed', 'bedclothes', 'beer', 'bell', 'bench', 'bicycle', 'binoculars',
    'bird', 'bird cage', 'bird feeder', 'bird nest', 'blackboard', 'board', 'boat', 'bone', 'book', 'bottle',
    'bottle opener', 'bowl', 'box', 'bracelet', 'brick', 'bridge', 'broom', 'brush', 'bucket', 'building',
    'bus', 'cabinet', 'cabinet door', 'cage', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camera lens',
    'can', 'candle', 'candle holder', 'cap', 'car', 'card', 'cart', 'case', 'casette recorder', 'cash register',
    'cat', 'cd', 'cd player', 'ceiling', 'cell phone', 'cello', 'chain', 'chair', 'chessboard', 'chicken',
    'chopstick', 'clip', 'clippers', 'clock', 'closet', 'cloth', 'clothes tree', 'coffee', 'coffee machine',
    'comb', 'computer', 'concrete', 'cone', 'container', 'control booth', 'controller', 'cooker', 'copying machine',
    'coral', 'cork', 'corkscrew', 'counter', 'court', 'cow', 'crabstick', 'crane', 'crate', 'cross', 'crutch', 'cup',
    'curtain', 'cushion', 'cutting board', 'dais', 'disc', 'disc case', 'dishwasher', 'dock', 'dog', 'dolphin',
    'door', 'drainer', 'dray', 'drink dispenser', 'drinking machine', 'drop', 'drug', 'drum', 'drum kit', 'duck',
    'dumbbell', 'earphone', 'earrings', 'egg', 'electric fan', 'electric iron', 'electric pot', 'electric saw',
    'electronic keyboard', 'engine', 'envelope', 'equipment', 'escalator', 'exhibition booth', 'extinguisher',
    'eyeglass', 'fan', 'faucet', 'fax machine', 'fence', 'ferris wheel', 'fire extinguisher', 'fire hydrant',
    'fire place', 'fish', 'fish tank', 'fishbowl', 'fishing net', 'fishing pole', 'flag', 'flagstaff', 'flame',
    'flashlight', 'floor', 'flower', 'fly', 'foam', 'food', 'footbridge', 'forceps', 'fork', 'forklift', 'fountain',
    'fox', 'frame', 'fridge', 'frog', 'fruit', 'funnel', 'furnace', 'game controller', 'game machine', 'gas cylinder',
    'gas hood', 'gas stove', 'gift box', 'glass', 'glass marble', 'globe', 'glove', 'goal', 'grandstand', 'grass',
    'gravestone', 'ground', 'guardrail', 'guitar', 'gun', 'hammer', 'hand cart', 'handle', 'handrail', 'hanger',
    'hard disk drive', 'hat', 'hay', 'headphone', 'heater', 'helicopter', 'helmet', 'holder', 'hook', 'horse',
    'horse-drawn carriage', 'hot-air balloon', 'hydrovalve', 'ice', 'inflator pump', 'ipod', 'iron', 'ironing board',
    'jar', 'kart', 'kettle', 'key', 'keyboard', 'kitchen range', 'kite', 'knife', 'knife block', 'ladder',
    'ladder truck', 'ladle', 'laptop', 'leaves', 'lid', 'life buoy', 'light', 'light bulb', 'lighter', 'line',
    'lion', 'lobster', 'lock', 'machine', 'mailbox', 'mannequin', 'map', 'mask', 'mat', 'match book', 'mattress',
    'menu', 'metal', 'meter box', 'microphone', 'microwave', 'mirror', 'missile', 'model', 'money', 'monkey', 'mop',
    'motorbike', 'mountain', 'mouse', 'mouse pad', 'musical instrument', 'napkin', 'net', 'newspaper', 'oar',
    'ornament', 'outlet', 'oven', 'oxygen bottle', 'pack', 'pan', 'paper', 'paper box', 'paper cutter', 'parachute',
    'parasol', 'parterre', 'patio', 'pelage', 'pen', 'pen container', 'pencil', 'person', 'photo', 'piano', 'picture',
    'pig', 'pillar', 'pillow', 'pipe', 'pitcher', 'plant', 'plastic', 'plate', 'platform', 'player', 'playground',
    'pliers', 'plume', 'poker', 'poker chip', 'pole', 'pool table', 'postcard', 'poster', 'pot', 'pottedplant',
    'printer', 'projector', 'pumpkin', 'rabbit', 'racket', 'radiator', 'radio', 'rail', 'rake', 'ramp', 'range hood',
    'receiver', 'recorder', 'recreational machines', 'remote control', 'road', 'robot', 'rock', 'rocket',
    'rocking horse', 'rope', 'rug', 'ruler', 'runway', 'saddle', 'sand', 'saw', 'scale', 'scanner', 'scissors',
    'scoop', 'screen', 'screwdriver', 'sculpture', 'scythe', 'sewer', 'sewing machine', 'shed', 'sheep', 'shell',
    'shelves', 'shoe', 'shopping cart', 'shovel', 'sidecar', 'sidewalk', 'sign', 'signal light', 'sink', 'skateboard',
    'ski', 'sky', 'sled', 'slippers', 'smoke', 'snail', 'snake', 'snow', 'snowmobiles', 'sofa', 'spanner', 'spatula',
    'speaker', 'speed bump', 'spice container', 'spoon', 'sprayer', 'squirrel', 'stage', 'stair', 'stapler', 'stick',
    'sticky note', 'stone', 'stool', 'stove', 'straw', 'stretcher', 'sun', 'sunglass', 'sunshade',
    'surveillance camera', 'swan', 'sweeper', 'swim ring', 'swimming pool', 'swing', 'switch', 'table',
    'tableware', 'tank', 'tap', 'tape', 'tarp', 'telephone', 'telephone booth', 'tent', 'tire', 'toaster',
    'toilet', 'tong', 'tool', 'toothbrush', 'towel', 'toy', 'toy car', 'track', 'train', 'trampoline', 'trash bin',
    'tray', 'tree', 'tricycle', 'tripod', 'trophy', 'truck', 'tube', 'turtle', 'tvmonitor', 'tweezers', 'typewriter',
    'umbrella', 'unknown', 'vacuum cleaner', 'vending machine', 'video camera', 'video game console', 'video player',
    'video tape', 'violin', 'wakeboard', 'wall', 'wallet', 'wardrobe', 'washing machine', 'watch', 'water',
    'water dispenser', 'water pipe', 'water skate board', 'watermelon', 'whale', 'wharf', 'wheel', 'wheelchair',
    'window', 'window blinds', 'wineglass', 'wire', 'wood', 'wool'
]


def read_pascal_context(
        root=r"D:\temp_data\seg_dataset\PascalContext", except_cls="unknown", dataset_type="train"
):
    root = os.path.join(root, dataset_type)
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    images = glob(os.path.join(image_dir, "*"))
    print(f"发现[DUTS]({dataset_type})有{len(images)}个样本")
    data = []
    for img_path in images:
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".pth"))
        if not os.path.isfile(mask_path):
            continue
        item = {
            "image": img_path,
            "mask": mask_path
        }
        data.append(item)
    return data


def split_dataset(root=r"D:\temp_data\seg_dataset\PascalContext", split_ratio=0.9):
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    train_image = os.path.join(root, "train", "images")
    train_mask = os.path.join(root, "train", "masks")
    valid_image = os.path.join(root, "valid", "images")
    valid_mask = os.path.join(root, "valid", "masks")
    images = glob(os.path.join(image_dir, "*"))
    print(f"发现[DUTS]有{len(images)}个样本")
    for path in [train_image, train_mask, valid_image, valid_mask]:
        os.makedirs(path, exist_ok=True)
    for img_path in images:
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".mat"))
        if not os.path.isfile(mask_path):
            continue
        if random.random() < split_ratio:
            shutil.move(img_path, train_image)
            shutil.move(mask_path, train_mask)
        else:
            shutil.move(img_path, valid_image)
            shutil.move(mask_path, valid_mask)


def convert_mask(
        root=r"D:\temp_data\seg_dataset\PascalContext", except_cls="unknown", dataset_type="train"
):
    root = os.path.join(root, dataset_type)
    image_dir = os.path.join(root, "images")
    mask_dir = os.path.join(root, "masks")
    images = glob(os.path.join(image_dir, "*"))
    print(f"发现[DUTS]({dataset_type})有{len(images)}个样本")
    data = []
    for img_path in images:
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".mat"))
        if not os.path.isfile(mask_path):
            continue
        mask = scipy.io.loadmat(mask_path)["LabelMap"].astype(int)
        # cv2.imshow("image", image)
        indices = np.unique(mask)
        mask_dict = {}
        for i in indices:
            # print(i)
            # print(index2cls[i])
            if index2cls[i] == except_cls:
                continue
            mask_cls = (mask == i).astype(np.uint8) * 255
            # cv2.imshow("mask", mask_cls)
            # cv2.imshow("merge", combine_image_and_mask(image, mask_cls))
            # cv2.waitKey()
            mask_dict[index2cls[i]] = mask_cls

        if len(mask_dict) == 0:
            continue
        torch.save(mask_dict, mask_path.replace(".mat", ".pth"))
    return data


if __name__ == '__main__':
    # split_dataset()
    # read_pascal_context()
    convert_mask(dataset_type="train")