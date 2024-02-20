import os
from tqdm import tqdm
import random
random.seed(1)
from fsspec.core import url_to_fs
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric
from .base import BaseDataset
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET

magazine_cmap = {
    "text": (254, 231, 44),
    "image": (27, 187, 146),
    "headline": (255, 0, 0),
    "text-over-image": (0, 102, 255),
    "headline-over-image": (204, 0, 255),
    "background": (200, 200, 200),
}

label_dict = {"text": 0, "image": 1, "headline": 2, "text-over-image": 3, "headline-over-image": 4, "background": 5}


class MagazineDataset(BaseDataset):
    name = "magazine"
    labels = ["text", "image", "headline", "text-over-image", "headline-over-image", "background"]

    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        super().__init__(dir, split, max_seq_length, transform)

    def download(self):
        # super().download()
        pass

    def process(self):

        fs, _ = url_to_fs(self.raw_dir)
        # self.data_root
        self.colormap = magazine_cmap
        self.labels = ("text", "image", "headline", "text-over-image", "headline-over-image", "background")

        file_folder = os.path.join(self.raw_dir, 'annotations')

        file_cls_wise = {'fashion': [], 'food': [], 'science': [], 'news': [], 'travel': [], 'wedding': []}
        for file_name in sorted(os.listdir(file_folder)):
            cls = file_name.split('_')[0]
            file_cls_wise[cls].append(file_name)

        file_lists = dict()
        file_lists['train'] = sum([file_cls_wise[k][:int(0.9 * len(file_cls_wise[k]))] for k in file_cls_wise.keys()], [])
        file_lists['test'] = sum([file_cls_wise[k][int(0.9 * len(file_cls_wise[k])):] for k in file_cls_wise.keys()], [])

        for split in ['train', 'test']:
            file_list = file_lists[split]
            data_list = []

            for file_name in file_list:
                boxes = []
                labels = []

                tree = ET.parse(os.path.join(file_folder, file_name))
                root = tree.getroot()
                try:
                    for layout in root.findall('layout'):
                        if len(layout.findall('element')) > self.max_seq_length:
                            continue
                        for i, element in enumerate(layout.findall('element')):
                            label = element.get('label')
                            c = label_dict[label]
                            px = [int(i) for i in element.get('polygon_x').split(" ")]
                            py = [int(i) for i in element.get('polygon_y').split(" ")]
                            # get center coordinate (x,y), width and height (w, h)
                            x = (px[2] + px[0]) / 2 / 225
                            y = (py[2] + py[0]) / 2 / 300
                            w = (px[2] - px[0]) / 225
                            h = (py[2] - py[0]) / 300
                            boxes.append([x, y, w, h])
                            labels.append(c)

                except:
                    continue

                if len(labels) == 0:
                    continue

                boxes = torch.tensor(boxes, dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.long)

                data = Data(x=boxes, y=labels)
                data.attr = {
                    "name": file_name,
                    "width": 225,
                    "height": 300,
                    "filtered": True,
                    "has_canvas_element": False,
                    "NoiseAdded": False,
                }
                data_list.append(data)

            if split == "train":
                train_list = data_list
            else:
                test_list = data_list

        # train 90% / test 10%
        # self.processed_paths: [train. val, test]
        with fs.open(self.processed_paths[0], "wb") as file_obj:
            print('duplicate training split (x100) for more batches per epoch')
            torch.save(self.collate(train_list * 100), file_obj)
        with fs.open(self.processed_paths[1], "wb") as file_obj:
            torch.save([], file_obj)
        with fs.open(self.processed_paths[2], "wb") as file_obj:
            torch.save(self.collate(test_list), file_obj)
