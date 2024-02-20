import os
from tqdm import tqdm
import random
random.seed(1)
from fsspec.core import url_to_fs
import numpy as np
import torch
from torch_geometric.data import Data
from .base import BaseDataset
import datasets

def ltrb_2_xywh(bbox_ltrb):
    bbox_xywh = torch.zeros(bbox_ltrb.shape)
    bbox_wh = torch.abs(bbox_ltrb[:, 2:] - bbox_ltrb[:, :2])
    bbox_xy = bbox_ltrb[:, :2] + 0.5 * bbox_wh
    bbox_xywh[:, :2] = bbox_xy
    bbox_xywh[:, 2:] = bbox_wh
    return bbox_xywh

class CrelloDataset(BaseDataset):
    name = "crello"
    label_names = ['coloredBackground', 'imageElement', 'maskElement', 'svgElement', 'textElement']
    label_dict = {k: i for i, k in enumerate(label_names)}

    def __init__(self, dir: str, split: str, max_seq_length: int, transform=None):
        super().__init__(dir, split, max_seq_length, transform)

    def download(self):
        _ = datasets.load_dataset("cyberagent/crello")

    def process(self):
        fs, _ = url_to_fs(self.raw_dir)
        for split_i, split in enumerate(['train', 'validation', 'test']):

            dataset = datasets.load_dataset("cyberagent/crello", split=split)
            # label_names = dataset.features['type'].feature.names
            width_list = [int(x) for x in dataset.features['canvas_width'].names]
            height_list = [int(x) for x in dataset.features['canvas_height'].names]

            data_list = []
            for i, elements in tqdm(enumerate(dataset), total=len(dataset), desc=f'split: {split}', ncols=150):
                labels = torch.tensor(elements['type']).to(torch.long)
                wi = elements['canvas_width']
                hi = elements['canvas_height']
                W = width_list[wi]
                H = height_list[hi]
                left = torch.tensor(elements['left'])
                top = torch.tensor(elements['top'])
                width = torch.tensor(elements['width'])
                height = torch.tensor(elements['height'])

                def is_valid(x1, y1, width, height):

                    if torch.min(width) <= 0 or torch.min(height) <= 0:
                        return None, False

                    x2, y2 = x1 + width, y1 + height
                    bboxes = torch.stack([x1, y1, x2, y2], dim=1)
                    if torch.max(bboxes) > 1 or torch.min(bboxes) < 0:
                        bboxes_ltbr = torch.clamp(bboxes, min=0, max=1)
                        bboxes = ltrb_2_xywh(bboxes_ltbr)
                        return bboxes, False

                    return bboxes, True

                bboxes, filtered = is_valid(left, top, width, height)
                if bboxes is None:
                    continue

                if bboxes.shape[0] == 0 or bboxes.shape[0] > self.max_seq_length:
                    continue

                data = Data(x=bboxes, y=labels)
                data.attr = {
                    "name": elements['id'],
                    "width": W,
                    "height": H,
                    "filtered": filtered,
                    "has_canvas_element": False,
                    "NoiseAdded": False,
                }
                data_list.append(data)

            with fs.open(self.processed_paths[split_i], "wb") as file_obj:
                if split_i == 0:
                    print('duplicate training split (x10) for more batches per epoch')
                    torch.save(self.collate(data_list * 10), file_obj)
                else:
                    torch.save(self.collate(data_list), file_obj)



if __name__ == '__main__':

    data_dir = '../../download/datasets'
    # get_train_labels(data_dir)
    # get_val_test_labels(data_dir)

    # train_dataset = MagazineDataset(dir=data_dir, split='train')
    # print(len(train_dataset))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=4)
    # for bbox, label, mask in tqdm(train_loader):
    #     print(bbox[0])
    #     print(label[0])
    #     print(mask[0])
    #     break

