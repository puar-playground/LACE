import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from util.datasets.load_data import init_dataset
from util.visualization import save_image
from util.seq_util import sparse_to_dense, pad_until
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=500, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=4, help="batch_size", type=int)
    parser.add_argument("--device", default='cuda', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=1, help="num_workers", type=int)
    parser.add_argument("--num_plot", default=50, help="num_workers", type=int)
    parser.add_argument("--dataset", default='magazine',
                        help="choose from [publaynet, rico13, rico25]", type=str)
    parser.add_argument("--data_dir", default='./datasets', help="dir of datasets", type=str)
    args = parser.parse_args()

    # prepare data
    train_dataset, train_loader = init_dataset(args.dataset, args.data_dir, batch_size=args.batch_size,
                                               split='train', shuffle=False)
    if not os.path.exists(f'./plot/{args.dataset}_train/'):
        os.mkdir(f'./plot/{args.dataset}_train/')

    with tqdm(enumerate(train_loader, 1), total=args.num_plot, desc=f'load data',
              ncols=120) as pbar:

        for i, data in pbar:

            if i > args.num_plot:
                break

            bbox, label, _, mask = sparse_to_dense(data)
            label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)

            a = save_image(bbox, label, mask, draw_label=True, dataset=f'{args.dataset}')
            plt.figure(figsize=[15, 20])
            plt.imshow(a)
            plt.tight_layout()
            plt.savefig(f'./plot/{args.dataset}_train/{i}.png')
            plt.close()



