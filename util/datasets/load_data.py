from util.datasets.publaynet import PubLayNetDataset
from util.datasets.magazine import MagazineDataset
from util.datasets.rico import Rico25Dataset, Rico5Dataset, Rico13Dataset
from util.datasets.crello import CrelloDataset
import torch
import torch_geometric
from util.seq_util import sparse_to_dense, pad_until

dataset_dict = {'publaynet': PubLayNetDataset, 'rico13': Rico13Dataset,
            'rico25': Rico25Dataset, 'magazine': MagazineDataset, 'crello': CrelloDataset}


def init_dataset(dataset_name, data_dir, batch_size=128, split='train', transform=None, shuffle=False):

    main_dataset = dataset_dict[dataset_name](dir=data_dir, split=split, max_seq_length=25, transform=transform)
    main_loader = torch_geometric.loader.DataLoader(main_dataset, shuffle=shuffle, batch_size=batch_size,
                                                    num_workers=4, pin_memory=True)
    return main_dataset, main_loader


if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = '/Users/chenjian/Documents/Projects/Layout/LayoutDiff/datasets'

    train_dataset, train_loader = init_dataset('crello', data_dir, batch_size=120, split='train')
    print(train_dataset.num_classes)

    for data in train_loader:
        bbox, label, _, mask = sparse_to_dense(data)
        label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)
        bbox = bbox.to(device)

        print(bbox[0])
        break

    #

