import torch
import numpy as np
from typing import List, Tuple
from torch import BoolTensor, FloatTensor, LongTensor
from torch_geometric.utils import to_dense_batch

def sparse_to_dense(
    batch,
    device: torch.device = torch.device("cpu"),
    remove_canvas: bool = False,
) -> Tuple[FloatTensor, LongTensor, BoolTensor, BoolTensor]:
    batch = batch.to(device)
    bbox, _ = to_dense_batch(batch.x, batch.batch)
    label, mask = to_dense_batch(batch.y, batch.batch)

    if remove_canvas:
        bbox = bbox[:, 1:].contiguous()
        label = label[:, 1:].contiguous() - 1  # cancel +1 effect in transform
        label = label.clamp(min=0)
        mask = mask[:, 1:].contiguous()

    padding_mask = ~mask
    return bbox, label, padding_mask, mask


def pad_sequence(seq: LongTensor, max_seq_length: int, value) -> LongTensor:
    S = seq.shape[1]
    new_shape = list(seq.shape)
    s = max_seq_length - S
    if s > 0:
        new_shape[1] = s
        pad = torch.full(new_shape, value, dtype=seq.dtype)
        new_seq = torch.cat([seq, pad], dim=1)
    else:
        new_seq = seq

    return new_seq


def pad_until(label: LongTensor, bbox: FloatTensor, mask: BoolTensor, max_seq_length: int
) -> Tuple[LongTensor, FloatTensor, BoolTensor]:
    label = pad_sequence(label, max_seq_length, 0)
    bbox = pad_sequence(bbox, max_seq_length, 0)
    mask = pad_sequence(mask, max_seq_length, False)
    return label, bbox, mask


def _to(inputs, device):
    """
    recursively send tensor to the specified device
    """
    outputs = {}
    for k, v in inputs.items():
        if isinstance(v, dict):
            outputs[k] = _to(v, device)
        elif isinstance(v, torch.Tensor):
            outputs[k] = v.to(device)
    return outputs

def loader_to_list(
    loader: torch.utils.data.dataloader.DataLoader,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    layouts = []
    for batch in loader:
        bbox, label, _, mask = sparse_to_dense(batch)
        for i in range(len(label)):
            valid = mask[i].numpy()
            layouts.append((bbox[i].numpy()[valid], label[i].numpy()[valid]))
    return layouts