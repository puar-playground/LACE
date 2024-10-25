from typing import Optional

import torch
import torch.nn.functional as F

from lace.utils import xywh_to_ltrb


def piou_xywh(
    bbox_xywh: torch.Tensor, mask: Optional[torch.Tensor] = None, xy_only: bool = True
) -> torch.Tensor:
    """Compute the pairwise IoU for a set of bounding boxes.

    Args:
        bbox_xywh: Bounding boxes to compute the pairwise IoU for.
        mask: Mask of bounding boxes to compute the pairwise IoU for.
        xy_only: Compute the pairwise IoU using only the x, y coordinates.
    """
    # shape: (b, n, 4); 4 = [x, y, w, h]
    assert bbox_xywh.ndim == 3 and bbox_xywh.shape[-1] == 4

    if xy_only:
        wh = torch.abs(bbox_xywh[:, :, 2:].clone().detach())
        bbox = torch.cat([bbox_xywh[:, :, :2], wh], dim=2)
        bbox_ltrb = xywh_to_ltrb(bbox)
    else:
        bbox_ltrb = xywh_to_ltrb(bbox_xywh)

    return piou_ltrb(bbox_ltrb, mask)


def piou_ltrb(
    bbox_ltrb: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the pairwise IoU for a set of bounding boxes.

    Args:
        bbox_ltrb: Bounding boxes to compute the pairwise IoU for.
        mask: Mask of bounding boxes to compute the pairwise IoU for.

    Returns:
        torch.Tensor: Pairwise IoU.
    """
    n_box = bbox_ltrb.shape[1]
    device = bbox_ltrb.device

    # compute area of bboxes
    area_bbox = (bbox_ltrb[:, :, 2] - bbox_ltrb[:, :, 0]) * (
        bbox_ltrb[:, :, 3] - bbox_ltrb[:, :, 1]
    )
    area_bbox_psum = area_bbox.unsqueeze(-1) + area_bbox.unsqueeze(-2)

    # compute pairwise intersection
    x1y1 = bbox_ltrb[:, :, [0, 1]]
    x1y1 = torch.swapaxes(x1y1, 1, 2)
    x1y1_I = torch.max(x1y1.unsqueeze(-1), x1y1.unsqueeze(-2))

    x2y2 = bbox_ltrb[:, :, [2, 3]]
    x2y2 = torch.swapaxes(x2y2, 1, 2)
    x2y2_I = torch.min(x2y2.unsqueeze(-1), x2y2.unsqueeze(-2))

    # compute area of Is
    wh_I = F.relu(x2y2_I - x1y1_I)
    area_I = wh_I[:, 0, :, :] * wh_I[:, 1, :, :]

    # compute pairwise IoU
    piou = area_I / (area_bbox_psum - area_I + 1e-10)

    piou.masked_fill_(torch.eye(n_box, n_box).to(torch.bool).to(device), 0)

    if mask is not None:
        mask = mask.unsqueeze(2)
        select_mask = torch.matmul(mask, torch.transpose(mask, dim0=1, dim1=2))
        piou = piou * select_mask.to(device)

    return piou
