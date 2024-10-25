from typing import Tuple

import torch


def xywh_to_ltrb(bbox_xywh: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (x, y, w, h) to (left, top, right, bottom).

    Args:
        bbox_xywh: Bounding boxes in (x, y, w, h) format.

    Returns:
        torch.Tensor: Bounding boxes in (left, top, right, bottom) format.
    """
    bbox_ltrb = torch.zeros(bbox_xywh.shape, device=bbox_xywh.device)
    bbox_xy = torch.abs(bbox_xywh[:, :, :2])
    bbox_wh = torch.abs(bbox_xywh[:, :, 2:])
    bbox_ltrb[:, :, :2] = bbox_xy - 0.5 * bbox_wh
    bbox_ltrb[:, :, 2:] = bbox_xy + 0.5 * bbox_wh
    return bbox_ltrb


def ltrb_to_xywh(bbox_ltrb: torch.Tensor) -> torch.Tensor:
    """Convert bounding boxes from (left, top, right, bottom) to (x, y, w, h).

    Args:
        bbox_ltrb: Bounding boxes in (left, top, right, bottom) format.

    Returns:
        torch.Tensor: Bounding boxes in (x, y, w, h) format.
    """
    bbox_xywh = torch.zeros(bbox_ltrb.shape, device=bbox_ltrb.device)
    bbox_wh = torch.abs(bbox_ltrb[:, :, 2:] - bbox_ltrb[:, :, :2])
    bbox_xy = bbox_ltrb[:, :, :2] + 0.5 * bbox_wh
    bbox_xywh[:, :, :2] = bbox_xy
    bbox_xywh[:, :, 2:] = bbox_wh
    return bbox_xywh


def xywh_to_ltrb_split(
    bbox: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split bounding boxes from (x, y, w, h) to (left, top, right, bottom).

    Args:
        bbox: Bounding boxes in (x, y, w, h) format.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Bounding boxes in (left, top, right, bottom) format.
    """
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return (x1, y1, x2, y2)
