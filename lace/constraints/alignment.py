import torch

from lace.utils import xywh_to_ltrb_split


def layout_alignment_matrix(bbox: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the alignment matrix for a set of bounding boxes.

    Args:
        bbox: Bounding boxes to compute the alignment matrix for.
        mask: Mask of bounding boxes to compute the alignment matrix for.

    Returns:
        torch.Tensor: Alignment matrix.
    """
    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = xywh_to_ltrb_split(bbox)
    xc, yc = bbox[0], bbox[1]
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    return X
