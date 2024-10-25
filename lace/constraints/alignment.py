import torch
from einops import rearrange

from lace.utils import xywh_to_ltrb_split


def layout_alignment_matrix(bbox: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the alignment matrix for a set of bounding boxes.

    Args:
        bbox: Bounding boxes to compute the alignment matrix for.
        mask: Mask of bounding boxes to compute the alignment matrix for.

    Returns:
        torch.Tensor: Alignment matrix.
    """

    bbox = rearrange(bbox, "b n c -> c b n")
    xl, yt, xr, yb = xywh_to_ltrb_split(bbox)
    xc, yc = bbox[0], bbox[1]

    X = rearrange([xl, xc, xr, yt, yc, yb], "n b c -> b n c")
    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0

    X = rearrange(X.abs(), "b n c1 c2 -> b c1 n c2")
    X[~mask] = 1.0
    return X
