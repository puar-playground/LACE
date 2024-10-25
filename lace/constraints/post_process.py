import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

from lace.constraints.alignment import layout_alignment_matrix
from lace.constraints.piou import piou_xywh

logger = logging.getLogger(__name__)


@dataclass
class PostProcessOutput(object):
    bbox: torch.Tensor
    mask: torch.Tensor
    loss: Optional[torch.Tensor] = None


@torch.enable_grad
def post_process(
    bbox: torch.Tensor,
    mask: torch.Tensor,
    xy_only: bool = False,
    w_o: float = 1.0,
    w_a: float = 1.0,
    ali_threshold: float = 1 / 64,
    max_iterations: int = 1000,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    adam_betas: Tuple[float, float] = (0.9, 0.999),
    adam_eps: float = 1e-08,
    disable_progress_bar: bool = False,
) -> PostProcessOutput:
    """LACE's post-processing optimization for bounding boxes.

    Args:
        bbox: Bounding boxes to optimize.
        mask: Mask of bounding boxes to optimize.
        xy_only: Optimize only the x, y coordinates of the bounding boxes.
        w_o: Weight for the overlap constraint.
        w_a: Weight for the alignment constraint.
        ali_threshold: Threshold for the alignment constraint.
        max_iterations: Maximum number of iterations.
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        adam_betas: Betas for the Adam optimizer.
        adam_eps: Epsilon for the Adam optimizer.

    Returns:
        PostProcessOutput: Optimized bounding boxes, mask, and loss
    """
    # shape: (batch_size, max_bboxes, 4)
    assert bbox.ndim == 3
    # shape: (batch_size, max_bboxes)
    assert mask.ndim == 2

    if torch.sum(mask) == 1:
        return PostProcessOutput(bbox=bbox, mask=mask)

    if xy_only:
        wh = torch.abs(bbox[:, :, 2:].clone().detach())
        bbox_in = torch.cat([bbox[:, :, :2], wh], dim=2)
    else:
        bbox_in = bbox

    bbox_in[:, :, [0, 2]] *= 10 / 4
    bbox_in[:, :, [1, 3]] *= 10 / 6

    bbox_initial = bbox_in.clone().detach()

    bbox_p = nn.Parameter(bbox_in)
    optimizer = optim.Adam(
        [bbox_p],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=adam_betas,
        eps=adam_eps,
        amsgrad=False,
    )

    with tqdm(
        total=max_iterations,
        desc="Perform LACE's post-processing optimization",
        disable=disable_progress_bar,
        dynamic_ncols=True,
    ) as pbar:
        for _ in range(max_iterations):
            bbox_1 = torch.relu(bbox_p)
            align_score_target = layout_alignment_matrix(bbox_1, mask)
            align_mask = (align_score_target < ali_threshold).clone().detach()
            align_loss = torch.mean(align_score_target * align_mask)

            piou_m = piou_xywh(bbox_1, mask=mask.to(torch.float32), xy_only=True)
            piou = torch.mean(piou_m)

            mse_loss = F.mse_loss(bbox_1, bbox_initial)

            aux_loss = w_a * align_loss + w_o * piou
            loss = mse_loss + aux_loss

            pbar.set_postfix(
                {
                    "mse": mse_loss.item(),
                    "ali": align_loss.item(),
                    "ove": piou.item(),
                    "total": loss.item(),
                }
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([bbox_p], 1.0)
            optimizer.step()

            a, _ = torch.min(bbox_1[:, :, [2, 3]], dim=2)
            mask = mask * (a > 0.01)

            pbar.update()

    bbox_out = torch.relu(bbox_p)
    bbox_out[:, :, [0, 2]] *= 4 / 10
    bbox_out[:, :, [1, 3]] *= 6 / 10

    return PostProcessOutput(bbox=bbox_out, mask=mask, loss=loss)
