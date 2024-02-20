import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
import imageio
torch.manual_seed(0)
import math


def xywh_2_ltrb(bbox_xywh):

    bbox_ltrb = torch.zeros(bbox_xywh.shape).to(bbox_xywh.device)
    bbox_xy = torch.abs(bbox_xywh[:, :, :2])
    bbox_wh = torch.abs(bbox_xywh[:, :, 2:])
    bbox_ltrb[:, :, :2] = bbox_xy - 0.5 * bbox_wh
    bbox_ltrb[:, :, 2:] = bbox_xy + 0.5 * bbox_wh
    return bbox_ltrb


def ltrb_2_xywh(bbox_ltrb):
    bbox_xywh = torch.zeros(bbox_ltrb.shape)
    bbox_wh = torch.abs(bbox_ltrb[:, :, 2:] - bbox_ltrb[:, :, :2])
    bbox_xy = bbox_ltrb[:, :, :2] + 0.5 * bbox_wh
    bbox_xywh[:, :, :2] = bbox_xy
    bbox_xywh[:, :, 2:] = bbox_wh
    return bbox_xywh


def xywh_to_ltrb_split(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def rand_bbox_ltrb(batch_shape):

    bbox_lt = torch.rand(batch_shape + [2])
    bbox_wh_max = 1 - bbox_lt
    bbox_wh_weight = torch.rand(batch_shape).unsqueeze(-1).repeat([1 for _ in range(len(batch_shape))] + [2])

    bbox_wh = 1 * bbox_wh_weight * bbox_wh_max
    bbox_rb = bbox_lt + bbox_wh

    bbox = torch.cat([bbox_lt, bbox_rb], dim=-1)
    return bbox


def rand_bbox_xywh(batch_shape):

    bbox_ltrb = rand_bbox_ltrb(batch_shape)
    bbox_xywh = ltrb_2_xywh(bbox_ltrb)
    return bbox_xywh


def GIoU_ltrb(bbox_1, bbox_2):

    # step 1 calculate area of bbox_1 and bbox_2
    a_1 = (bbox_1[:, :, 2] - bbox_1[:, :, 0]) * (bbox_1[:, :, 3] - bbox_1[:, :, 1])
    a_2 = (bbox_2[:, :, 2] - bbox_2[:, :, 0]) * (bbox_2[:, :, 3] - bbox_2[:, :, 1])

    # step 2.1 compute intersection I bbox
    bbox = torch.cat([bbox_1.unsqueeze(-1), bbox_2.unsqueeze(-1)], dim=-1)
    bbox_I_lt = torch.max(bbox, dim=-1)[0][:, :, :2]
    bbox_I_rb = torch.min(bbox, dim=-1)[0][:, :, 2:]

    # step 2.2 compute area of I
    a_I = F.relu((bbox_I_rb[:, :, 0] - bbox_I_lt[:, :, 0])) * F.relu((bbox_I_rb[:, :, 1] - bbox_I_lt[:, :, 1]))

    # step 3.1 compute smallest enclosing box C
    bbox_C_lt = torch.min(bbox, dim=-1)[0][:, :, :2]
    bbox_C_rb = torch.max(bbox, dim=-1)[0][:, :, 2:]

    # step 3.2 compute area of C
    a_C = (bbox_C_rb[:, :, 0] - bbox_C_lt[:, :, 0]) * (bbox_C_rb[:, :, 1] - bbox_C_lt[:, :, 1])

    # step 4 compute IoU
    a_U = (a_1 + a_2 - a_I)
    iou = a_I / (a_U + 1e-10)

    # step 5 copute giou
    giou = iou - (a_C - a_U) / (a_C + 1e-10)

    return iou, giou


def GIoU_xywh(bbox_pred, bbox_true, xy_only=False):

    if xy_only:
        wh = torch.abs(bbox_pred[:, :, 2:].clone().detach())
        bbox = torch.cat([bbox_pred[:, :, :2], wh], dim=2)
    else:
        bbox = bbox_pred

    bbox_pred_ltrb = xywh_2_ltrb(torch.abs(bbox))
    bbox_true_ltrb = xywh_2_ltrb(torch.abs(bbox_true))
    return GIoU_ltrb(bbox_pred_ltrb, bbox_true_ltrb)


def PIoU_ltrb(bbox_ltrb, mask=None):

    n_box = bbox_ltrb.shape[1]
    device = bbox_ltrb.device

    # compute area of bboxes
    area_bbox = (bbox_ltrb[:, :, 2] - bbox_ltrb[:, :, 0]) * (bbox_ltrb[:, :, 3] - bbox_ltrb[:, :, 1])
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


def PIoU_xywh(bbox_xywh, mask=None, xy_only=True):

    if xy_only:
        wh = torch.abs(bbox_xywh[:, :, 2:].clone().detach())
        bbox = torch.cat([bbox_xywh[:, :, :2], wh], dim=2)
        bbox_ltrb = xywh_2_ltrb(bbox)
    else:
        bbox_ltrb = xywh_2_ltrb(bbox_xywh)

    return PIoU_ltrb(bbox_ltrb, mask)


def Pdist(bbox):
    xy = bbox[:, :, :2]
    pdist_m = torch.cdist(xy, xy, p=2)

    return pdist_m

def layout_alignment(bbox, mask, xy_only=False, mode='all'):
    """
    alignment metrics in Attribute-conditioned Layout GAN for Automatic Graphic Design (TVCG2020)
    https://arxiv.org/abs/2009.05284
    """

    if xy_only:
        wh = torch.abs(bbox[:, :, 2:].clone().detach())
        bbox = torch.cat([bbox[:, :, :2], wh], dim=2)

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = xywh_to_ltrb_split(bbox)
    xc, yc = bbox[0], bbox[1]
    if mode == 'all':
        X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    elif mode == 'partial':
        X = torch.stack([xl, xc, yt, yb], dim=1)
    else:
        raise Exception('mode must be all or partial')

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0

    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)
    X = -torch.log(1 - X)

    score = einops.reduce(X, "b s -> b", reduction="sum")
    score_normalized = score / einops.reduce(mask, "b s -> b", reduction="sum")
    score_normalized[torch.isnan(score_normalized)] = 0.0

    return score, score_normalized


def layout_alignment_matrix(bbox, mask):
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


def mean_alignment_error(bbox_target, bbox_true, mask_true, th=1e-5, xy_only=False):
    """
    mean coordinate difference error for aligned positions, a function for a batch
    tau_t: misalignment tolerance threshold
    th: threshold for alignment error in real-data
    mask_true: indices where the coordinate difference is smaller than th
    """

    if xy_only:
        wh = torch.abs(bbox_target[:, :, 2:].clone().detach())
        bbox = torch.cat([bbox_target[:, :, :2], wh], dim=2)
    else:
        bbox = bbox_target

    align_score_target = layout_alignment_matrix(bbox, mask_true)

    align_score_true = layout_alignment_matrix(bbox_true, mask_true)
    align_mask = (align_score_true < th).clone().detach()

    selected_difference = align_score_target * align_mask

    mae = einops.reduce(selected_difference, "n a b c -> n", reduction="sum")

    return mae


def constraint_temporal_weight(t, schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        w = 1 - torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        w = 1 - 4 * end * torch.ones(num_timesteps)
    elif schedule == "quad":
        w = 1 - torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        w = 1 - 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        w = 1 - torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        w = 1 - torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
    elif schedule == "cosine_anneal":
        w = 1 - torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])

    weight = w.cumprod(dim=0).to(t.device)

    return weight[t]


def post_process(bbox, mask_generated, xy_only=False, w_o=1, w_a=1):

    # print('beautify')
    if torch.sum(mask_generated) == 1:
        return bbox, mask_generated

    if xy_only:
        wh = torch.abs(bbox[:, :, 2:].clone().detach())
        bbox_in = torch.cat([bbox[:, :, :2], wh], dim=2)
    else:
        bbox_in = bbox

    bbox_in[:, :, [0, 2]] *= 10 / 4
    bbox_in[:, :, [1, 3]] *= 10 / 6

    bbox_initial = bbox_in.clone().detach()
    mse_loss = nn.MSELoss()

    bbox_p = nn.Parameter(bbox_in)
    optimizer = optim.Adam([bbox_p], lr=1e-4, weight_decay=0.0, betas=(0.9, 0.999), amsgrad=False, eps=1e-08)
    with torch.enable_grad():
        for i in range(1000):
            bbox_1 = torch.relu(bbox_p)
            align_score_target = layout_alignment_matrix(bbox_1, mask_generated)
            align_mask = (align_score_target < 1/64).clone().detach()
            align_loss = torch.mean(align_score_target * align_mask)

            piou_m = PIoU_xywh(bbox_1, mask=mask_generated.to(torch.float32), xy_only=True)
            piou = torch.mean(piou_m)

            mse = mse_loss(bbox_1, bbox_initial)
            loss = 1 * mse + w_a * align_loss + w_o * piou
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([bbox_p], 1.0)
            optimizer.step()

            a, _ = torch.min(bbox_1[:, :, [2, 3]], dim=2)
            mask_generated = mask_generated * (a > 0.01)

        bbox_out = torch.relu(bbox_p)
        bbox_out[:, :, [0, 2]] *= 4 / 10
        bbox_out[:, :, [1, 3]] *= 6 / 10

    return bbox_out, mask_generated


if __name__ == "__main__":

    w = constraint_temporal_weight(torch.tensor([225]), schedule="const")
    print(w)