import math

import pytest
import torch

import lace


@pytest.fixture
def batch_size() -> int:
    return 512


@pytest.fixture
def max_bboxes() -> int:
    return 25


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_post_process(batch_size: int, max_bboxes: int, device: torch.device):
    def create_random_bbox(device: torch.device, max_bboxes: int) -> torch.Tensor:
        # shape: (batch_size, max_bboxes)
        x = torch.rand(batch_size, max_bboxes)
        y = torch.rand(batch_size, max_bboxes)
        w = torch.rand(batch_size, max_bboxes) * (1 - x)
        h = torch.rand(batch_size, max_bboxes) * (1 - y)

        # shape: (batch_size, max_bboxes, 4)
        bbox = torch.stack([x, y, w, h], dim=2)
        bbox = bbox.to(device)

        assert bbox.shape == (batch_size, max_bboxes, 4)
        return bbox

    def create_random_mask(
        batch_size: int,
        max_bboxes: int,
        device: torch.device,
    ) -> torch.Tensor:
        # The random number generator is different between CPU and GPU,
        # so first generate random numbers on CPU and then move them to GPU.
        num_true_tensor = torch.randint(
            low=1,
            high=max_bboxes + 1,
            size=(batch_size, 1),
        )
        num_true_tensor = num_true_tensor.to(device)

        indices = torch.arange(max_bboxes, device=device)
        indices = indices.expand(batch_size, max_bboxes)

        masks = indices < num_true_tensor

        return masks

    bbox = create_random_bbox(
        max_bboxes=max_bboxes,
        device=device,
    )
    mask = create_random_mask(
        batch_size=batch_size,
        max_bboxes=max_bboxes,
        device=device,
    )
    output = lace.post_process(bbox=bbox, mask=mask)

    assert output.loss is not None and math.isclose(
        output.loss.item(), 0.0040787, rel_tol=1e-5
    )
