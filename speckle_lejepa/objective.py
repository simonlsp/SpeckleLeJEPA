"""Global-view prediction and sliced Epps–Pulley regularization."""

import math
from typing import NamedTuple

import torch
from torch import Tensor, distributed as dist, nn
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F


class Losses(NamedTuple):
    total: Tensor
    prediction: Tensor
    sigreg: Tensor


class LeJEPALoss(nn.Module):
    """Compute both LeJEPA losses on projections ordered as [view, cell, feature]."""

    def __init__(
        self,
        *,
        reg_weight: float = 0.05,
        num_slices: int = 256,
        num_knots: int = 17,
        t_max: float = 3.0,
        distributed: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= reg_weight <= 1.0:
            raise ValueError("reg_weight must be between 0 and 1.")
        if num_slices < 1 or num_knots < 2 or not math.isfinite(t_max) or t_max <= 0:
            raise ValueError("SIGReg needs positive slices and t_max, and at least two knots.")
        self.reg_weight = reg_weight
        self.num_slices = num_slices
        self.distributed = distributed

        frequencies = torch.linspace(0.0, t_max, num_knots, dtype=torch.float32)
        target = torch.exp(-0.5 * frequencies.square())
        step = t_max / (num_knots - 1)
        weights = torch.full_like(frequencies, 2.0 * step)
        weights[[0, -1]] = step
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.register_buffer("target", target, persistent=False)
        self.register_buffer("weights", weights * target, persistent=False)

    def forward(self, projected: Tensor, global_count: int) -> Losses:
        if projected.ndim != 3 or any(size == 0 for size in projected.shape):
            raise ValueError("projected must be a nonempty [V, B, D] tensor.")
        if not 1 <= global_count <= projected.size(0):
            raise ValueError("global_count must identify the first global_count views.")
        if self.distributed and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("Initialize the process group before distributed LeJEPA forward.")

        with torch.autocast(device_type=projected.device.type, enabled=False):
            projected = projected.float()
            center = projected[:global_count].mean(dim=0)
            prediction_sum = (projected - center).square().mean(dim=(0, 2)).sum()

            directions = torch.randn(
                projected.size(-1), self.num_slices, device=projected.device, dtype=torch.float32
            )
            directions = F.normalize(directions, dim=0)
            if self.distributed:
                dist.broadcast(directions, src=0)

            angles = (projected @ directions).unsqueeze(-1) * self.frequencies
            real_sum = angles.cos().sum(dim=1)
            imag_sum = angles.sin().sum(dim=1)
            count = projected.new_tensor(projected.size(1))

            if self.distributed:
                # Sum sufficient statistics so the result is weighted by cells, not ranks.
                packed = torch.cat(
                    (
                        prediction_sum.reshape(1),
                        count.reshape(1),
                        real_sum.flatten(),
                        imag_sum.flatten(),
                    )
                )
                packed = all_reduce(packed, op=dist.ReduceOp.SUM)
                prediction_sum, count = packed[0], packed[1]
                real, imag = packed[2:].chunk(2)
                real_sum = real.reshape_as(real_sum)
                imag_sum = imag.reshape_as(imag_sum)

            error = (real_sum / count - self.target).square() + (imag_sum / count).square()
            sigreg = (error @ self.weights).mean() * count
            prediction = prediction_sum / count
            total = (1.0 - self.reg_weight) * prediction + self.reg_weight * sigreg
        return Losses(total=total, prediction=prediction, sigreg=sigreg)
