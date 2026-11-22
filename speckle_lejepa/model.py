"""Shared ResNet-50 encoder and a three-layer training projector."""

import torch
from torch import Tensor, nn
from torchvision.models import resnet50

from .objective import LeJEPALoss, Losses


class SpeckleLeJEPA(nn.Module):
    """Accept global and local views in batch-first order and return SSL losses."""

    embedding_dim = 2048
    global_size = 96
    local_size = 48

    def __init__(
        self,
        *,
        projector_dim: int = 128,
        hidden_dim: int = 2048,
        reg_weight: float = 0.05,
        num_slices: int = 256,
        num_knots: int = 17,
        t_max: float = 3.0,
        distributed: bool = False,
    ) -> None:
        super().__init__()
        if projector_dim < 1 or hidden_dim < 1:
            raise ValueError("Projector dimensions must be positive.")

        self.backbone = resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projector_dim),
        )
        self.objective = LeJEPALoss(
            reg_weight=reg_weight,
            num_slices=num_slices,
            num_knots=num_knots,
            t_max=t_max,
            distributed=distributed,
        )

    def encode(self, images: Tensor) -> Tensor:
        """Return the pooled backbone features for floating-point [N, 3, H, W] images."""
        return self.backbone(images)

    @staticmethod
    def _validate_views(views: Tensor, size: int, name: str) -> None:
        if views.ndim != 5 or views.shape[2:] != (3, size, size):
            raise ValueError(f"{name} must have shape [B, V, 3, {size}, {size}].")
        if views.size(0) == 0 or views.size(1) == 0:
            raise ValueError(f"{name} requires at least one cell and one view.")
        if not views.is_floating_point():
            raise TypeError(f"{name} must contain floating-point pixels.")

    def forward(self, global_views: Tensor, local_views: Tensor) -> Losses:
        self._validate_views(global_views, self.global_size, "global_views")
        self._validate_views(local_views, self.local_size, "local_views")
        batch_size, global_count = global_views.shape[:2]
        if local_views.size(0) != batch_size:
            raise ValueError("Global and local views must have the same batch size.")
        if local_views.device != global_views.device or local_views.dtype != global_views.dtype:
            raise ValueError("Global and local views must share a device and dtype.")

        global_features = self.encode(global_views.flatten(0, 1)).unflatten(
            0, (batch_size, global_count)
        )
        local_features = self.encode(local_views.flatten(0, 1)).unflatten(
            0, (batch_size, local_views.size(1))
        )
        features = torch.cat((global_features, local_features), dim=1)
        projected = self.projector(features.flatten(0, 1))
        projected = projected.unflatten(0, (batch_size, features.size(1))).transpose(0, 1)
        return self.objective(projected, global_count)
