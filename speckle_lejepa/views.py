"""Build masked cell views with measurable foreground and coverage constraints."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import interpolate


@dataclass(frozen=True, slots=True)
class CropRule:
    min_side_ratio: float
    max_side_ratio: float
    min_coverage: float
    min_foreground: float

    def __post_init__(self) -> None:
        if not 0 < self.min_side_ratio <= self.max_side_ratio < math.inf:
            raise ValueError("Crop side ratios must be finite, positive and ordered.")
        if not (0 < self.min_coverage <= 1 and 0 < self.min_foreground <= 1):
            raise ValueError("Coverage and foreground thresholds must be in (0, 1].")


@dataclass(frozen=True, slots=True)
class CellViews:
    cell_size: int = 128
    global_count: int = 2
    local_count: int = 2
    global_crop: CropRule = CropRule(0.9, 1.2, 0.75, 0.15)
    local_crop: CropRule = CropRule(0.35, 0.65, 0.05, 0.45)

    def __post_init__(self) -> None:
        if min(self.cell_size, self.global_count, self.local_count) < 1:
            raise ValueError("Cell size and both view counts must be positive.")

    def __call__(
        self,
        rgb: np.ndarray,
        labels: np.ndarray,
        label: int,
        box: tuple[int, int, int, int],
        rng: np.random.Generator,
    ) -> tuple[Tensor, Tensor] | None:
        pixels, mask = self._cell_canvas(rgb, labels, label, box)
        rows, columns = np.nonzero(mask)
        if not rows.size:
            return None
        extent = max(int(np.ptp(rows)), int(np.ptp(columns))) + 1
        integral = np.pad(mask.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32), ((1, 0), (1, 0)))
        windows: dict[int, np.ndarray] = {}
        groups = []
        for rule, output_side, count in (
            (self.global_crop, 96, self.global_count),
            (self.local_crop, 48, self.local_count),
        ):
            views = []
            for _ in range(count):
                crop = self._choose_crop(
                    mask, integral, windows, extent, rows.size, rule, output_side, rng
                )
                if crop is None:
                    return None
                top, left, side, output_mask = crop
                image = pixels[:, top : top + side, left : left + side]
                views.append(
                    _resize_rgb(image, output_side).masked_fill(~torch.from_numpy(output_mask), 0)
                )
            groups.append(torch.stack(views))
        return groups[0], groups[1]

    def _cell_canvas(
        self, rgb: np.ndarray, labels: np.ndarray, label: int, box: tuple[int, int, int, int]
    ) -> tuple[Tensor, np.ndarray]:
        y0, x0, y1, x1 = box
        side = max(self.cell_size, y1 - y0, x1 - x0)
        image, mask = masked_cell_crop(rgb, labels, label, box, side)
        if side != self.cell_size:
            mask = _resize_mask(mask, self.cell_size)
            image = _resize_rgb(image, self.cell_size).masked_fill(~torch.from_numpy(mask), 0)
        return image, mask

    def _choose_crop(
        self,
        mask: np.ndarray,
        integral: np.ndarray,
        windows: dict[int, np.ndarray],
        extent: int,
        label_area: int,
        rule: CropRule,
        output_side: int,
        rng: np.random.Generator,
    ) -> tuple[int, int, int, np.ndarray] | None:
        minimum = max(1, min(self.cell_size, math.ceil(extent * rule.min_side_ratio)))
        maximum = min(self.cell_size, max(minimum, math.ceil(extent * rule.max_side_ratio)))
        for side in rng.permutation(np.arange(minimum, maximum + 1)).tolist():
            required = math.ceil(max(label_area * rule.min_coverage, side**2 * rule.min_foreground))
            if required > min(side**2, label_area):
                continue
            if side not in windows:
                windows[side] = integral[side:, side:] - integral[side:, :-side]
                windows[side] -= integral[:-side, side:] - integral[:-side, :-side]
            sums = windows[side]
            candidates = np.flatnonzero(sums >= required)
            for index in rng.permutation(candidates).tolist():
                y, x = divmod(index, sums.shape[1])
                output_mask = _resize_mask(mask[y : y + side, x : x + side], output_side)
                if output_mask.sum() >= math.ceil(output_side**2 * rule.min_foreground):
                    return y, x, side, output_mask
        return None


def masked_cell_crop(
    rgb: np.ndarray, labels: np.ndarray, label: int, box: tuple[int, int, int, int], side: int
) -> tuple[Tensor, np.ndarray]:
    """Center a square crop on the bounding box and zero everything outside the instance."""
    y0, x0, y1, x1 = box
    top = (y0 + y1 - side) // 2
    left = (x0 + x1 - side) // 2
    height, width = labels.shape
    start_y, end_y = max(0, top), min(height, top + side)
    start_x, end_x = max(0, left), min(width, left + side)
    target_y = slice(start_y - top, end_y - top)
    target_x = slice(start_x - left, end_x - left)
    mask = np.zeros((side, side), dtype=bool)
    mask[target_y, target_x] = labels[start_y:end_y, start_x:end_x] == label
    pixels = np.zeros((3, side, side), dtype=np.float32)
    pixels[:, target_y, target_x] = rgb[:, start_y:end_y, start_x:end_x]
    pixels /= np.iinfo(rgb.dtype).max
    pixels *= mask
    return torch.from_numpy(pixels), mask


def _resize_rgb(image: Tensor, size: int) -> Tensor:
    if image.size(-1) == size:
        return image
    return interpolate(
        image.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False, antialias=True
    ).squeeze(0)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] == size:
        return mask.copy()
    indices = ((2 * np.arange(size) + 1) * mask.shape[0]) // (2 * size)
    return mask[np.ix_(indices, indices)]
