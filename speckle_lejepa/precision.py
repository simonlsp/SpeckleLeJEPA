"""Choose native BF16 or FP16 mixed precision on the selected CUDA device."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MixedPrecision:
    device_type: str
    dtype: torch.dtype | None

    @classmethod
    def for_device(cls, device: torch.device | str) -> "MixedPrecision":
        device = torch.device(device)
        if device.type == "cpu":
            return cls("cpu", None)
        if device.type != "cuda":
            raise ValueError("MixedPrecision supports CPU and CUDA devices.")
        with torch.cuda.device(device):
            native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        return cls("cuda", torch.bfloat16 if native_bf16 else torch.float16)

    def autocast(self) -> torch.autocast:
        return torch.autocast(self.device_type, dtype=self.dtype, enabled=self.dtype is not None)

    def grad_scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)
