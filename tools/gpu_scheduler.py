"""Creative Agent-style CUDA scheduling for the chroma-key demo.

The scheduler makes one preflight decision per asset.  A healthy, idle NVIDIA
GPU uses the CuPy path; a missing/busy GPU uses CPU.  The first CUDA exception
blacklists CUDA for the rest of the batch so later rows do not repeatedly pay
for the same failure.
"""

from __future__ import annotations

import subprocess
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class CudaMattingError(RuntimeError):
    """Raised when a CUDA matte cannot be completed safely."""


@dataclass(frozen=True)
class BackendDecision:
    backend: str
    gpu_utilization: int
    reason: str


class MattingScheduler:
    """GPU-first scheduler with CPU fallback and batch-level CUDA blacklisting."""

    def __init__(self, *, device: str = "auto", gpu_index: int = 0, util_threshold: int = 50) -> None:
        if device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"unsupported device: {device}")
        self.device = device
        self.gpu_index = gpu_index
        self.util_threshold = util_threshold
        self.cuda_disabled = False
        self.cuda_error = ""
        self._cupy: Any | None = None
        self._minimum_filter: Any | None = None
        self._gaussian_filter: Any | None = None
        self._gpu_name = ""
        self._util_cache: tuple[float, int] = (0.0, -1)

    @property
    def gpu_name(self) -> str:
        if not self._gpu_name:
            self._query_gpu()
        return self._gpu_name

    def _query_gpu(self) -> tuple[str, int]:
        now = time.monotonic()
        cached_at, cached_util = self._util_cache
        if now - cached_at < 5.0:
            return self._gpu_name, cached_util
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=name,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=True,
            )
            line = result.stdout.strip().splitlines()[0]
            name, util_text = [part.strip() for part in line.rsplit(",", 1)]
            util = int(util_text)
            self._gpu_name = name
        except Exception:
            util = -1
            self._gpu_name = ""
        self._util_cache = (now, util)
        return self._gpu_name, util

    def _load_cuda(self) -> None:
        if self._cupy is not None:
            return
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="CUDA path could not be detected.*")
                import cupy as cp
            from cupyx.scipy.ndimage import gaussian_filter, minimum_filter

            device = cp.cuda.Device(self.gpu_index)
            device.use()
            probe = cp.arange(16, dtype=cp.float32)
            float(cp.sum(probe).get())
            device.synchronize()
            self._cupy = cp
            self._minimum_filter = minimum_filter
            self._gaussian_filter = gaussian_filter
        except Exception as exc:
            raise CudaMattingError(f"CUDA initialization failed: {exc}") from exc

    def decide(self) -> BackendDecision:
        _, util = self._query_gpu()
        if self.device == "cpu":
            return BackendDecision("cpu", util, "device=cpu")
        if self.cuda_disabled:
            return BackendDecision("cpu", util, f"CUDA disabled after failure: {self.cuda_error}")
        if util < 0:
            return BackendDecision("cpu", util, "nvidia-smi unavailable or no NVIDIA GPU")
        if self.device == "auto" and util >= self.util_threshold:
            return BackendDecision("cpu", util, f"GPU utilization {util}% >= {self.util_threshold}%")
        try:
            self._load_cuda()
        except CudaMattingError as exc:
            self.disable_cuda(str(exc))
            return BackendDecision("cpu", util, self.cuda_error)
        mode_reason = "device=cuda" if self.device == "cuda" else f"GPU utilization {util}%"
        return BackendDecision("cuda", util, mode_reason)

    def disable_cuda(self, reason: str) -> None:
        self.cuda_disabled = True
        self.cuda_error = reason
        try:
            if self._cupy is not None:
                self._cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    def matte_cuda(
        self,
        rgb: np.ndarray,
        sampled_key: np.ndarray,
        inner: float,
        outer: float,
    ) -> np.ndarray:
        """Run color distance, alpha refinement, and despill on CUDA."""
        self._load_cuda()
        cp = self._cupy
        try:
            with cp.cuda.Device(self.gpu_index):
                pixels = cp.asarray(rgb, dtype=cp.float32)
                key = cp.asarray(sampled_key, dtype=cp.float32).reshape(1, 1, 3)
                delta = pixels - key
                distance = cp.sqrt(cp.sum(delta * delta, axis=2))
                alpha = cp.clip((distance - inner) / max(outer - inner, 1.0), 0.0, 1.0)

                # Match the CPU sequence: quantize -> 1 px erosion -> 0.8 px blur.
                alpha_u8 = cp.rint(alpha * 255.0).astype(cp.uint8)
                alpha_u8 = self._minimum_filter(alpha_u8, size=3, mode="nearest")
                alpha_blurred = self._gaussian_filter(alpha_u8.astype(cp.float32), sigma=0.8, mode="nearest")
                alpha_u8 = cp.rint(cp.clip(alpha_blurred, 0.0, 255.0)).astype(cp.uint8)
                alpha = alpha_u8.astype(cp.float32) / 255.0

                corrected = pixels.copy()
                edge_weight = cp.clip((1.0 - alpha) * 1.35, 0.0, 1.0)
                key_r, key_g, key_b = [float(value) for value in sampled_key]
                if key_g >= key_r and key_g >= key_b:
                    excess = cp.maximum(0.0, corrected[:, :, 1] - cp.maximum(corrected[:, :, 0], corrected[:, :, 2]))
                    corrected[:, :, 1] -= excess * edge_weight
                elif key_b >= key_r and key_b >= key_g:
                    excess = cp.maximum(0.0, corrected[:, :, 2] - cp.maximum(corrected[:, :, 0], corrected[:, :, 1]))
                    corrected[:, :, 2] -= excess * edge_weight
                else:
                    excess = cp.maximum(0.0, cp.minimum(corrected[:, :, 0], corrected[:, :, 2]) - corrected[:, :, 1])
                    corrected[:, :, 0] -= excess * edge_weight
                    corrected[:, :, 2] -= excess * edge_weight

                rgba = cp.empty((*rgb.shape[:2], 4), dtype=cp.uint8)
                rgba[:, :, :3] = cp.clip(corrected, 0, 255).astype(cp.uint8)
                rgba[:, :, 3] = alpha_u8
                result = cp.asnumpy(rgba)
                cp.cuda.get_current_stream().synchronize()
                return result
        except Exception as exc:
            raise CudaMattingError(f"CUDA matte failed: {exc}") from exc

    def run_matte(
        self,
        decision: BackendDecision,
        rgb: np.ndarray,
        sampled_key: np.ndarray,
        inner: float,
        outer: float,
        cpu_matte: Callable[[np.ndarray, np.ndarray, float, float], np.ndarray],
    ) -> np.ndarray:
        if decision.backend == "cuda":
            return self.matte_cuda(rgb, sampled_key, inner, outer)
        return cpu_matte(rgb, sampled_key, inner, outer)

    def diagnostics(self) -> dict[str, Any]:
        decision = self.decide()
        return {
            "requested_device": self.device,
            "selected_backend": decision.backend,
            "reason": decision.reason,
            "gpu_index": self.gpu_index,
            "gpu_name": self.gpu_name,
            "gpu_utilization": decision.gpu_utilization,
            "gpu_util_threshold": self.util_threshold,
            "cuda_disabled": self.cuda_disabled,
            "cuda_error": self.cuda_error,
        }
