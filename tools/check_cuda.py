"""Print the CUDA scheduler decision without requiring a Pak20 asset package."""

from __future__ import annotations

import argparse
import json

from gpu_scheduler import MattingScheduler


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CUDA availability for the matting demo.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-util-threshold", type=int, default=50)
    args = parser.parse_args()
    scheduler = MattingScheduler(
        device=args.device,
        gpu_index=args.gpu_index,
        util_threshold=args.gpu_util_threshold,
    )
    result = scheduler.diagnostics()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["selected_backend"] == args.device or args.device == "auto" else 1


if __name__ == "__main__":
    raise SystemExit(main())

