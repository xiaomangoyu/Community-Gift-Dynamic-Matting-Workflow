"""Verify the portable Pak20 matting demo without modifying its assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duration_and_frames(path: Path) -> tuple[float, float, int, int, int]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            duration = float(stream.frames or 0) / fps
        frames = int(stream.frames or round(duration * fps))
        return duration, fps, frames, int(stream.width), int(stream.height)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "matting_demo_manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    items = manifest.get("items") or []
    if len(items) != 20:
        errors.append(f"expected 20 manifest items, got {len(items)}")
    key_counts: dict[str, int] = {}
    for item in items:
        row = int(item["row_id"])
        key = str(item["key_color_hex"])
        key_counts[key] = key_counts.get(key, 0) + 1
        image = ROOT / item["input_image"]
        video = ROOT / item["input_video"]
        icon = ROOT / item["icon_path"]
        preview = ROOT / item["preview_video"]
        for path in (image, video, icon, preview):
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"row {row:03d}: missing or empty {path.name}")
        if not icon.is_file() or not preview.is_file():
            continue
        with Image.open(icon) as icon_image:
            if icon_image.size != (780, 780) or icon_image.mode != "RGBA":
                errors.append(f"row {row:03d}: invalid icon {icon_image.mode} {icon_image.size}")
            alpha = np.asarray(icon_image.getchannel("A"))
            if not np.any(alpha > 16) or not np.any(alpha == 0):
                errors.append(f"row {row:03d}: icon alpha is empty or opaque")
            if any(int(alpha[y, x]) != 0 for x, y in ((0, 0), (779, 0), (0, 779), (779, 779))):
                errors.append(f"row {row:03d}: icon corners are not transparent")
        source_duration, source_fps, source_frames, _, _ = duration_and_frames(video)
        out_duration, out_fps, out_frames, out_width, out_height = duration_and_frames(preview)
        if (out_width, out_height) != (780, 1688):
            errors.append(f"row {row:03d}: preview is {out_width}x{out_height}")
        if abs(source_fps - out_fps) > 0.01 or abs(source_frames - out_frames) > 1:
            errors.append(f"row {row:03d}: fps/frame mismatch")
        if abs(source_duration - out_duration) > max(1 / source_fps, 0.05):
            errors.append(f"row {row:03d}: duration mismatch")
        if args.original_root:
            for relative in (item["input_image"], item["input_video"]):
                copied, original = ROOT / relative, args.original_root / relative
                if not original.is_file() or sha256_file(copied) != sha256_file(original):
                    errors.append(f"row {row:03d}: copied source differs: {relative}")
    report = {
        "status": "passed" if not errors else "failed",
        "items": len(items),
        "icons": len(list((ROOT / "workflow_viewer_assets" / "matting_demo" / "icons").glob("*.png"))),
        "video_previews": len(list((ROOT / "workflow_viewer_assets" / "matting_demo" / "video_previews").glob("*.mp4"))),
        "key_color_counts": key_counts,
        "source_hash_comparison": bool(args.original_root),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
