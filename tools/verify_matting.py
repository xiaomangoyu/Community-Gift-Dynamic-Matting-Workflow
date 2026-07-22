"""Verify generic matting and post-processing outputs without changing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "matting_outputs" / "manifest.json"
EXPECTED_OUTPUT_SECONDS = 3.0
EXPECTED_SUBJECT_Y_OFFSET_PX = 78
EXPECTED_SUBJECT_Y_OFFSET_RATIO = 0.10


def resolve_recorded_path(value: str, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


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


def valid_arm_cap(mask: dict) -> bool:
    return all(
        int(mask.get(field, 0)) > 0
        for field in (
            "center_x",
            "center_y",
            "radius",
            "feather_px",
            "vertical_feather_px",
            "protect_left_until_x",
            "left_feather_px",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify matting output manifest and media files.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_manifest_path = Path(str(manifest.get("input_manifest") or ""))
    input_base = input_manifest_path.parent if input_manifest_path.is_absolute() else ROOT
    errors: list[str] = []
    if not manifest.get("pipeline_version"):
        errors.append("pipeline_version is missing")
    items = manifest.get("items") or []
    if not items:
        errors.append("output manifest contains no items")
    item_ids = [str(item.get("id") or "") for item in items]
    if any(not item_id for item_id in item_ids) or len(set(item_ids)) != len(item_ids):
        errors.append("item IDs must be present and unique")

    panel_template_path = ROOT / "assets" / "gift_panel_template.png"
    panel_template = None
    if panel_template_path.is_file():
        with Image.open(panel_template_path) as template_image:
            panel_template = np.asarray(template_image.convert("RGBA"))
    else:
        errors.append("gift panel template is missing")

    for item in items:
        item_id = str(item.get("id") or "unknown")
        image = resolve_recorded_path(str(item["input_image"]), input_base)
        video = resolve_recorded_path(str(item["input_video"]), input_base)
        icon = resolve_recorded_path(str(item["icon_path"]))
        gift_panel = resolve_recorded_path(str(item["gift_panel_path"]))
        output_video = resolve_recorded_path(str(item["video_path"]))
        for path in (image, video, icon, gift_panel, output_video):
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"{item_id}: missing or empty {path}")
        if not all(path.is_file() for path in (image, video, icon, gift_panel, output_video)):
            continue

        with Image.open(icon) as icon_image:
            if icon_image.size != (780, 780) or icon_image.mode != "RGBA":
                errors.append(f"{item_id}: invalid icon {icon_image.mode} {icon_image.size}")
            alpha = np.asarray(icon_image.getchannel("A"))
            if not np.any(alpha > 16) or not np.any(alpha == 0):
                errors.append(f"{item_id}: icon alpha is empty or opaque")
            if any(int(alpha[y, x]) != 0 for x, y in ((0, 0), (779, 0), (0, 779), (779, 779))):
                errors.append(f"{item_id}: icon corners are not transparent")

        icon_arm_cap = (item.get("icon") or {}).get("arm_cap_mask") or {}
        if not valid_arm_cap(icon_arm_cap):
            errors.append(f"{item_id}: missing or invalid icon arm cap mask")
        elif int(icon_arm_cap["center_y"]) + int(icon_arm_cap["radius"]) >= 780:
            errors.append(f"{item_id}: icon arm cap extends outside the 780px canvas")

        with Image.open(gift_panel) as panel_image:
            if panel_image.size != (780, 904) or panel_image.mode != "RGBA":
                errors.append(f"{item_id}: invalid gift panel {panel_image.mode} {panel_image.size}")
            elif panel_template is not None:
                rendered_panel = np.asarray(panel_image.convert("RGBA"))
                if np.array_equal(rendered_panel[190:390, 35:180], panel_template[190:390, 35:180]):
                    errors.append(f"{item_id}: gift icon was not composited into the panel slot")
        panel_meta = item.get("gift_panel") or {}
        if panel_meta.get("status") != "succeeded":
            errors.append(f"{item_id}: gift panel did not succeed")
        if panel_meta.get("gift_name") != "Community" or int(panel_meta.get("icon_size", 0)) != 123:
            errors.append(f"{item_id}: unexpected gift panel label or icon size")

        source_duration, source_fps, _, _, _ = duration_and_frames(video)
        out_duration, out_fps, out_frames, out_width, out_height = duration_and_frames(output_video)
        video_meta = item.get("video") or {}
        arm_cap = video_meta.get("arm_cap_mask") or {}
        if int(video_meta.get("subject_y_offset_px", -1)) != EXPECTED_SUBJECT_Y_OFFSET_PX:
            errors.append(f"{item_id}: subject vertical offset is not 78px")
        if abs(float(video_meta.get("subject_y_offset_ratio", -1)) - EXPECTED_SUBJECT_Y_OFFSET_RATIO) > 1e-6:
            errors.append(f"{item_id}: subject vertical offset ratio is not 10%")
        if not valid_arm_cap(arm_cap):
            errors.append(f"{item_id}: missing or invalid video arm cap mask")
        elif int(arm_cap["center_y"]) + int(arm_cap["radius"]) >= 780:
            errors.append(f"{item_id}: video arm cap extends outside the 780px slot")
        if (out_width, out_height) != (780, 1688):
            errors.append(f"{item_id}: output video is {out_width}x{out_height}")
        expected_duration = min(EXPECTED_OUTPUT_SECONDS, source_duration)
        expected_frames = round(expected_duration * source_fps)
        if abs(source_fps - out_fps) > 0.01 or abs(expected_frames - out_frames) > 1:
            errors.append(f"{item_id}: fps/frame mismatch")
        if abs(expected_duration - out_duration) > max(1 / source_fps, 0.05):
            errors.append(f"{item_id}: expected {expected_duration:.3f}s, got {out_duration:.3f}s")

    report = {
        "status": "passed" if not errors else "failed",
        "items": len(items),
        "icons": sum(1 for item in items if resolve_recorded_path(str(item.get("icon_path", ""))).is_file()),
        "gift_panels": sum(1 for item in items if resolve_recorded_path(str(item.get("gift_panel_path", ""))).is_file()),
        "videos": sum(1 for item in items if resolve_recorded_path(str(item.get("video_path", ""))).is_file()),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
