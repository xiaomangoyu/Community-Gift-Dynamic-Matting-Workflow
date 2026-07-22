"""Generic image/video chroma-key matting and post-processing pipeline.

The pipeline consumes a standalone JSON manifest, writes media outputs plus a
machine-readable result manifest, and never owns or generates a presentation
layer. Source images and videos are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from gpu_scheduler import BackendDecision, CudaMattingError, MattingScheduler


ROOT = Path(__file__).resolve().parents[1]
INPUT_MANIFEST_PATH = ROOT / "matting_input.json"
BG_PATH = ROOT / "assets" / "BG71-1000.png"
GIFT_PANEL_TEMPLATE_PATH = ROOT / "assets" / "gift_panel_template.png"
GIFT_PANEL_COIN_PATH = ROOT / "assets" / "coin_icon.png"
GIFT_PANEL_FONT_PATH = ROOT / "assets" / "TikTokSans-Medium.ttf"
OUTPUT_ROOT = ROOT / "matting_outputs"
ICON_DIR = OUTPUT_ROOT / "icons"
GIFT_PANEL_DIR = OUTPUT_ROOT / "gift_panels"
VIDEO_DIR = OUTPUT_ROOT / "videos"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

ICON_SIZE = 780
ICON_CONTENT_SIZE = 720
ICON_BOTTOM_PADDING = 30
GIFT_PANEL_SIZE = (780, 904)
GIFT_PANEL_SLOT_CENTER = (106, 260)
GIFT_PANEL_SLOT_MASK_SIZE = 112
GIFT_PANEL_ICON_SIZE = round(GIFT_PANEL_SLOT_MASK_SIZE * 1.10)
GIFT_PANEL_NAME = "Community"
GIFT_PANEL_FONT_SIZE = 18
GIFT_PANEL_LINE_HEIGHT = 24
GIFT_PANEL_NAME_MAX_WIDTH = 180
GIFT_PANEL_PRICE = 1000
GIFT_PANEL_TEXT_COLOR = (255, 255, 255, 191)
GIFT_PANEL_MASK_BOTTOM_TO_NAME = 16
GIFT_PANEL_COIN_SIZE = 20
GIFT_PANEL_COIN_TEXT_GAP = 4
LIVE_WIDTH = 780
LIVE_HEIGHT = 1688
LIVE_SLOT_X = 0
LIVE_SLOT_Y = 908
FADE_SECONDS = 0.3
OUTPUT_DURATION_SECONDS = 3.0
LIVE_SUBJECT_SCALE = 0.672
LIVE_SUBJECT_Y_OFFSET_RATIO = 0.10
LIVE_SUBJECT_Y_OFFSET_PX = round(ICON_SIZE * LIVE_SUBJECT_Y_OFFSET_RATIO)
ARM_CAP_START_RATIO = 0.72
ARM_CAP_RADIUS_RATIO = 0.29
ARM_CAP_FEATHER_PX = 60
ARM_CAP_VERTICAL_FEATHER_PX = 96
ARM_CAP_LEFT_PROTECT_RATIO = 0.55
ARM_CAP_LEFT_FEATHER_PX = 48
PIPELINE_VERSION = "generic-matting-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generic chroma-key matting and post-processing.")
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=INPUT_MANIFEST_PATH,
        help="Standalone JSON input manifest. Relative media paths resolve from its directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for processed media and manifest.json.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate existing outputs.")
    parser.add_argument("--items", default="", help="Optional comma-separated item IDs.")
    parser.add_argument("--icons-only", action="store_true", help="Skip video processing.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=os.getenv("MATTING_DEVICE", "auto"),
        help="Matting backend. auto follows GPU utilization and falls back to CPU.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=int(os.getenv("MATTING_GPU_INDEX", "0")),
        help="NVIDIA GPU index used by CuPy and nvidia-smi.",
    )
    parser.add_argument(
        "--gpu-util-threshold",
        type=int,
        default=int(os.getenv("MATTING_GPU_UTIL_THRESHOLD", "50")),
        help="auto mode uses CPU when GPU utilization is at or above this percentage.",
    )
    parser.add_argument(
        "--encoder",
        choices=("auto", "nvenc", "libx264"),
        default=os.getenv("MATTING_VIDEO_ENCODER", "auto"),
        help="Video encoder. auto uses NVENC with CUDA and libx264 with CPU.",
    )
    return parser.parse_args()


def parse_item_ids(value: str) -> set[str] | None:
    value = value.strip()
    if not value:
        return None
    return {token.strip() for token in value.split(",") if token.strip()}


def load_input_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items") or []
    if not isinstance(items, list):
        raise ValueError("input manifest items must be a list")
    return items


def resolve_media_path(value: str, manifest_dir: Path) -> Path:
    path = Path(str(value or ""))
    if not value:
        raise ValueError("media path is empty")
    return path.resolve() if path.is_absolute() else (manifest_dir / path).resolve()


def source_record(item: dict[str, Any], index: int) -> dict[str, Any]:
    item_id = str(item.get("id") or f"item-{index:03d}").strip()
    if not item_id:
        raise ValueError("item id is empty")
    file_id = re.sub(r"[^A-Za-z0-9._-]+", "-", item_id).strip("-._")
    if not file_id:
        raise ValueError(f"item id cannot form a safe filename: {item_id!r}")
    image_asset = str(item.get("image") or "")
    video_asset = str(item.get("video") or "")
    if not image_asset:
        raise ValueError("image is missing")
    if not video_asset:
        raise ValueError("video is missing")
    key_hex = str(item.get("key_color_hex") or "#00FF00").upper()
    if not re.fullmatch(r"#[0-9A-F]{6}", key_hex):
        raise ValueError(f"invalid key color: {key_hex!r}")
    return {
        "id": item_id,
        "file_id": file_id,
        "key_color_hex": key_hex,
        "input_image": image_asset,
        "input_video": video_asset,
    }


def rgb_from_hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def border_pixels(rgb: np.ndarray, width: int = 8) -> np.ndarray:
    height, image_width, _ = rgb.shape
    width = max(1, min(width, height // 4, image_width // 4))
    return np.concatenate(
        [
            rgb[:width].reshape(-1, 3),
            rgb[-width:].reshape(-1, 3),
            rgb[width:-width, :width].reshape(-1, 3),
            rgb[width:-width, -width:].reshape(-1, 3),
        ],
        axis=0,
    )


def tune_key(
    rgb: np.ndarray,
    declared_key: tuple[int, int, int],
) -> tuple[np.ndarray, float, float]:
    border = border_pixels(rgb).astype(np.float32)
    declared = np.array(declared_key, dtype=np.float32)
    sampled = np.median(border, axis=0)
    if float(np.linalg.norm(sampled - declared)) > 110:
        sampled = declared
    distances = np.linalg.norm(border - sampled, axis=1)
    p99 = float(np.percentile(distances, 99))
    inner = max(18.0, min(58.0, p99 * 1.8 + 8.0))
    outer = min(230.0, max(150.0, inner + 125.0))
    return sampled, inner, outer


def matte_rgba(
    rgb: np.ndarray,
    sampled_key: np.ndarray,
    inner: float,
    outer: float,
) -> np.ndarray:
    pixels = rgb.astype(np.float32)
    distance = np.linalg.norm(pixels - sampled_key.reshape(1, 1, 3), axis=2)
    # Chroma is a color contract, not a topology contract. Apply the key over
    # the complete frame so screen pixels enclosed by rings and cut-outs are
    # removed too. Upstream selects a safe green/blue/magenta key per asset.
    alpha = np.clip((distance - inner) / max(outer - inner, 1.0), 0.0, 1.0)

    alpha_image = Image.fromarray(np.round(alpha * 255).astype(np.uint8), mode="L")
    alpha_image = alpha_image.filter(ImageFilter.MinFilter(3))
    alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(0.8))
    alpha = np.asarray(alpha_image).astype(np.float32) / 255.0

    corrected = pixels.copy()
    edge_weight = np.clip((1.0 - alpha) * 1.35, 0.0, 1.0)
    key_r, key_g, key_b = [float(value) for value in sampled_key]
    if key_g >= key_r and key_g >= key_b:
        excess = np.maximum(0.0, corrected[:, :, 1] - np.maximum(corrected[:, :, 0], corrected[:, :, 2]))
        corrected[:, :, 1] -= excess * edge_weight
    elif key_b >= key_r and key_b >= key_g:
        excess = np.maximum(0.0, corrected[:, :, 2] - np.maximum(corrected[:, :, 0], corrected[:, :, 1]))
        corrected[:, :, 2] -= excess * edge_weight
    else:
        # Magenta spill is the shared red+blue excess above green.
        excess = np.maximum(0.0, np.minimum(corrected[:, :, 0], corrected[:, :, 2]) - corrected[:, :, 1])
        corrected[:, :, 0] -= excess * edge_weight
        corrected[:, :, 2] -= excess * edge_weight

    rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = np.clip(corrected, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.round(alpha * 255).astype(np.uint8)
    return rgba


def icon_from_rgba(rgba: np.ndarray) -> Image.Image:
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 16)
    if not len(xs):
        raise ValueError("alpha bbox is empty")
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    subject = Image.fromarray(rgba, mode="RGBA").crop((left, top, right, bottom))
    scale = min(ICON_CONTENT_SIZE / subject.width, ICON_CONTENT_SIZE / subject.height)
    target_size = (max(1, round(subject.width * scale)), max(1, round(subject.height * scale)))
    subject = subject.resize(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    x = (ICON_SIZE - subject.width) // 2
    y = ICON_SIZE - ICON_BOTTOM_PADDING - subject.height
    canvas.alpha_composite(subject, (x, max(0, y)))
    return canvas


def process_icon(
    source: Path,
    output: Path,
    key_hex: str,
    scheduler: MattingScheduler,
    decision: BackendDecision,
) -> dict[str, Any]:
    started = time.perf_counter()
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"))
    declared = rgb_from_hex(key_hex)
    sampled, inner, outer = tune_key(rgb, declared)
    rgba = scheduler.run_matte(decision, rgb, sampled, inner, outer, matte_rgba)
    icon = icon_from_rgba(rgba)
    arm_cap_mask = build_arm_cap_mask(np.asarray(icon.getchannel("A")))
    icon = apply_arm_cap_mask(icon, arm_cap_mask)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    icon.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, output)
    alpha = np.asarray(icon.getchannel("A"))
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "backend": decision.backend,
        "gpu_utilization": decision.gpu_utilization,
        "backend_reason": decision.reason,
        "sampled_key_rgb": [round(float(value), 2) for value in sampled],
        "key_inner": round(inner, 2),
        "key_outer": round(outer, 2),
        "alpha_bbox": alpha_bbox(alpha),
        "arm_cap_mask": arm_cap_mask,
    }


def alpha_bbox(alpha: np.ndarray) -> dict[str, int]:
    ys, xs = np.where(alpha > 16)
    if not len(xs):
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    x, y = int(xs.min()), int(ys.min())
    return {
        "x": x,
        "y": y,
        "width": int(xs.max()) - x + 1,
        "height": int(ys.max()) - y + 1,
    }


def inspect_existing_icon(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        return {
            "status": "succeeded",
            "reused": True,
            "backend": "reused",
            "elapsed_seconds": 0.0,
            "dimensions": {"width": rgba.width, "height": rgba.height},
            "alpha_bbox": alpha_bbox(alpha),
        }


def load_gift_panel_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if GIFT_PANEL_FONT_PATH.is_file():
        return ImageFont.truetype(str(GIFT_PANEL_FONT_PATH), GIFT_PANEL_FONT_SIZE)
    return ImageFont.load_default()


def truncate_panel_name(name: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> str:
    if font.getlength(name) <= GIFT_PANEL_NAME_MAX_WIDTH:
        return name
    suffix = "..."
    shortened = name
    while shortened and font.getlength(shortened + suffix) > GIFT_PANEL_NAME_MAX_WIDTH:
        shortened = shortened[:-1]
    return shortened + suffix if shortened else suffix


def process_gift_panel(icon_path: Path, output: Path, gift_name: str) -> dict[str, Any]:
    """Place the transparent icon in Creative Agent's 780x904 gift-store template."""
    started = time.perf_counter()
    with Image.open(GIFT_PANEL_TEMPLATE_PATH) as source:
        panel = source.convert("RGBA")
    if panel.size != GIFT_PANEL_SIZE:
        raise ValueError(f"gift panel template must be {GIFT_PANEL_SIZE}, got {panel.size}")

    with Image.open(icon_path) as source:
        gift = source.convert("RGBA").resize(
            (GIFT_PANEL_ICON_SIZE, GIFT_PANEL_ICON_SIZE),
            Image.Resampling.LANCZOS,
        )
    center_x, center_y = GIFT_PANEL_SLOT_CENTER
    panel.alpha_composite(
        gift,
        (
            center_x - GIFT_PANEL_ICON_SIZE // 2,
            center_y - GIFT_PANEL_ICON_SIZE // 2,
        ),
    )

    text_layer = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_layer)
    font = load_gift_panel_font()
    display_name = truncate_panel_name(gift_name.strip() or GIFT_PANEL_NAME, font)
    name_y = center_y + GIFT_PANEL_SLOT_MASK_SIZE // 2 + GIFT_PANEL_MASK_BOTTOM_TO_NAME
    draw.text(
        (center_x, name_y),
        display_name,
        fill=GIFT_PANEL_TEXT_COLOR,
        font=font,
        anchor="mt",
    )

    price_text = str(GIFT_PANEL_PRICE)
    price_width = font.getlength(price_text)
    price_y = name_y + GIFT_PANEL_LINE_HEIGHT
    coin = None
    if GIFT_PANEL_COIN_PATH.is_file():
        with Image.open(GIFT_PANEL_COIN_PATH) as source:
            coin = source.convert("RGBA").resize(
                (GIFT_PANEL_COIN_SIZE, GIFT_PANEL_COIN_SIZE),
                Image.Resampling.LANCZOS,
            )
    total_width = price_width
    if coin is not None:
        total_width += GIFT_PANEL_COIN_SIZE + GIFT_PANEL_COIN_TEXT_GAP
    group_x = center_x - total_width / 2
    if coin is not None:
        coin_y = price_y + (GIFT_PANEL_LINE_HEIGHT - GIFT_PANEL_COIN_SIZE) // 2
        text_layer.alpha_composite(coin, (round(group_x), coin_y))
        text_x = round(group_x) + GIFT_PANEL_COIN_SIZE + GIFT_PANEL_COIN_TEXT_GAP
    else:
        text_x = round(group_x)
    draw.text((text_x, price_y), price_text, fill=GIFT_PANEL_TEXT_COLOR, font=font)
    panel = Image.alpha_composite(panel, text_layer)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    panel.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, output)
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "dimensions": {"width": panel.width, "height": panel.height},
        "template": rel(GIFT_PANEL_TEMPLATE_PATH),
        "slot_center": {"x": center_x, "y": center_y},
        "icon_size": GIFT_PANEL_ICON_SIZE,
        "gift_name": display_name,
        "price": GIFT_PANEL_PRICE,
    }


def inspect_existing_gift_panel(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {
            "status": "succeeded",
            "reused": True,
            "elapsed_seconds": 0.0,
            "dimensions": {"width": image.width, "height": image.height},
            "template": rel(GIFT_PANEL_TEMPLATE_PATH),
            "slot_center": {"x": GIFT_PANEL_SLOT_CENTER[0], "y": GIFT_PANEL_SLOT_CENTER[1]},
            "icon_size": GIFT_PANEL_ICON_SIZE,
            "gift_name": GIFT_PANEL_NAME,
            "price": GIFT_PANEL_PRICE,
        }


def inspect_existing_video(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or Fraction(24, 1)
        fps = float(rate)
        duration = video_duration_seconds(container, stream)
        frame_count = int(stream.frames or round(duration * fps))
        return {
            "status": "succeeded",
            "reused": True,
            "backend": "reused",
            "elapsed_seconds": 0.0,
            "fps": round(fps, 4),
            "frame_count": frame_count,
            "duration_seconds": round(duration, 3),
            "dimensions": {"width": int(stream.width), "height": int(stream.height)},
            "codec": str(stream.codec_context.name or "h264"),
            "pixel_format": str(stream.codec_context.format.name if stream.codec_context.format else "yuv420p"),
        }


def fit_rgba(image: Image.Image, size: tuple[int, int], content_scale: float = 1.0) -> Image.Image:
    if not 0 < content_scale <= 1:
        raise ValueError("content_scale must be in the range (0, 1]")
    target_width = size[0] * content_scale
    target_height = size[1] * content_scale
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    canvas.alpha_composite(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def offset_rgba(image: Image.Image, *, x: int = 0, y: int = 0) -> Image.Image:
    canvas = Image.new("RGBA", image.size, (0, 0, 0, 0))
    canvas.alpha_composite(image, (x, y))
    return canvas


def build_arm_cap_mask(alpha: np.ndarray) -> dict[str, int]:
    """Build a fixed lower-arm semicircle from the first unfaded matte.

    Everything above center_y is untouched.  Below it, alpha is kept inside a
    downward-facing semicircle so the source frame's straight arm crop becomes
    a soft rounded cap.  The geometry remains fixed for the whole clip to avoid
    frame-to-frame jitter.
    """
    ys, xs = np.where(alpha > 16)
    if not len(xs):
        raise ValueError("cannot build arm cap from empty alpha")
    top, bottom = int(ys.min()), int(ys.max())
    height = max(1, bottom - top)
    center_y = int(round(top + height * ARM_CAP_START_RATIO))
    lower_xs = xs[ys >= center_y]
    center_x = int(round(float(np.median(lower_xs)))) if len(lower_xs) else int(round(float(np.median(xs))))
    radius = max(48, int(round(height * ARM_CAP_RADIUS_RATIO)))
    protect_left_until_x = center_x - int(round(radius * ARM_CAP_LEFT_PROTECT_RATIO))
    return {
        "center_x": center_x,
        "center_y": center_y,
        "radius": radius,
        "feather_px": ARM_CAP_FEATHER_PX,
        "vertical_feather_px": ARM_CAP_VERTICAL_FEATHER_PX,
        "protect_left_until_x": protect_left_until_x,
        "left_feather_px": ARM_CAP_LEFT_FEATHER_PX,
    }


def apply_arm_cap_mask(image: Image.Image, mask: dict[str, int]) -> Image.Image:
    rgba = np.asarray(image, dtype=np.uint8).copy()
    alpha = rgba[:, :, 3].astype(np.float32)
    yy, xx = np.ogrid[:alpha.shape[0], :alpha.shape[1]]
    center_x = mask["center_x"]
    center_y = mask["center_y"]
    radius = float(mask["radius"])
    feather = max(1.0, float(mask["feather_px"]))
    vertical_feather = max(1.0, float(mask["vertical_feather_px"]))
    protect_left_until_x = float(mask["protect_left_until_x"])
    left_feather = max(1.0, float(mask["left_feather_px"]))
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    def smoothstep(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    # Feather completely inward from the circle boundary. At the outer arc
    # alpha is already zero, so the source frame's hard bottom edge cannot
    # remain visible as a faint straight line.
    semicircle = smoothstep((radius - distance) / feather)
    transition = smoothstep((yy - center_y) / vertical_feather)
    left_transition = smoothstep((xx - protect_left_until_x) / left_feather)
    keep = 1.0 - transition * left_transition * (1.0 - semicircle)
    rgba[:, :, 3] = np.round(alpha * keep).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def video_duration_seconds(container: av.container.InputContainer, stream: av.video.stream.VideoStream) -> float:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    if stream.frames and stream.average_rate:
        return float(stream.frames / stream.average_rate)
    return 0.0


def select_encoder(requested: str, backend: str) -> str:
    if requested == "libx264":
        return "libx264"
    if requested == "nvenc" or (requested == "auto" and backend == "cuda"):
        try:
            av.codec.Codec("h264_nvenc", "w")
            return "h264_nvenc"
        except Exception:
            if requested == "nvenc":
                raise RuntimeError("h264_nvenc is not available in this PyAV/FFmpeg build")
    return "libx264"


def process_video(
    source: Path,
    output: Path,
    key_hex: str,
    background: Image.Image,
    scheduler: MattingScheduler,
    decision: BackendDecision,
    requested_encoder: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.mp4")
    temporary.unlink(missing_ok=True)
    input_container = av.open(str(source))
    input_stream = input_container.streams.video[0]
    rate = input_stream.average_rate or Fraction(24, 1)
    fps = float(rate)
    duration = video_duration_seconds(input_container, input_stream)
    clip_duration = min(OUTPUT_DURATION_SECONDS, duration) if duration > 0 else OUTPUT_DURATION_SECONDS
    clip_frame_limit = max(1, round(clip_duration * fps))
    output_container = av.open(str(temporary), mode="w", options={"movflags": "+faststart"})
    encoder = select_encoder(requested_encoder, decision.backend)
    output_stream = output_container.add_stream(encoder, rate=rate)
    output_stream.width = LIVE_WIDTH
    output_stream.height = LIVE_HEIGHT
    output_stream.pix_fmt = "yuv420p"
    if encoder == "h264_nvenc":
        output_stream.options = {"preset": "p4", "rc": "vbr", "cq": "20", "b": "0"}
    else:
        output_stream.options = {"crf": "20", "preset": "medium"}

    declared = rgb_from_hex(key_hex)
    sampled: np.ndarray | None = None
    arm_cap_mask: dict[str, int] | None = None
    inner = outer = 0.0
    frame_count = 0
    try:
        for frame in input_container.decode(input_stream):
            if frame_count >= clip_frame_limit:
                break
            rgb = frame.to_ndarray(format="rgb24")
            if sampled is None:
                sampled, inner, outer = tune_key(rgb, declared)
            rgba = scheduler.run_matte(decision, rgb, sampled, inner, outer, matte_rgba)
            elapsed = frame_count / fps
            remaining = max(0.0, clip_duration - ((frame_count + 1) / fps))
            fade = min(1.0, elapsed / FADE_SECONDS, remaining / FADE_SECONDS)
            layer = fit_rgba(
                Image.fromarray(rgba, mode="RGBA"),
                (ICON_SIZE, ICON_SIZE),
                content_scale=LIVE_SUBJECT_SCALE,
            )
            layer = offset_rgba(layer, y=LIVE_SUBJECT_Y_OFFSET_PX)
            if arm_cap_mask is None:
                arm_cap_mask = build_arm_cap_mask(np.asarray(layer.getchannel("A")))
            layer = apply_arm_cap_mask(layer, arm_cap_mask)
            layer_alpha = np.asarray(layer.getchannel("A"), dtype=np.float32)
            layer.putalpha(Image.fromarray(np.round(layer_alpha * max(0.0, fade)).astype(np.uint8), mode="L"))
            composite = background.copy()
            composite.alpha_composite(layer, (LIVE_SLOT_X, LIVE_SLOT_Y))
            output_frame = av.VideoFrame.from_ndarray(np.asarray(composite.convert("RGB")), format="rgb24")
            for packet in output_stream.encode(output_frame):
                output_container.mux(packet)
            frame_count += 1
        for packet in output_stream.encode():
            output_container.mux(packet)
    finally:
        input_container.close()
        output_container.close()
    if frame_count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("video contains no decodable frames")
    os.replace(temporary, output)
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "backend": decision.backend,
        "gpu_utilization": decision.gpu_utilization,
        "backend_reason": decision.reason,
        "encoder": encoder,
        "sampled_key_rgb": [round(float(value), 2) for value in sampled],
        "key_inner": round(inner, 2),
        "key_outer": round(outer, 2),
        "fps": round(fps, 4),
        "frame_count": frame_count,
        "duration_seconds": round(frame_count / fps, 3),
        "source_duration_seconds": round(duration, 3),
        "clip_start_seconds": 0.0,
        "clip_target_seconds": OUTPUT_DURATION_SECONDS,
        "subject_scale": LIVE_SUBJECT_SCALE,
        "subject_y_offset_px": LIVE_SUBJECT_Y_OFFSET_PX,
        "subject_y_offset_ratio": LIVE_SUBJECT_Y_OFFSET_RATIO,
        "arm_cap_mask": arm_cap_mask,
        "dimensions": {"width": LIVE_WIDTH, "height": LIVE_HEIGHT},
        "codec": "h264",
        "pixel_format": "yuv420p",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    global OUTPUT_ROOT, ICON_DIR, GIFT_PANEL_DIR, VIDEO_DIR, MANIFEST_PATH
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
    args = parse_args()
    input_manifest = args.input_manifest.resolve()
    OUTPUT_ROOT = args.output_dir.resolve()
    ICON_DIR = OUTPUT_ROOT / "icons"
    GIFT_PANEL_DIR = OUTPUT_ROOT / "gift_panels"
    VIDEO_DIR = OUTPUT_ROOT / "videos"
    MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
    selected_items = parse_item_ids(args.items)
    previous_items: dict[str, dict[str, Any]] = {}
    if MANIFEST_PATH.is_file():
        try:
            previous_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if previous_manifest.get("pipeline_version") == PIPELINE_VERSION:
                previous_items = {
                    str(item["id"]): item
                    for item in previous_manifest.get("items", [])
                    if item.get("id") is not None
                }
        except (OSError, ValueError, TypeError):
            previous_items = {}
    if not 0 <= args.gpu_util_threshold <= 100:
        raise ValueError("--gpu-util-threshold must be between 0 and 100")
    scheduler = MattingScheduler(
        device=args.device,
        gpu_index=args.gpu_index,
        util_threshold=args.gpu_util_threshold,
    )
    startup_diagnostics = scheduler.diagnostics()
    print(
        "Matting scheduler: "
        f"requested={args.device} selected={startup_diagnostics['selected_backend']} "
        f"gpu={startup_diagnostics['gpu_name'] or 'unavailable'} "
        f"util={startup_diagnostics['gpu_utilization']}% "
        f"reason={startup_diagnostics['reason']}"
    )
    if not input_manifest.is_file():
        raise FileNotFoundError(input_manifest)
    if not BG_PATH.is_file():
        raise FileNotFoundError(BG_PATH)
    if not GIFT_PANEL_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(GIFT_PANEL_TEMPLATE_PATH)
    with Image.open(BG_PATH) as background_source:
        background = background_source.convert("RGBA")
    if background.size != (LIVE_WIDTH, LIVE_HEIGHT):
        raise ValueError(f"BG71 must be {LIVE_WIDTH}x{LIVE_HEIGHT}, got {background.size}")

    ICON_DIR.mkdir(parents=True, exist_ok=True)
    GIFT_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, item in enumerate(load_input_items(input_manifest), start=1):
        try:
            record = source_record(item, index)
        except Exception as exc:
            print(f"[input] skipped malformed item: {exc}", file=sys.stderr)
            continue
        item_id = record["id"]
        file_id = record["file_id"]
        if selected_items is not None and item_id not in selected_items:
            continue
        image_path = resolve_media_path(record["input_image"], input_manifest.parent)
        video_path = resolve_media_path(record["input_video"], input_manifest.parent)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        icon_path = ICON_DIR / f"{file_id}_icon_780.png"
        gift_panel_path = GIFT_PANEL_DIR / f"{file_id}_gift_panel.png"
        output_video_path = VIDEO_DIR / f"{file_id}_composited.mp4"
        print(f"[{item_id}] processing")
        result = {
            **record,
            "status": "running",
            "icon_path": rel(icon_path),
            "gift_panel_path": rel(gift_panel_path),
            "video_path": "" if args.icons_only else rel(output_video_path),
            "input_image_sha256": sha256_file(image_path),
            "input_video_sha256": sha256_file(video_path),
            "icon": {},
            "gift_panel": {},
            "video": {},
            "errors": {},
        }
        previous_item = previous_items.get(item_id, {})
        image_is_unchanged = (
            previous_item.get("input_image_sha256") == result["input_image_sha256"]
            and previous_item.get("key_color_hex") == result["key_color_hex"]
        )
        video_is_unchanged = (
            previous_item.get("input_video_sha256") == result["input_video_sha256"]
            and previous_item.get("key_color_hex") == result["key_color_hex"]
        )
        previous_icon = previous_item.get("icon") or {}
        previous_icon_mask = previous_icon.get("arm_cap_mask") or {}
        icon_mask_is_current = all(
            int(previous_icon_mask.get(field, 0)) > 0
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
        if icon_path.is_file() and not args.force and image_is_unchanged and icon_mask_is_current:
            result["icon"] = {**previous_icon, "reused": True, "elapsed_seconds": 0.0}
        else:
            try:
                icon_decision = scheduler.decide()
                try:
                    icon_details = process_icon(
                        image_path,
                        icon_path,
                        record["key_color_hex"],
                        scheduler,
                        icon_decision,
                    )
                except CudaMattingError as exc:
                    scheduler.disable_cuda(str(exc))
                    print(f"[{item_id}] CUDA icon failed; retrying on CPU: {exc}", file=sys.stderr)
                    icon_decision = scheduler.decide()
                    icon_details = process_icon(
                        image_path,
                        icon_path,
                        record["key_color_hex"],
                        scheduler,
                        icon_decision,
                    )
                result["icon"] = {"status": "succeeded", "reused": False, **icon_details}
            except Exception as exc:
                result["icon"] = {"status": "failed"}
                result["errors"]["icon"] = str(exc)
                print(f"[{item_id}] icon failed: {exc}", file=sys.stderr)
        if result["icon"].get("status") != "succeeded":
            result["gift_panel"] = {"status": "skipped"}
        elif (
            gift_panel_path.is_file()
            and not args.force
            and result["icon"].get("reused") is True
            and int((previous_item.get("gift_panel") or {}).get("icon_size", 0)) == GIFT_PANEL_ICON_SIZE
            and (previous_item.get("gift_panel") or {}).get("gift_name") == GIFT_PANEL_NAME
        ):
            result["gift_panel"] = inspect_existing_gift_panel(gift_panel_path)
        else:
            try:
                panel_details = process_gift_panel(
                    icon_path,
                    gift_panel_path,
                    GIFT_PANEL_NAME,
                )
                result["gift_panel"] = {"status": "succeeded", "reused": False, **panel_details}
            except Exception as exc:
                gift_panel_path.unlink(missing_ok=True)
                result["gift_panel"] = {"status": "failed"}
                result["errors"]["gift_panel"] = str(exc)
                print(f"[{item_id}] gift panel failed: {exc}", file=sys.stderr)
        if args.icons_only:
            result["video"] = {"status": "skipped"}
        elif output_video_path.is_file() and not args.force and video_is_unchanged:
            previous_video = previous_item.get("video") or {}
            if previous_video.get("status") == "succeeded":
                result["video"] = {**previous_video, "reused": True, "elapsed_seconds": 0.0}
            else:
                result["video"] = inspect_existing_video(output_video_path)
        else:
            try:
                video_decision = scheduler.decide()
                try:
                    video_details = process_video(
                        video_path,
                        output_video_path,
                        record["key_color_hex"],
                        background,
                        scheduler,
                        video_decision,
                        args.encoder,
                    )
                except Exception as exc:
                    if video_decision.backend != "cuda":
                        raise
                    scheduler.disable_cuda(str(exc))
                    print(f"[{item_id}] CUDA/NVENC video failed; retrying on CPU/libx264: {exc}", file=sys.stderr)
                    video_decision = scheduler.decide()
                    video_details = process_video(
                        video_path,
                        output_video_path,
                        record["key_color_hex"],
                        background,
                        scheduler,
                        video_decision,
                        "libx264",
                    )
                result["video"] = {"status": "succeeded", "reused": False, **video_details}
            except Exception as exc:
                output_video_path.unlink(missing_ok=True)
                result["video"] = {"status": "failed"}
                result["errors"]["video"] = str(exc)
                print(f"[{item_id}] video failed: {exc}", file=sys.stderr)
        required = [result["icon"]["status"], result["gift_panel"]["status"]]
        if not args.icons_only:
            required.append(result["video"]["status"])
        result["status"] = "succeeded" if all(status == "succeeded" for status in required) else "failed"
        results.append(result)
        atomic_write_json(
            MANIFEST_PATH,
            {
                "schema_version": 2,
                "pipeline_version": PIPELINE_VERSION,
                "status": "running",
                "input_manifest": str(input_manifest),
                "background": rel(BG_PATH),
                "scheduler": scheduler.diagnostics(),
                "items": results,
            },
        )

    succeeded = sum(item["status"] == "succeeded" for item in results)
    manifest = {
        "schema_version": 2,
        "pipeline_version": PIPELINE_VERSION,
        "status": "succeeded" if results and succeeded == len(results) else "partial",
        "input_manifest": str(input_manifest),
        "background": rel(BG_PATH),
        "scheduler": scheduler.diagnostics(),
        "icon_spec": {
            "width": ICON_SIZE,
            "height": ICON_SIZE,
            "content_max": ICON_CONTENT_SIZE,
            "anchor": "bottom_center",
            "arm_cap_mask": {
                "enabled": True,
                "start_ratio": ARM_CAP_START_RATIO,
                "radius_ratio": ARM_CAP_RADIUS_RATIO,
                "feather_px": ARM_CAP_FEATHER_PX,
                "vertical_feather_px": ARM_CAP_VERTICAL_FEATHER_PX,
                "left_protect_ratio": ARM_CAP_LEFT_PROTECT_RATIO,
                "left_feather_px": ARM_CAP_LEFT_FEATHER_PX,
            },
        },
        "gift_panel_spec": {
            "width": GIFT_PANEL_SIZE[0],
            "height": GIFT_PANEL_SIZE[1],
            "template": rel(GIFT_PANEL_TEMPLATE_PATH),
            "slot_center": {"x": GIFT_PANEL_SLOT_CENTER[0], "y": GIFT_PANEL_SLOT_CENTER[1]},
            "icon_size": GIFT_PANEL_ICON_SIZE,
            "name_source": "fixed_label",
            "gift_name": GIFT_PANEL_NAME,
            "price": GIFT_PANEL_PRICE,
        },
        "video_spec": {
            "width": LIVE_WIDTH,
            "height": LIVE_HEIGHT,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "clip_start_seconds": 0.0,
            "clip_duration_seconds": OUTPUT_DURATION_SECONDS,
            "fade_seconds": FADE_SECONDS,
            "subject_scale": LIVE_SUBJECT_SCALE,
            "subject_y_offset_px": LIVE_SUBJECT_Y_OFFSET_PX,
            "subject_y_offset_ratio": LIVE_SUBJECT_Y_OFFSET_RATIO,
            "arm_cap_mask": {
                "enabled": True,
                "start_ratio": ARM_CAP_START_RATIO,
                "radius_ratio": ARM_CAP_RADIUS_RATIO,
                "feather_px": ARM_CAP_FEATHER_PX,
                "vertical_feather_px": ARM_CAP_VERTICAL_FEATHER_PX,
                "left_protect_ratio": ARM_CAP_LEFT_PROTECT_RATIO,
                "left_feather_px": ARM_CAP_LEFT_FEATHER_PX,
            },
            "slot": {"x": LIVE_SLOT_X, "y": LIVE_SLOT_Y},
        },
        "items": results,
    }
    atomic_write_json(MANIFEST_PATH, manifest)
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Success:  {succeeded}/{len(results)}")
    return 0 if manifest["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
