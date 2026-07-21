"""Build the Pak20 Windows matting design demo.

This is deliberately a lightweight chroma-key preview pipeline.  It reads the
portable viewer's embedded workflow-data, never calls a model API, and never
overwrites the source images or videos.
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
from PIL import Image, ImageFilter

from gpu_scheduler import BackendDecision, CudaMattingError, MattingScheduler


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "index.html"
BG_PATH = ROOT / "assets" / "BG71-1000.png"
OUTPUT_ROOT = ROOT / "workflow_viewer_assets" / "matting_demo"
ICON_DIR = OUTPUT_ROOT / "icons"
VIDEO_DIR = OUTPUT_ROOT / "video_previews"
MANIFEST_PATH = ROOT / "matting_demo_manifest.json"
VIEWER_PATH = ROOT / "matting_demo.html"

ICON_SIZE = 780
ICON_CONTENT_SIZE = 720
ICON_BOTTOM_PADDING = 30
LIVE_WIDTH = 780
LIVE_HEIGHT = 1688
LIVE_SLOT_X = 0
LIVE_SLOT_Y = 908
FADE_SECONDS = 0.3
PREVIEW_DURATION_SECONDS = 3.0
LIVE_SUBJECT_SCALE = 0.672
ARM_CAP_START_RATIO = 0.72
ARM_CAP_RADIUS_RATIO = 0.29
ARM_CAP_FEATHER_PX = 60
ARM_CAP_VERTICAL_FEATHER_PX = 96
ARM_CAP_LEFT_PROTECT_RATIO = 0.55
ARM_CAP_LEFT_FEATHER_PX = 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Pak20 matting design demo.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing outputs.")
    parser.add_argument("--rows", default="", help="Optional rows such as 1-3,8.")
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


def parse_rows(value: str) -> set[int] | None:
    value = value.strip()
    if not value:
        return None
    rows: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid row range: {token}")
            rows.update(range(start, end + 1))
        else:
            rows.add(int(token))
    return rows


def load_workflow_items() -> list[dict[str, Any]]:
    html = INDEX_PATH.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="workflow-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError(f"workflow-data was not found in {INDEX_PATH}")
    data = json.loads(match.group(1))
    items = data.get("items") or []
    if not isinstance(items, list):
        raise ValueError("workflow-data.items must be a list")
    return sorted(items, key=lambda item: int(item.get("row_id") or 0))


def safe_asset_path(relative: str) -> Path:
    relative_path = Path(str(relative or ""))
    if not relative or relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe or empty asset path: {relative!r}")
    result = (ROOT / relative_path).resolve()
    if ROOT.resolve() not in result.parents:
        raise ValueError(f"asset escapes the demo root: {relative!r}")
    return result


def source_record(item: dict[str, Any]) -> dict[str, Any]:
    outputs = item.get("outputs") or {}
    handheld = outputs.get("handheld") or outputs.get("clean") or {}
    video = item.get("video") or {}
    chroma = (item.get("handheld") or {}).get("chroma_screen") or video.get("chroma_screen") or {}
    image_asset = str(handheld.get("asset") or "")
    video_asset = str(video.get("asset") or "")
    if not image_asset:
        raise ValueError("mainline image asset is missing")
    if not video_asset:
        raise ValueError("mainline video asset is missing")
    key_hex = str(chroma.get("key_color_hex") or "#00FF00").upper()
    if not re.fullmatch(r"#[0-9A-F]{6}", key_hex):
        raise ValueError(f"invalid key color: {key_hex!r}")
    return {
        "row_id": int(item.get("row_id") or 0),
        "anchor_id": str(item.get("anchor_id") or ""),
        "host_name": str(item.get("host_name") or ""),
        "community_name": str(item.get("community_name") or ""),
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
    clip_duration = min(PREVIEW_DURATION_SECONDS, duration) if duration > 0 else PREVIEW_DURATION_SECONDS
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
        "clip_target_seconds": PREVIEW_DURATION_SECONDS,
        "subject_scale": LIVE_SUBJECT_SCALE,
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
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


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


def build_viewer_html(manifest: dict[str, Any]) -> str:
    embedded = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Matting Preview</title>
  <style>
    :root{{--bg:#f5f6f8;--card:#fff;--ink:#171a21;--muted:#697386;--line:#dfe3ea;--accent:#6d45d9;--ok:#18864b;}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,"Segoe UI",Arial,sans-serif}}
    header{{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:18px;padding:16px 24px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}}
    header h1{{margin:0;font-size:20px}} .spacer{{flex:1}} a,button,select{{font:inherit}} a{{color:var(--accent);text-decoration:none}}
    button,select{{border:1px solid var(--line);background:#fff;border-radius:9px;padding:8px 11px;color:var(--ink)}} button{{cursor:pointer}} button:hover{{border-color:var(--accent)}}
    main{{max-width:1480px;margin:auto;padding:22px}} .toolbar{{display:flex;align-items:center;gap:10px;margin-bottom:18px}} .toolbar select{{min-width:280px}}
    .identity{{padding:16px 18px;background:var(--card);border:1px solid var(--line);border-radius:14px;margin-bottom:18px;display:flex;gap:16px;align-items:center}}
    .identity h2{{margin:0 0 3px;font-size:18px}} .meta{{color:var(--muted)}} .key{{display:inline-flex;align-items:center;gap:7px;padding:4px 8px;border:1px solid var(--line);border-radius:99px}}
    .swatch{{width:13px;height:13px;border-radius:50%;box-shadow:inset 0 0 0 1px rgba(0,0,0,.15)}}
    .grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}} .card{{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden}}
    .card h3{{margin:0;padding:13px 15px;border-bottom:1px solid var(--line);font-size:15px}} .media{{min-height:360px;display:flex;align-items:center;justify-content:center;background:#111;overflow:hidden}}
    .media.checker{{background-color:#fff;background-image:linear-gradient(45deg,#ddd 25%,transparent 25%),linear-gradient(-45deg,#ddd 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ddd 75%),linear-gradient(-45deg,transparent 75%,#ddd 75%);background-size:28px 28px;background-position:0 0,0 14px,14px -14px,-14px 0}}
    img,video{{display:block;max-width:100%;max-height:720px;object-fit:contain}} video{{width:100%;background:#111}} .status{{color:var(--ok);font-weight:600}}
    footer{{padding:22px;color:var(--muted);text-align:center}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}} header{{padding:12px}} main{{padding:12px}} .toolbar{{flex-wrap:wrap}} .toolbar select{{min-width:0;flex:1}}}}
  </style>
</head>
<body>
  <header>
    <h1 data-i18n="title">抠图效果预览</h1>
    <a href="index.html" data-i18n="review">返回结果审核台</a>
    <a href="workflow.html" data-i18n="workflow">工作流画布</a>
    <span class="spacer"></span>
    <button id="languageButton" type="button">English</button>
  </header>
  <main>
    <div class="toolbar">
      <button id="previousButton" type="button" data-i18n="previous">上一位</button>
      <select id="rowSelect" aria-label="Creator"></select>
      <button id="nextButton" type="button" data-i18n="next">下一位</button>
      <span class="status" id="rowStatus"></span>
    </div>
    <section class="identity">
      <div><h2 id="hostName"></h2><div class="meta" id="communityName"></div></div>
      <span class="spacer"></span>
      <span class="key"><i class="swatch" id="keySwatch"></i><span id="keyText"></span></span>
    </section>
    <section class="grid">
      <article class="card"><h3 data-i18n="originalImage">原始幕布图片</h3><div class="media"><img id="originalImage" alt=""></div></article>
      <article class="card"><h3 data-i18n="transparentIcon">透明 Icon · 780×780</h3><div class="media checker"><img id="iconImage" alt=""></div></article>
      <article class="card"><h3 data-i18n="originalVideo">原始幕布视频</h3><div class="media"><video id="originalVideo" controls muted playsinline preload="metadata"></video></div></article>
      <article class="card"><h3 data-i18n="panelVideo">BG71 直播间面板视频</h3><div class="media"><video id="previewVideo" controls muted playsinline preload="metadata"></video></div></article>
    </section>
  </main>
  <footer data-i18n="footer">Windows 设计验证版 · 色键抠图 · 原素材保持不变</footer>
  <script id="matting-data" type="application/json">{embedded}</script>
  <script>
    const DATA=JSON.parse(document.getElementById('matting-data').textContent); const rows=DATA.items;
    const TEXT={{zh:{{title:'抠图效果预览',review:'返回结果审核台',workflow:'工作流画布',previous:'上一位',next:'下一位',originalImage:'原始幕布图片',transparentIcon:'透明 Icon · 780×780',originalVideo:'原始幕布视频',panelVideo:'BG71 直播间面板视频',footer:'Seedance 源视频为 5 秒 · 面板预览取首帧开始的前 3 秒 · 原素材保持不变',success:'处理成功',failed:'处理失败',row:'第'}},en:{{title:'Matting Preview',review:'Back to Review',workflow:'Workflow Canvas',previous:'Previous',next:'Next',originalImage:'Original Chroma Image',transparentIcon:'Transparent Icon · 780×780',originalVideo:'Original Chroma Video',panelVideo:'BG71 Live Panel Video',footer:'Seedance source: 5s · Panel preview: first 3s from frame 0 · Original assets preserved',success:'Processed',failed:'Processing Failed',row:'Row'}}}};
    let language=localStorage.getItem('mattingDemoLanguage')==='en'?'en':'zh'; let current=0;
    const $=id=>document.getElementById(id); const rowSelect=$('rowSelect');
    function applyLanguage(){{document.documentElement.lang=language==='en'?'en':'zh-CN'; document.querySelectorAll('[data-i18n]').forEach(node=>node.textContent=TEXT[language][node.dataset.i18n]); $('languageButton').textContent=language==='zh'?'English':'中文'; document.title=TEXT[language].title; renderSelect(); render();}}
    function renderSelect(){{const value=String(current); rowSelect.innerHTML=rows.map((item,index)=>`<option value="${{index}}">${{TEXT[language].row}} ${{String(item.row_id).padStart(3,'0')}} · ${{escapeHtml(item.host_name||item.anchor_id)}}</option>`).join(''); rowSelect.value=value;}}
    function escapeHtml(value){{return String(value).replace(/[&<>"']/g,ch=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));}}
    function setVideo(element,source){{element.pause(); element.removeAttribute('src'); if(source)element.src=source; element.load();}}
    function render(){{if(!rows.length)return; const item=rows[current]; rowSelect.value=String(current); $('hostName').textContent=item.host_name||item.anchor_id; $('communityName').textContent=item.community_name||item.anchor_id; $('keySwatch').style.background=item.key_color_hex; $('keyText').textContent=item.key_color_hex; $('rowStatus').textContent=item.status==='succeeded'?TEXT[language].success:TEXT[language].failed; $('originalImage').src=item.input_image; $('iconImage').src=item.icon_path||''; setVideo($('originalVideo'),item.input_video); setVideo($('previewVideo'),item.preview_video||''); $('previousButton').disabled=current===0; $('nextButton').disabled=current===rows.length-1; history.replaceState(null,'',`?row=${{item.row_id}}`);}}
    rowSelect.addEventListener('change',()=>{{current=Number(rowSelect.value);render();}}); $('previousButton').addEventListener('click',()=>{{current=Math.max(0,current-1);renderSelect();render();}}); $('nextButton').addEventListener('click',()=>{{current=Math.min(rows.length-1,current+1);renderSelect();render();}}); $('languageButton').addEventListener('click',()=>{{language=language==='zh'?'en':'zh';localStorage.setItem('mattingDemoLanguage',language);applyLanguage();}});
    const requested=Number(new URLSearchParams(location.search).get('row')); const found=rows.findIndex(item=>item.row_id===requested); if(found>=0)current=found; applyLanguage();
  </script>
</body>
</html>
"""


def inject_home_link() -> None:
    html = INDEX_PATH.read_text(encoding="utf-8")
    marker = "matting-demo-entry"
    if marker in html:
        return
    entry = (
        '<a id="matting-demo-entry" href="matting_demo.html" '
        'style="position:fixed;right:22px;bottom:22px;z-index:9999;padding:10px 14px;'
        'border-radius:999px;background:#6d45d9;color:white;text-decoration:none;'
        'font:600 13px Segoe UI,Arial,sans-serif;box-shadow:0 8px 24px rgba(50,30,100,.28)">'
        'Matting Preview</a>'
    )
    if "</body>" not in html:
        raise ValueError("index.html does not contain a closing body tag")
    INDEX_PATH.write_text(html.replace("</body>", entry + "\n</body>", 1), encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")
    args = parse_args()
    selected_rows = parse_rows(args.rows)
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
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(INDEX_PATH)
    if not BG_PATH.is_file():
        raise FileNotFoundError(BG_PATH)
    with Image.open(BG_PATH) as background_source:
        background = background_source.convert("RGBA")
    if background.size != (LIVE_WIDTH, LIVE_HEIGHT):
        raise ValueError(f"BG71 must be {LIVE_WIDTH}x{LIVE_HEIGHT}, got {background.size}")

    ICON_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for item in load_workflow_items():
        try:
            record = source_record(item)
        except Exception as exc:
            print(f"[input] skipped malformed row: {exc}", file=sys.stderr)
            continue
        row = record["row_id"]
        if selected_rows is not None and row not in selected_rows:
            continue
        image_path = safe_asset_path(record["input_image"])
        video_path = safe_asset_path(record["input_video"])
        icon_path = ICON_DIR / f"row{row:03d}_mainline_icon_780.png"
        preview_path = VIDEO_DIR / f"row{row:03d}_mainline_bg71.mp4"
        print(f"[{row:03d}] {record['host_name']} - processing")
        result = {
            **record,
            "status": "running",
            "icon_path": rel(icon_path),
            "preview_video": "" if args.icons_only else rel(preview_path),
            "input_image_sha256": sha256_file(image_path),
            "input_video_sha256": sha256_file(video_path),
            "icon": {},
            "video": {},
            "errors": {},
        }
        if icon_path.is_file() and not args.force:
            result["icon"] = inspect_existing_icon(icon_path)
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
                    print(f"[{row:03d}] CUDA icon failed; retrying on CPU: {exc}", file=sys.stderr)
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
                print(f"[{row:03d}] icon failed: {exc}", file=sys.stderr)
        if args.icons_only:
            result["video"] = {"status": "skipped"}
        elif preview_path.is_file() and not args.force:
            result["video"] = inspect_existing_video(preview_path)
        else:
            try:
                video_decision = scheduler.decide()
                try:
                    video_details = process_video(
                        video_path,
                        preview_path,
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
                    print(f"[{row:03d}] CUDA/NVENC video failed; retrying on CPU/libx264: {exc}", file=sys.stderr)
                    video_decision = scheduler.decide()
                    video_details = process_video(
                        video_path,
                        preview_path,
                        record["key_color_hex"],
                        background,
                        scheduler,
                        video_decision,
                        "libx264",
                    )
                result["video"] = {"status": "succeeded", "reused": False, **video_details}
            except Exception as exc:
                preview_path.unlink(missing_ok=True)
                result["video"] = {"status": "failed"}
                result["errors"]["video"] = str(exc)
                print(f"[{row:03d}] video failed: {exc}", file=sys.stderr)
        required = [result["icon"]["status"]]
        if not args.icons_only:
            required.append(result["video"]["status"])
        result["status"] = "succeeded" if all(status == "succeeded" for status in required) else "failed"
        results.append(result)
        atomic_write_json(
            MANIFEST_PATH,
            {
                "schema_version": 1,
                "status": "running",
                "background": rel(BG_PATH),
                "scheduler": scheduler.diagnostics(),
                "items": results,
            },
        )

    succeeded = sum(item["status"] == "succeeded" for item in results)
    manifest = {
        "schema_version": 1,
        "status": "succeeded" if results and succeeded == len(results) else "partial",
        "design_prototype": True,
        "background": rel(BG_PATH),
        "scheduler": scheduler.diagnostics(),
        "icon_spec": {"width": ICON_SIZE, "height": ICON_SIZE, "content_max": ICON_CONTENT_SIZE, "anchor": "bottom_center"},
        "video_spec": {
            "width": LIVE_WIDTH,
            "height": LIVE_HEIGHT,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "clip_start_seconds": 0.0,
            "clip_duration_seconds": PREVIEW_DURATION_SECONDS,
            "fade_seconds": FADE_SECONDS,
            "subject_scale": LIVE_SUBJECT_SCALE,
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
    VIEWER_PATH.write_text(build_viewer_html(manifest), encoding="utf-8")
    inject_home_link()
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Viewer:   {VIEWER_PATH}")
    print(f"Success:  {succeeded}/{len(results)}")
    return 0 if manifest["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
