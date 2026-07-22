# Community Gift Matting & Post-processing

一个与上层应用解耦的图片、视频抠图与后处理能力。它接收独立 JSON 素材清单，输出透明 PNG、合成预览视频、可选面板图和机器可读结果清单。

仓库只负责媒体处理，不包含上传页面、审核页面、网页生成、网络服务或部署逻辑。任何桌面端、研发平台或自动化任务都可以作为调用方；上层应用更新时，只要继续遵守输入/输出契约，这个处理层不需要同步修改。

## 能力边界

包含：

- 绿色、蓝色、洋红色幕布的图片与视频色键抠图；
- NVIDIA CUDA/CuPy 加速与 CPU 回退；
- 1 px alpha 收缩、0.8 px 羽化、对应幕色 despill；
- 静态 780×780 RGBA icon 规范化；
- 静态与动态素材共用的手臂圆弧遮罩；
- 视频取首帧开始的前 3 秒、淡入淡出和 H.264 合成；
- 参考 Gift Panel 与 BG71 后处理配置；
- 每项输入、输出、设备、耗时、alpha bbox、遮罩参数和错误记录。

不包含：

- 素材上传、账号、权限或审核逻辑；
- HTML、页面路由或前端组件；
- HTTP 服务、内网分享或云部署；
- 生图 API、素材生成或业务数据库。

## 输入契约

默认读取仓库根目录的 `matting_input.json`，也可以通过 `--input-manifest` 指定任意位置。相对媒体路径以输入清单所在目录为基准。

```json
{
  "schema_version": 1,
  "items": [
    {
      "id": "gift-001",
      "image": "inputs/gift-001.png",
      "video": "inputs/gift-001.mp4",
      "key_color_hex": "#00FF00"
    },
    {
      "id": "gift-002",
      "image": "inputs/gift-002.png",
      "video": "inputs/gift-002.mp4",
      "key_color_hex": "#FF00FF"
    }
  ]
}
```

字段：

- `id`：调用方提供的稳定唯一 ID，同时用于安全输出文件名；
- `image`：原始幕布图片；
- `video`：原始幕布视频；
- `key_color_hex`：`#RRGGBB` 格式的声明幕色。

每项目前需要同时提供图片和视频。处理不会覆盖、移动或删除输入文件。

## 输出契约

默认写入 `matting_outputs/`：

```text
matting_outputs/
├─ icons/
│  └─ gift-001_icon_780.png
├─ gift_panels/
│  └─ gift-001_gift_panel.png
├─ videos/
│  └─ gift-001_composited.mp4
└─ manifest.json
```

`manifest.json` 是唯一的程序集成入口，包含：

- 原输入路径与 SHA256；
- 每个媒体输出路径；
- 每一步的 `succeeded`、`failed` 或 `skipped` 状态；
- CUDA/CPU backend、编码器、耗时和错误；
- 图片 alpha bbox；
- 静态与动态手臂遮罩参数；
- 视频 FPS、帧数、时长、分辨率与像素格式。

调用方只需要等待命令结束并读取这个文件，不应依赖控制台文本或内部目录扫描。

## 后处理资源

部署环境需要在 `assets/` 提供：

```text
assets/
├─ BG71-1000.png              # 780×1688
├─ gift_panel_template.png    # 780×904
├─ coin_icon.png              # 20×20
└─ TikTokSans-Medium.ttf
```

这些是当前参考后处理 profile 的资源，不属于抠图算法本身。研发接入时可以保留该 profile、替换资源，或将合成步骤拆成独立 adapter；核心色键、alpha 与遮罩逻辑不依赖上层界面。

## Windows 使用

要求 64 位 Python 3.12。CPU 依赖：

```text
numpy==2.4.4
Pillow==12.2.0
av==17.0.0
```

双击 CPU/自动调度入口：

```text
RUN_MATTING.bat
```

NVIDIA GPU 入口：

```text
RUN_MATTING_GPU.bat
```

命令行：

```bat
.venv\Scripts\python.exe tools\process_matting.py ^
  --input-manifest matting_input.json ^
  --output-dir matting_outputs ^
  --device auto ^
  --encoder auto
```

常用参数：

```bat
.venv\Scripts\python.exe tools\process_matting.py --items gift-001,gift-002
.venv\Scripts\python.exe tools\process_matting.py --force
.venv\Scripts\python.exe tools\process_matting.py --icons-only
.venv\Scripts\python.exe tools\process_matting.py --device cuda --encoder nvenc
.venv\Scripts\python.exe tools\process_matting.py --device cpu --encoder libx264
```

- `auto`：GPU 利用率低于阈值时使用 CUDA，否则走 CPU；
- CUDA 初始化或执行失败时，本次任务自动回退 CPU；
- 已存在且输入 SHA256、幕色和处理参数一致的成功结果默认复用；同一 ID 上传新内容时会自动重做对应结果；
- 单项处理错误写入输出清单，不覆盖任何源素材。

## 算法摘要

### 静态图片

1. 从画面边缘采样真实幕色并校验声明色；
2. 对全画面计算 RGB 色距，处理主体包围的封闭幕色区域；
3. 生成软 alpha，并执行收缩、羽化与 despill；
4. 按 alpha bbox 裁切，等比缩放到最大 720×720；
5. 放入 780×780 RGBA 画布并底部居中；
6. 使用下半圆、60 px 向内羽化和左侧保护区柔化手臂末端。

### 视频

1. 从第 0 帧开始读取最多 3 秒；
2. 首帧确定幕色阈值与固定遮罩几何，后续帧复用以避免抖动；
3. 主体按 67.2% 放入 780×780 层并下移 78 px；
4. 应用相同圆弧遮罩、despill 和 0.3 秒首尾淡入淡出；
5. 合成至 780×1688 参考背景；
6. 输出 H.264、`yuv420p`、原 FPS、静音、faststart MP4。

## 调度

CUDA 路径使用 CuPy 执行色距、alpha、收缩、羽化和 despill，并优先使用 NVENC。调度器会读取 `nvidia-smi`：

- GPU 利用率达到阈值时选择 CPU；
- 同一批次出现 CUDA 错误后禁用 CUDA，后续项直接走 CPU；
- GPU 编码失败时整段视频以 CPU + `libx264` 重做；
- CPU 与 GPU 产生相同结构的结果清单。

检查设备选择：

```bat
.venv\Scripts\python.exe tools\check_cuda.py --device auto
```

## 验证

```bat
.venv\Scripts\python.exe tools\verify_matting.py ^
  --manifest matting_outputs\manifest.json
```

验证器检查透明度、尺寸、圆弧遮罩、面板合成、视频编码/FPS/帧数/3 秒时长及输出清单完整性。

## 已知限制

- 色键方案依赖幕色与主体有足够区分度；主体本身大面积包含同色时可能误抠；
- 毛发、半透明材质、运动模糊和强烈彩色溢出只能达到设计验证级质量；
- 当前视频交付是带参考背景的 MP4，不是带 alpha 的最终透明视频；
- 参考面板和背景是可替换的后处理 profile，部署方需要自行提供合法资源；
- GPU 主要改善吞吐和编码速度，不会改变色键算法的质量上限。
