# Community Gift Dynamic Matting Workflow

一个面向设计验证的 Windows 色键抠图工作流。它从 Pak20 Viewer 的 `index.html` 中读取 `workflow-data`，批量生成：

- 780×780 透明 PNG icon；
- 使用 Creative Agent 780×904 Gift Panel 模板生成“送礼物界面”静态 Mock；
- 从 Seedance 5 秒源视频首帧开始截取 3 秒，生成 780×1688、H.264、`yuv420p` 的 BG71 面板预览视频；
- 包含中英文切换的 `matting_demo.html` 对比页面；
- 逐行处理状态和输入哈希记录 `matting_demo_manifest.json`。

本仓库只包含处理脚本，不包含 Pak20 原始图片、视频、BG71 面板或任何生成结果。

## 定位

这是供设计师快速验证效果的原型，不是生产级抠图服务：

- 不调用生图 API；
- 不依赖 rembg 或 ONNX；支持 NVIDIA CUDA，也保留 CPU 回退；
- 支持绿色、蓝色和洋红色幕布；
- 单个素材失败时继续处理其他行；
- 永远不会覆盖输入图片或输入视频。

## 环境

- Windows 10/11
- 64 位 Python 3.12（Python 3.x 也可尝试）
- 首次运行时需要联网安装 Python 依赖

可选 GPU 环境：

- NVIDIA GPU 与可用的 `nvidia-smi`
- CUDA 12 兼容驱动
- `cupy-cuda12x[ctk]==14.1.1`（通过 pip 携带 CUDA 12 运行组件）

固定依赖：

```text
numpy==2.4.4
Pillow==12.2.0
av==17.0.0
```

GPU 版额外使用：

```text
cupy-cuda12x[ctk]==14.1.1
```

## 放入 Pak20 包

将本仓库文件复制到 Pak20 Viewer 根目录，并准备以下输入：

```text
Pak20 Viewer/
├─ index.html
├─ assets/
│  ├─ BG71-1000.png              # 必须为 780×1688
│  ├─ gift_panel_template.png     # Creative Agent 送礼面板，780×904
│  ├─ coin_icon.png               # Gift Panel 金币图标
│  └─ TikTokSans-Medium.ttf       # Gift Panel 字体
├─ workflow_viewer_assets/       # index.html 中 workflow-data 引用的素材
├─ tools/
│  ├─ process_matting_demo.py
│  └─ verify_matting_demo.py
├─ requirements-demo.txt
├─ requirements-gpu.txt
├─ RUN_MATTING_DEMO.bat
├─ RUN_MATTING_DEMO_GPU.bat
└─ START_WINDOWS_SERVER.bat
```

`index.html` 必须包含：

```html
<script id="workflow-data" type="application/json">...</script>
```

每一行数据应提供 `row_id`、`anchor_id`、Mainline 图片、Mainline 视频和 `key_color_hex`。脚本不维护第二份人工素材映射。

## 使用

双击：

```text
RUN_MATTING_DEMO.bat
```

NVIDIA CUDA 版本双击：

```text
RUN_MATTING_DEMO_GPU.bat
```

GPU 脚本会安装 CuPy、检查 CUDA，然后以 GPU 优先模式处理。GPU 缺失、占用过高或执行失败时会安全回退 CPU。

首次 GPU 安装会下载约 1.6 GB 的 CUDA 12 Python 运行组件；第一次执行还会编译并缓存少量 CUDA kernel，后续热启动明显更快。

首次运行会建立独立 `.venv` 并安装依赖。成功后生成：

```text
workflow_viewer_assets/matting_demo/icons/
workflow_viewer_assets/matting_demo/gift_panels/
workflow_viewer_assets/matting_demo/video_previews/
matting_demo_manifest.json
matting_demo.html
```

常用参数：

```bat
RUN_MATTING_DEMO.bat --rows 1-3,8
RUN_MATTING_DEMO.bat --rows 12 --force
RUN_MATTING_DEMO.bat --icons-only
```

命令行设备与编码参数：

```bat
.venv\Scripts\python.exe tools\process_matting_demo.py --device auto --encoder auto
.venv\Scripts\python.exe tools\process_matting_demo.py --device cuda --encoder nvenc --force
.venv\Scripts\python.exe tools\process_matting_demo.py --device cpu --encoder libx264 --force
```

- `auto`：GPU 利用率低于 50% 时走 CUDA，否则走 CPU。
- `cuda`：优先强制 CUDA；初始化或运行失败时仍会回退 CPU。
- CUDA 路径默认使用 NVENC；CPU 路径默认使用 `libx264`。
- 可通过 `MATTING_GPU_UTIL_THRESHOLD`、`MATTING_GPU_INDEX`、`MATTING_DEVICE` 和 `MATTING_VIDEO_ENCODER` 调整。

- 已存在的成功输出默认复用；处理逻辑更新后请加 `--force`。
- `--rows` 只处理指定行；随后不带 `--rows` 再运行一次，可重建包含全部行的 Viewer manifest。

启动本地和局域网预览：

```text
START_WINDOWS_SERVER.bat
```

默认地址：

```text
http://localhost:3001/matting_demo.html
http://<Windows IPv4>:3001/matting_demo.html
```

若 3001 已占用，启动脚本会显示 PID 并询问是否停止；不停止时会尝试 3002。

## 抠图原理

### 静态图片

1. 从画面边缘采样实际幕色，并与 manifest 声明色校验；
2. 对全画面计算 RGB 色距，而不是只删除边缘连通区域；
3. 生成软 alpha，并执行 1 px 收缩、0.8 px 羽化；
4. 针对绿、蓝、洋红分别执行 despill；
5. 按 alpha bbox 裁切，最大缩放到 720×720；
6. 放入 780×780 透明画布并底部居中；
7. 对手臂末端应用与视频相同的下半圆遮罩、60 px 向内羽化和左侧礼物棒保护，使透明 icon 及 Gift Panel 中的手臂切口更柔和。

全画面色键很重要：装饰、圆环或刀刃可能把幕布围成封闭区域。只做边缘连通删除会在少数运动帧中留下整块幕色。

### 视频

Seedance 输入仍可保持 5 秒；预览从第 0 帧开始，只取前 3 秒。每帧使用相同色键和边缘处理，再按原比例缩放到原 slot 尺寸的 67.2%（在 64% 版本上微调放大 5%），并在 780×780 透明层中整体向下移动 10%（78 px），最后定位到 BG71 的 `(0, 908)`。3 秒片段首尾各加入 0.3 秒淡入淡出，输出保持原始 FPS，静音编码为 H.264 MP4。

### 两个展示 Mock

- **送礼物界面**：复用 Creative Agent 的 `gift_panel_template.png`。透明 icon 按原 780×780 画布缩放至 123×123（在原 112 px 基础上放大 10%），中心仍固定在左上空白礼物位 `(106, 260)`；名称统一显示 `Community`，避免小语种字体缺字，演示价格固定为 BG71 档位对应的 1000 coins。
- **礼物送出界面**：复用 BG71-1000 直播间背景，展示处理后的 3 秒 H.264 动态视频。

Viewer 会把这两个最终 Mock 放在最上方并排展示，原始幕布图片、透明 icon 和原始幕布视频放在下方作为抠图证据。

为避免源素材手臂末端出现生硬直线，视频会根据首帧 alpha bbox 生成固定的下半圆遮罩：圆心位于主体高度约 72% 处，半径约为主体高度 29%，使下方弧线位置基本不变但曲率更平缓。圆边采用完全向内的 60 px smoothstep 羽化，使 Alpha 在抵达外弧前已降到 0；纵向使用 96 px 过渡。左下礼物棒区域完全保护，并通过 48 px 横向过渡进入右侧手臂遮罩。遮罩在整个片段中保持固定，避免逐帧追踪造成抖动；最终 MP4 不含 Alpha，被裁掉的区域直接显示 BG71。

## CUDA 调度

调度方式迁移自 Creative Agent 的 Windows/WSL GPU 保护思路：

1. 每个素材开始前通过 `nvidia-smi` 查询 GPU 利用率，结果缓存 5 秒；
2. `auto` 模式下 GPU 利用率达到阈值时直接走 CPU，避免与前台应用争抢；
3. GPU 色键、alpha 收缩/羽化和 despill 使用 CuPy/CUDA；
4. 同一进程只串行使用一个 GPU 路径，避免多个任务争抢 CUDA kernel；
5. 任一 CUDA 异常会拉黑本次 batch 的 GPU，当前素材用 CPU 重做，后续素材直接走 CPU；
6. 视频在 CUDA 路径优先使用 `h264_nvenc`，失败时整支视频用 CPU + `libx264` 重做。

检查调度结果：

```bat
.venv\Scripts\python.exe tools\check_cuda.py --device auto
```

## 验证

```bat
.venv\Scripts\python.exe tools\verify_matting_demo.py
```

验证器会检查 icon 尺寸和透明度、20 张 780×904 Gift Panel、视频编码/分辨率/帧率、固定 3 秒时长、20 行状态及输入哈希记录。

## 已知限制

- 色键抠图依赖幕布颜色与主体有足够区分度；主体本身含大量相同颜色时可能误抠。
- 毛发、半透明材质、运动模糊和强烈彩色溢出只能得到设计预览级结果。
- 当前 MP4 是带背景的预览，不是透明视频交付格式。
- BG71 面板是项目资产，需由使用者单独提供，本仓库不分发。
- GPU 加速主要改善色键计算和 H.264 编码，不会自动提升色键本身的抠图质量。
