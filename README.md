# Pak20 Windows Matting Demo

一个面向设计验证的 Windows 色键抠图工作流。它从 Pak20 Viewer 的 `index.html` 中读取 `workflow-data`，批量生成：

- 780×780 透明 PNG icon；
- 780×1688、H.264、`yuv420p` 的 BG71 面板预览视频；
- 包含中英文切换的 `matting_demo.html` 对比页面；
- 逐行处理状态和输入哈希记录 `matting_demo_manifest.json`。

本仓库只包含处理脚本，不包含 Pak20 原始图片、视频、BG71 面板或任何生成结果。

## 定位

这是供设计师快速验证效果的原型，不是生产级抠图服务：

- 不调用生图 API；
- 不依赖 rembg、ONNX 或 GPU；
- 支持绿色、蓝色和洋红色幕布；
- 单个素材失败时继续处理其他行；
- 永远不会覆盖输入图片或输入视频。

## 环境

- Windows 10/11
- 64 位 Python 3.12（Python 3.x 也可尝试）
- 首次运行时需要联网安装 Python 依赖

固定依赖：

```text
numpy==2.4.4
Pillow==12.2.0
av==17.0.0
```

## 放入 Pak20 包

将本仓库文件复制到 Pak20 Viewer 根目录，并准备以下输入：

```text
Pak20 Viewer/
├─ index.html
├─ assets/
│  └─ BG71-1000.png              # 必须为 780×1688
├─ workflow_viewer_assets/       # index.html 中 workflow-data 引用的素材
├─ tools/
│  ├─ process_matting_demo.py
│  └─ verify_matting_demo.py
├─ requirements-demo.txt
├─ RUN_MATTING_DEMO.bat
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

首次运行会建立独立 `.venv` 并安装依赖。成功后生成：

```text
workflow_viewer_assets/matting_demo/icons/
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
6. 放入 780×780 透明画布并底部居中。

全画面色键很重要：装饰、圆环或刀刃可能把幕布围成封闭区域。只做边缘连通删除会在少数运动帧中留下整块幕色。

### 视频

每帧使用相同色键和边缘处理，再按原比例放入 780×780 透明层，定位到 BG71 的 `(0, 908)`。首尾各加入 0.3 秒淡入淡出，输出保持原始帧率和时长，静音编码为 H.264 MP4。

## 验证

```bat
.venv\Scripts\python.exe tools\verify_matting_demo.py
```

验证器会检查 icon 尺寸和透明度、视频编码/分辨率/帧率、20 行状态及输入哈希记录。

## 已知限制

- 色键抠图依赖幕布颜色与主体有足够区分度；主体本身含大量相同颜色时可能误抠。
- 毛发、半透明材质、运动模糊和强烈彩色溢出只能得到设计预览级结果。
- 当前 MP4 是带背景的预览，不是透明视频交付格式。
- BG71 面板是项目资产，需由使用者单独提供，本仓库不分发。

