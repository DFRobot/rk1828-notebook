# -*- coding: utf-8 -*-
"""重建 rk1828-notebook 下全部 16 个 notebook。

用法: python3 _build_notebooks.py
依赖: /home/ubuntu/hugging_face/ 下的 qhddz HF 仓库（用于枚举每个仓库的文件列表，
     生成 notebook 第一步"下载模型"的 FILES 清单）。

每个 notebook 统一结构:
  0. 环境检查
  1. 下载模型（下拉框选择 HF / HF-Mirror，已存在的文件自动跳过）
  2. ... demo 具体步骤
"""
import json
import os

HF_DIR = os.environ.get("HF_DIR", "/home/ubuntu/hugging_face")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EXCLUDE = {".gitattributes", "README.md", "output.npy", "out.wav",
           "convert_left.jpg", "convert_right.jpg"}


# ---------------------------------------------------------------- 工具

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "cells": cells,
    }


def repo_files(repo, dest_prefix):
    """枚举 HF 仓库文件 -> (repo, remote, local) 列表。"""
    out = []
    root = os.path.join(HF_DIR, repo)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for fn in sorted(filenames):
            if fn in EXCLUDE:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            out.append((repo, rel, os.path.join(dest_prefix, rel)))
    return out


def files_literal(files, indent="    "):
    lines = [f'{indent}("{r}", "{rp}", "{lp}"),' for r, rp, lp in files]
    return "[\n" + "\n".join(lines) + "\n]"


# ---------------------------------------------------------------- 公共单元格

ENV_MD = md("""
## 0. 环境检查

先确认 RKNN3 运行库（`librknn3_api.so`）和 RK1828 设备都正常。
如失败，请先安装驱动/运行时（`rknn3-rk182x` DEB 包），并用 `sudo rknn-smi info` 确认能看到 RK1828。
""")

ENV_CODE = code("""
from rknn3 import get_lib, find_devices

_ = get_lib()  # 加载 /usr/lib/librknn3_api.so；失败说明 Runtime 未安装

devices = find_devices()
if not devices:
    raise RuntimeError("未发现 RK1828 设备！请检查: sudo rknn-smi info")
for dev_id, dev_type in devices:
    print(f"发现设备: id={dev_id}  接口={dev_type}")
""")


def download_cells(files, intro=None):
    """第一步：下载模型（下拉选择下载源 + 跳过已存在）。"""
    md_src = """
## 1. 下载模型

本 demo 的模型文件托管在 Hugging Face（`qhddz` 账号）。
**先用下拉框选择下载源**，再运行下一个单元格开始下载：

- **HF-Mirror 国内镜像**：`hf-mirror.com`，国内直连，推荐
- **Hugging Face 官方**：`huggingface.co`，海外或已有代理时用

> 已存在（且非空）的文件会**自动跳过**，重复运行不会重新下载。
"""
    if intro:
        md_src += "\n\n" + intro
    return [
        md(md_src),
        code(f"""
from nb_download import source_picker

# (HF仓库名, 仓库内路径, 本地路径)
FILES = {files_literal(files)}

w_source = source_picker()
"""),
        code("""
from nb_download import ensure_files

ensure_files(FILES, w_source.value)
"""),
    ]


EXTRACT_MD = md("""
## 2. 解压预编译 demo

`demo_x86_64.tar.gz` 里是官方流程编译好的 x86_64 C++ demo 和运行时库（`lib/`）。
HF 不允许直接托管可执行文件，所以打包成 tar.gz，解压一次即可（重复运行自动跳过）。
""")


def extract_cells(demo_dirs):
    body = "\n".join(f'extract_tarball("{d}/demo_x86_64.tar.gz", "{d}")'
                     for d in demo_dirs)
    dirs_literal = "{" + ", ".join(f'"{d}"' for d in demo_dirs) + "}"
    return [
        EXTRACT_MD,
        code(f"""
from nb_download import extract_tarball

{body}
"""),
    ]


def run_helper_code():
    return code("""
from nb_demo import run_demo, snapshot, new_files, show_images, show_new_images, show_audio
""")


# ================================================================ 01 分类

def build_01():
    files = []
    for m in ("mobilenet_v2", "resnet50_v2"):
        repo = f"{m}-rknn3-rk1828"
        for q in ("i8", "fp16"):
            for ext in ("rknn", "weight"):
                files.append((repo, f"{q}/{m}.{ext}", f"models/cnn/{m}/{q}/{m}.{ext}"))
        files.append((repo, "imagenet_labels.txt", "datasets/imagenet_labels.txt"))
        files.append((repo, "dog_224x224.jpg", "datasets/dog_224x224.jpg"))

    cells = [
        md("""
# RK1828 图像分类 Demo（MobileNetV2 / ResNet50-v2）

本 notebook 在 **RK1828 AI 协处理器**（PCIe 连接，20 TOPS NPU）上运行 ImageNet 图像分类，
完整演示 AI 推理的标准流程：

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型（可选 HF / HF-Mirror 源） |
| 2 | 选择模型和图片 |
| 3 | 前处理（把图片变成模型能吃的张量） |
| 4 | RKNN 推理（在 NPU 上运行） |
| 5 | 后处理（反量化 + Softmax，输出 Top-5） |

> 小知识：**i8** 是把权重压成 8bit 整数的量化版本，更小更快；
> **fp16** 是 16bit 浮点版本。两种都可以跑一遍对比结果。
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(
        files,
        intro="本 demo 会下载 MobileNetV2 和 ResNet50-v2 两个模型（各 i8/fp16 两版），"
              "以及 ImageNet 标签和示例图片。")
    cells += [
        md("""
## 2. 选择模型和图片

- **模型**：`mobilenet_v2`（轻量，3.4M 参数）或 `resnet50_v2`（经典，25M 参数）
- **量化版本**：`i8`（int8 量化）或 `fp16`（半精度浮点）
- **图片**：`datasets/` 里的示例图（也可以自己放 224×224 左右的图片进去）
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(options=["mobilenet_v2", "resnet50_v2"],
                           description="模型:")
w_quant = widgets.Dropdown(options=["i8", "fp16"], description="量化:")
w_image = widgets.Dropdown(options=sorted(glob.glob("datasets/*.jpg")),
                           description="图片:", layout=widgets.Layout(width="60%"))
display(w_model, w_quant, w_image)

LABELS = open("datasets/imagenet_labels.txt").read().splitlines()
print(f"{len(LABELS)} 个类别。选好后继续往下运行 ▼")
"""),
        md("""
## 3. 前处理

模型输入是 **224×224 NHWC uint8 RGB**（归一化已烘焙进模型，无需减均值除方差）。
我们把图片缩放到 224×224 并加上 batch 维度。
"""),
        code("""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from vision_post import setup_cjk_font

setup_cjk_font()
img_orig = Image.open(w_image.value).convert("RGB")
img = img_orig.resize((224, 224))
x = np.asarray(img)[None]  # [1,224,224,3] uint8 NHWC
print("输入张量:", x.shape, x.dtype)

# 前处理每个阶段都画出来：原图 -> resize -> 输入张量
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, im, title in [
    (axes[0], img_orig, f"1. 原图 {img_orig.size[0]}x{img_orig.size[1]}"),
    (axes[1], img, "2. resize 224x224"),
    (axes[2], x[0], f"3. 模型输入 {x.shape}\\nuint8 NHWC"),
]:
    ax.imshow(np.asarray(im))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## 4. RKNN 推理

加载 `.rknn`（计算图）+ `.weight`（权重）到 RK1828，执行一次推理并计时。
"""),
        code("""
import time
from rknn3 import Rknn3Model

model_path = f"models/cnn/{w_model.value}/{w_quant.value}/{w_model.value}.rknn"
model = Rknn3Model(model_path)
print("输入属性:", model.input_attrs)
print("输出属性:", model.output_attrs)

_ = model.infer([x])  # 预热
t0 = time.time()
outputs = model.infer([x])
print(f"推理耗时: {(time.time()-t0)*1000:.1f} ms")
"""),
        md("""
## 5. 后处理：反量化 + Softmax

i8 模型的输出是 int8，要用输出张量自带的 `scale` / `zero_point` 还原成浮点 logits；
再 Softmax 取 Top-5。
"""),
        code("""
from vision_post import dequant

attr = model.output_attrs[0]
logits = dequant(outputs[0], attr).ravel()
probs = np.exp(logits - logits.max())
probs /= probs.sum()

top5 = probs.argsort()[::-1][:5]
for i in top5:
    print(f"{probs[i]*100:5.1f}%  {LABELS[i]}")
"""),
        code("""
import matplotlib.pyplot as plt
from vision_post import setup_cjk_font

setup_cjk_font()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 1.6]})
axes[0].imshow(img)
axes[0].set_title(f"{w_model.value} ({w_quant.value})")
axes[0].axis("off")
names = [LABELS[i] for i in top5][::-1]
axes[1].barh(names, [probs[i] for i in top5][::-1], color="#3b82f6")
axes[1].set_title("Top-5 概率")
axes[1].set_xlim(0, 1)
plt.tight_layout()
plt.show()

model.close()
"""),
        md("""
## 小结

- 完整流程：**下载模型 → 前处理 → NPU 推理 → 反量化/Softmax → Top-5**
- 可以回到第 2 步换模型/量化版本，对比 i8 与 fp16 的 Top-1 是否一致、耗时差多少。
"""),
    ]
    return notebook(cells)


# ================================================================ 02 YOLO

def build_02():
    repo = "yolo-rknn3-rk1828"
    files = []
    for m in ("yolov5", "yolov6", "yolov8", "yolo26n", "yolo26s", "yolo26m"):
        for q in ("i8", "fp16"):
            for ext in ("rknn", "weight"):
                files.append((repo, f"{m}/{q}/{m}.{ext}", f"models/yolo/{m}/{q}/{m}.{ext}"))
    files.append(("yolo26-segment-rknn3-rk1828", "model/coco_80_labels_list.txt",
                  "datasets/coco_80_labels_list.txt"))
    files.append(("yolo26-segment-rknn3-rk1828", "model/bus.jpg", "datasets/bus.jpg"))
    files.append(("mobilenet_v2-rknn3-rk1828", "dog_224x224.jpg",
                  "datasets/dog_224x224.jpg"))

    cells = [
        md("""
# RK1828 YOLO 目标检测 Demo（YOLOv5/v6/v8 + YOLO26）

一次跑通 6 个 YOLO 模型，理解两类输出格式：

| 模型 | 输出 | 后处理 |
|---|---|---|
| YOLOv5 / v6 / v8 | `[1,K,6]`（score, class, x1, y1, x2, y2） | 框解码 + NMS 已在模型内完成，只需坐标还原 |
| YOLO26 (n/s/m) | 6 个原始检测头（stride 8/16/32 的 box + score） | host 端 sigmoid + 框解码 + 按类 NMS |

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型（可选 HF / HF-Mirror 源） |
| 2 | 选择模型和图片 |
| 3 | 前处理（letterbox 等比缩放 + 灰边填充） |
| 4 | RKNN 推理 |
| 5 | 后处理与可视化 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(
        files,
        intro="本 demo 会下载 YOLOv5/v6/v8 和 YOLO26 n/s/m（各 i8/fp16 两版），"
              "以及 COCO 80 类标签和示例图片。")
    cells += [
        md("""
## 2. 选择模型和图片

模型文件在 `models/yolo/<名称>/<量化>/` 下；图片可以换成自己的。

**各模型的实际规格**（从权重体积判定）：

| 目录名 | 实际规格 | 参数量 | 备注 |
|---|---|---|---|
| `yolov5` | **YOLOv5s** | 7.2M | i8/fp16 目录与内容一致 |
| `yolov6` | **YOLOv6s** | 17.2M | ⚠️ `i8`/`fp16` 目录**疑似标反**（"i8"目录的权重更大且更慢，实测如此） |
| `yolov8` | **YOLOv8s** | 11.2M | ⚠️ 同上，`i8`/`fp16` 目录**疑似标反** |
| `yolo26n/s/m` | YOLO26n / 26s / 26m | 2.4M / 9.5M / 20.4M | 目录与内容一致 |

> v6/v8 的目录问题来自瑞芯微官方 SDK 预转换包本身（i8 版本内嵌的 NMS
> 后处理头以 fp16 存储，导致体积更大、速度更慢），选择时以实测为准。
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(
    options=["yolov5", "yolov6", "yolov8", "yolo26n", "yolo26s", "yolo26m"],
    value="yolov8", description="模型:")
w_quant = widgets.Dropdown(options=["i8", "fp16"], description="量化:")
w_image = widgets.Dropdown(options=sorted(glob.glob("datasets/*.jpg")),
                           description="图片:", layout=widgets.Layout(width="60%"))
display(w_model, w_quant, w_image)

LABELS = open("datasets/coco_80_labels_list.txt").read().splitlines()
print(f"{len(LABELS)} 个 COCO 类别。选好后继续往下运行 ▼")
"""),
        md("""
## 3. 前处理：letterbox

YOLO 要求 640×640 正方形输入。直接拉伸会改变物体长宽比，所以**等比缩放后灰边填充**，
并记下缩放比例和 padding —— 后处理时要用它们把框还原回原图坐标。
"""),
        code("""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from vision_post import letterbox, to_input_array, setup_cjk_font

setup_cjk_font()
img = Image.open(w_image.value).convert("RGB")
img_lb, SCALE, PAD_X, PAD_Y = letterbox(img, 640)
x = to_input_array(img_lb)
print(f"原图 {img.size} -> 输入 {x.shape}, scale={SCALE:.3f}, pad=({PAD_X},{PAD_Y})")

# 前处理每个阶段都画出来：原图 -> letterbox -> 输入张量
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, im, title in [
    (axes[0], img, f"1. 原图 {img.size[0]}x{img.size[1]}"),
    (axes[1], img_lb, "2. letterbox 640x640\\n(等比缩放 + 灰边填充)"),
    (axes[2], x[0], f"3. 模型输入 {x.shape}\\nuint8 NHWC"),
]:
    ax.imshow(np.asarray(im))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## 4. RKNN 推理
"""),
        code("""
import time
from rknn3 import Rknn3Model

model_path = f"models/yolo/{w_model.value}/{w_quant.value}/{w_model.value}.rknn"
model = Rknn3Model(model_path)
print(f"{len(model.output_attrs)} 个输出:")
for a in model.output_attrs:
    print("  ", a)

_ = model.infer([x])
t0 = time.time()
outputs = model.infer([x])
print(f"推理耗时: {(time.time()-t0)*1000:.1f} ms")
"""),
        md("""
## 5. 后处理

- **YOLOv5/v6/v8**：输出已经是 NMS 后的 `(score, class, x1, y1, x2, y2)`，按置信度过滤即可
- **YOLO26**：6 个检测头（3 个 box 头 `[1,4,h,w]` + 3 个 score 头 `[1,80,h,w]`），
  在 host 端做 sigmoid、框解码和按类 NMS；i8 输出还要先反量化
"""),
        code("""
from vision_post import (decode_builtin_nms, decode_yolo26_heads,
                         unletterbox_boxes)

if w_model.value.startswith("yolo26"):
    boxes, classes, scores = decode_yolo26_heads(outputs, model.output_attrs,
                                                 conf_thresh=0.4)
else:
    boxes, classes, scores = decode_builtin_nms(outputs[0], conf_thresh=0.4)

boxes = unletterbox_boxes(boxes, SCALE, PAD_X, PAD_Y, *img.size)
print(f"检测到 {len(boxes)} 个目标:")
for b, c, s in zip(boxes, classes, scores):
    print(f"  {LABELS[c]:12s} {s:.2f}  box=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")
"""),
        md("""
## 6. 可视化
"""),
        code("""
from vision_post import draw_boxes

result = draw_boxes(img, boxes, classes, scores, LABELS)
model.close()
result
"""),
        md("""
## 7. 多模型对比（i8 + fp16 全量）

同一张图，把 **6 个模型 × 2 种量化（i8 / fp16）共 12 组**全部跑一遍。
运行过程实时打印进度；全部完成后输出对比表格、耗时柱状图和检测效果拼图。
每个模型计时 3 次取最快（排除首次预热抖动）。
"""),
        code("""
import time
from rknn3 import Rknn3Model
from vision_post import (decode_builtin_nms, decode_yolo26_heads,
                         unletterbox_boxes)

# 目录名 -> 实际规格: yolov5=YOLOv5s, yolov6=YOLOv6s, yolov8=YOLOv8s
# 注意: yolov6/yolov8 的 i8 与 fp16 目录疑似标反（官方 SDK 包如此，见上文表格）
ALL_MODELS = ["yolov5", "yolov6", "yolov8", "yolo26n", "yolo26s", "yolo26m"]
QUANTS = ["i8", "fp16"]

compare = []  # (model, quant, latency_ms, boxes, classes, scores)
total = len(ALL_MODELS) * len(QUANTS)
step = 0
for name in ALL_MODELS:
    for quant in QUANTS:
        step += 1
        print(f"[{step:2d}/{total}] 加载 {name} ({quant}) ...", flush=True)
        m = Rknn3Model(f"models/yolo/{name}/{quant}/{name}.rknn")
        m.infer([x])  # 预热
        best, outs = float("inf"), None
        for _ in range(3):
            t0 = time.time()
            outs = m.infer([x])
            best = min(best, (time.time() - t0) * 1000)
        if name.startswith("yolo26"):
            b, c, s = decode_yolo26_heads(outs, m.output_attrs, conf_thresh=0.4)
        else:
            b, c, s = decode_builtin_nms(outs[0], conf_thresh=0.4)
        b = unletterbox_boxes(b, SCALE, PAD_X, PAD_Y, *img.size)
        compare.append((name, quant, best, b, c, s))
        m.close()
        print(f"      -> {best:6.1f} ms, {len(b)} 个目标", flush=True)
print("\\n全部完成 ✅", flush=True)
"""),
        md("""
### 对比表格
"""),
        code("""
from IPython.display import HTML

rows = "".join(
    f"<tr><td>{n}</td><td>{q}</td>"
    f"<td>{lat:.1f}</td><td>{len(b)}</td>"
    f"<td>{', '.join(f'{LABELS[ci]}({si:.2f})' for ci, si in list(zip(c, s))[:5]) or '-'}</td></tr>"
    for n, q, lat, b, c, s in compare)
HTML("<table style='text-align:left'>"
     "<tr><th>模型</th><th>量化</th><th>耗时(ms)</th><th>目标数</th><th>Top-5 检测</th></tr>"
     + rows + "</table>")
"""),
        md("""
### 耗时柱状图（i8 vs fp16）
"""),
        code("""
import numpy as np
import matplotlib.pyplot as plt
from vision_post import setup_cjk_font

setup_cjk_font()
colors = {"i8": "#3b82f6", "fp16": "#f59e0b"}
xs = np.arange(len(ALL_MODELS))
width = 0.38
fig, ax = plt.subplots(figsize=(10, 4.5))
for k, q in enumerate(QUANTS):
    lats = [lat for n, qq, lat, *_ in compare if qq == q]
    cnts = [len(b) for n, qq, _, b, *_ in compare if qq == q]
    bars = ax.bar(xs + (k - 0.5) * width, lats, width, label=q, color=colors[q])
    for bar, lat, cnt in zip(bars, lats, cnts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{lat:.0f}ms\\n{cnt}个", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs, ALL_MODELS)
ax.set_ylabel("推理耗时 (ms)")
ax.set_title("YOLO 多模型对比（同一张图，i8 vs fp16）")
ax.set_ylim(0, max(lat for _, _, lat, *_ in compare) * 1.3)
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""
### 检测效果拼图（上排 i8，下排 fp16）
"""),
        code("""
import matplotlib.pyplot as plt
from vision_post import draw_boxes

fig, axes = plt.subplots(2, 6, figsize=(20, 7))
for k, q in enumerate(QUANTS):
    row = [(n, lat, b, c, s) for n, qq, lat, b, c, s in compare if qq == q]
    for ax, (n, lat, b, c, s) in zip(axes[k], row):
        ax.imshow(draw_boxes(img, b, c, s, LABELS))
        ax.set_title(f"{n} ({q})  {lat:.0f}ms  {len(b)}个", fontsize=10)
        ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## 小结

- YOLOv5/v6/v8 把解码和 NMS 做进了模型，host 端几乎零后处理；
  YOLO26 输出原始检测头换取部署灵活性，后处理在 host 完成（见 `vision_post.decode_yolo26_heads`）。
- 一般规律：i8 比 fp16 快，模型越大越慢但检测更全；
  官方参考 mAP@0.5:0.95: 26n 40.9 / 26s 48.6 / 26m 53.1（PyTorch FP32）。
- 模型规格：v5/v6/v8 是瑞芯微官方 SDK 的预转换模型，按家族名发布。
  从权重体积判定均为 **s 规格**：`yolov5`=YOLOv5s（7.2M 参数）、`yolov6`=YOLOv6s（17.2M）、
  `yolov8`=YOLOv8s（11.2M），**都不是 n 规格**。
  且 v6/v8 的 `i8`/`fp16` 目录疑似标反（"i8"目录权重更大、实测更慢），
  原因是 i8 版本内嵌的 NMS 后处理头以 fp16 存储；选量化版本时以实测为准。
"""),
    ]
    return notebook(cells)


# ================================================================ 03 FastSAM

def build_03():
    files = [
        ("fastsam-s-rknn3-rk1828", "fastsam_s.rknn", "models/fastsam/fastsam_s.rknn"),
        ("fastsam-s-rknn3-rk1828", "fastsam_s.weight", "models/fastsam/fastsam_s.weight"),
        ("yolo26-segment-rknn3-rk1828", "model/bus.jpg", "datasets/bus.jpg"),
        ("mobilenet_v2-rknn3-rk1828", "dog_224x224.jpg", "datasets/dog_224x224.jpg"),
    ]
    cells = [
        md("""
# RK1828 FastSAM-s 实例分割 Demo

[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)（中科院自动化所）是基于 YOLOv8-seg 的
**可提示实例分割**模型。本 demo 用 "segment everything" 模式：一次前向找出图中所有物体并给出掩码。

模型两个输出：

| 输出 | 形状 | 含义 |
|---|---|---|
| `output0` | `[1,37,8400]` | cx, cy, w, h, 置信度 + 32 个 mask 系数（已解码） |
| `output1` | `[1,32,160,160]` | mask prototypes（掩码基） |

后处理：置信度过滤 → NMS → `sigmoid(系数 @ prototypes)` 得到每个物体的掩码。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型（可选 HF / HF-Mirror 源） |
| 2 | 选择图片 |
| 3 | 前处理（letterbox） |
| 4 | RKNN 推理 |
| 5 | 后处理（NMS + 掩码合成）与可视化 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files, intro="本 demo 下载 FastSAM-s 模型和示例图片。")
    cells += [
        md("""
## 2. 选择图片
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_image = widgets.Dropdown(options=sorted(glob.glob("datasets/*.jpg")),
                           description="图片:", layout=widgets.Layout(width="60%"))
w_conf = widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05,
                             description="置信度:", style={"description_width": "80px"})
display(w_image, w_conf)
print("选好后继续往下运行 ▼")
"""),
        md("""
## 3. 前处理：letterbox
"""),
        code("""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from vision_post import letterbox, to_input_array, setup_cjk_font

setup_cjk_font()
img = Image.open(w_image.value).convert("RGB")
img_lb, SCALE, PAD_X, PAD_Y = letterbox(img, 640)
x = to_input_array(img_lb)
print(f"原图 {img.size} -> 输入 {x.shape}, scale={SCALE:.3f}, pad=({PAD_X},{PAD_Y})")

# 前处理每个阶段都画出来：原图 -> letterbox -> 输入张量
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, im, title in [
    (axes[0], img, f"1. 原图 {img.size[0]}x{img.size[1]}"),
    (axes[1], img_lb, "2. letterbox 640x640\\n(等比缩放 + 灰边填充)"),
    (axes[2], x[0], f"3. 模型输入 {x.shape}\\nuint8 NHWC"),
]:
    ax.imshow(np.asarray(im))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## 4. RKNN 推理
"""),
        code("""
import time
from rknn3 import Rknn3Model

model = Rknn3Model("models/fastsam/fastsam_s.rknn")
print("输出:", model.output_attrs)

_ = model.infer([x])
t0 = time.time()
outputs = model.infer([x])
print(f"推理耗时: {(time.time()-t0)*1000:.1f} ms")
"""),
        md("""
## 5. 后处理：NMS + 掩码合成

`sigmoid(mask系数 @ prototypes)` 得到 160×160 掩码，放大到输入尺寸、按框裁剪，
再映射回原图；过滤掉过小（面积占比 < 0.2%）的碎片掩码。
"""),
        code("""
import numpy as np
from vision_post import decode_fastsam, scale_masks, unletterbox_boxes

boxes, scores, masks = decode_fastsam(outputs, conf_thresh=w_conf.value)
masks_full = scale_masks(masks, boxes, (img.height, img.width), SCALE, PAD_X, PAD_Y)
boxes = unletterbox_boxes(boxes, SCALE, PAD_X, PAD_Y, *img.size)

ratio = (masks_full > 0.5).mean(axis=(1, 2))
keep = ratio > 0.002
boxes, scores, masks_full = boxes[keep], scores[keep], masks_full[keep]
print(f"共 {len(boxes)} 个物体（过滤碎片后）")
"""),
        md("""
## 6. 可视化
"""),
        code("""
from vision_post import draw_boxes, draw_masks

overlay = draw_masks(img, masks_full)
result = draw_boxes(overlay, boxes, np.zeros(len(boxes)), scores)
model.close()
result
"""),
        md("""
## 小结

- FastSAM 是**类别无关**的：只输出"是不是物体"和掩码，不分类别。
- 掩码 = `sigmoid(32 个系数 × 32 通道 prototypes)`，这是 YOLOv8-seg 系列的通用做法。
"""),
    ]
    return notebook(cells)


# ================================================================ C++ demo 公共

ASR_WAVS = [
    ("whisper-base-rknn3-rk1828", "model/test_zh.wav", "datasets/asr/test_zh.wav"),
    ("whisper-base-rknn3-rk1828", "model/test_en.wav", "datasets/asr/test_en.wav"),
    ("whisper-base-rknn3-rk1828", "model/long_en.wav", "datasets/asr/long_en.wav"),
    ("sensevoice-small-rknn3-rk1828", "model/zh.wav", "datasets/asr/sensevoice_zh.wav"),
    ("paraformer-rknn3-rk1828", "model/asr_example.wav", "datasets/asr/asr_example.wav"),
    ("paraformer-rknn3-rk1828", "model/asr_example_hotword.wav",
     "datasets/asr/asr_example_hotword.wav"),
    ("qwen3-forced-aligner-rknn3-rk1828", "audio.wav", "datasets/asr/aligner_demo.wav"),
]

def wav_dropdown(default=None):
    """ASR 类 notebook 的音频选择单元格；default 为优先选中的文件名。"""
    val = ""
    if default:
        val = (f"\nif any(o.endswith('/{default}') for o in w_wav.options):\n"
               f"    w_wav.value = [o for o in w_wav.options "
               f"if o.endswith('/{default}')][0]\n")
    return code(f"""
import glob
import ipywidgets as widgets
from IPython.display import display

w_wav = widgets.Dropdown(options=sorted(glob.glob("datasets/asr/*.wav")),
                         description="音频:", layout=widgets.Layout(width="60%"))
display(w_wav){val}print("选好后继续往下运行 ▼")
""")

AUDIO_PLAY_MD = md("""
先听一下这段音频：
""")

AUDIO_PLAY_CODE = code("""
from nb_demo import show_audio

show_audio(w_wav.value)
""")


def cpp_intro(title, body):
    return md(f"# {title}\n\n{body}")


# ================================================================ 04 姿态/分割

def build_04():
    pose_files = repo_files("yolo26-pose-rknn3-rk1828", "demos/yolo26_pose")
    seg_files = repo_files("yolo26-segment-rknn3-rk1828", "demos/yolo26_segment")
    files = pose_files + seg_files
    cells = [
        cpp_intro("RK1828 YOLO26 姿态估计 & 实例分割 Demo",
                  """
两个基于 **YOLO26** 的下游任务，都是官方 `rknn3-model-zoo` 的预编译 C++ demo：

- **姿态估计**（pose）：检测人体 + 17 个关键点（n/s/m 三个规格，i8/fp16 两种量化）
- **实例分割**（segment）：检测框 + 像素级掩码（n/s/m，i8/fp16）

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型和 demo 程序（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 姿态估计 |
| 4 | 实例分割 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/yolo26_pose", "demos/yolo26_segment"])
    cells += [
        run_helper_code(),
        md("""
## 3. 姿态估计（17 关键点）

参考成绩：yolo26m-pose i8 约 152ms/帧，示例图上人体置信度 0.93。
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_size = widgets.Dropdown(options=["n", "s", "m"], value="m", description="规格:")
w_quant = widgets.Dropdown(options=["i8", "fp16"], description="量化:")
display(w_size, w_quant)
print("选好后继续往下运行 ▼")
"""),
        code("""
DEMO = "demos/yolo26_pose"
m = f"yolo26{w_size.value}-pose_{w_quant.value}"
before = snapshot(DEMO)
run_demo([f"./rknn_yolo26_pose_demo", f"model/{m}.rknn", f"model/{m}.weight",
          "model/little_boy.jpg", "0x1"], DEMO)
show_new_images(DEMO, before)
"""),
        md("""
## 4. 实例分割（检测框 + 像素掩码）

参考成绩：yolo26m-seg i8 约 261ms/帧，COCO 示例图检出 4 个人 + 1 辆巴士。
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_size2 = widgets.Dropdown(options=["n", "s", "m"], value="m", description="规格:")
w_quant2 = widgets.Dropdown(options=["i8", "fp16"], description="量化:")
display(w_size2, w_quant2)
print("选好后继续往下运行 ▼")
"""),
        code("""
DEMO = "demos/yolo26_segment"
m = f"yolo26{w_size2.value}-seg_{w_quant2.value}"
before = snapshot(DEMO)
run_demo([f"./rknn_yolo26_segment_demo", f"model/{m}.rknn", f"model/{m}.weight",
          "model/bus.jpg", "0x1"], DEMO)
show_new_images(DEMO, before)
"""),
        md("""
## 小结

- 两个 demo 的模型都是**全套官方转换产物**，后处理（含 NMS）由 C++ demo 完成。
- 可换 n/s（更小更快）规格对比精度与耗时。
"""),
    ]
    return notebook(cells)


# ================================================================ 05 RAFT 双目

def build_05():
    files = repo_files("raft-stereo-rknn3-rk1828", "demos/raft_stereo")
    cells = [
        cpp_intro("RK1828 RAFT-Stereo 双目视差 Demo",
                  """
[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)（普林斯顿视觉与学习实验室）
从一对左右目图片估计**视差图**（disparity）：越亮表示越近。
这里用的是 realtime 版本，输入 160×240，int8 量化，RK1828 上约 90ms/对。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 运行并查看视差图 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/raft_stereo"])
    cells += [
        run_helper_code(),
        md("""
## 3. 运行 demo

输入：`model/xt_left.jpg` / `model/xt_right.jpg`（一对示例双目图）
输出：`out.png` 视差热力图
"""),
        code("""
show_images(["demos/raft_stereo/model/xt_left.jpg",
             "demos/raft_stereo/model/xt_right.jpg"])
"""),
        code("""
DEMO = "demos/raft_stereo"
before = snapshot(DEMO)
run_demo(["./rknn_raft_stereo_demo",
          "model/raftstereo-realtime-160x240.rknn",
          "model/raftstereo-realtime-160x240.weight",
          "model/xt_left.jpg", "model/xt_right.jpg", "0x1"], DEMO)
show_new_images(DEMO, before)
"""),
        md("""
## 小结

- 视差 = 同一物体在左右图中的水平位移，视差越大距离越近，是双目深度估计的核心。
- 可换成自己的双目照片（保持 160×240 或由 demo 内部缩放）。
"""),
    ]
    return notebook(cells)


# ================================================================ 06 DA3 深度

def build_06():
    files = repo_files("depth-anything-v3-rknn3-rk1828", "demos/depth_anything_v3")
    cells = [
        cpp_intro("RK1828 Depth Anything V3 多视角深度估计 Demo",
                  """
[Depth Anything V3](https://github.com/ByteDance-Seed/depth-anything-3)（字节跳动）
多视角深度估计。模型被拆成三个子模型（local / global / head），固定输入 **10 个视角**，
输出每个视角的深度热力图。RK1828 上 10 视角合计约 12 秒。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型和 10 张示例图（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 准备图片列表并运行 |
| 4 | 查看深度图 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/depth_anything_v3"])
    cells += [
        run_helper_code(),
        md("""
## 3. 准备 10 视角图片列表并运行

仓库自带同一场景的 10 张示例图（`model/frame_01.jpg` ~ `frame_10.jpg`），
demo 需要一个恰好 10 行的图片路径列表文件。
"""),
        code("""
import glob, os

DEMO = "demos/depth_anything_v3"
frames = sorted(glob.glob(os.path.join(DEMO, "model/frame_*.jpg")))
assert len(frames) == 10, f"需要恰好 10 张图，当前 {len(frames)} 张"
with open(os.path.join(DEMO, "image_list.txt"), "w") as f:
    f.write("\\n".join(os.path.abspath(p) for p in frames))  # demo 要求绝对路径
show_images(frames[:5], max_width=320)
print(f"共 {len(frames)} 张，列表已写入 image_list.txt")
"""),
        code("""
run_demo(["./rknn_da3_base_demo",
          "model/da3_base_local.rknn", "model/da3_base_global.rknn",
          "model/da3_base_head.rknn",
          "image_list.txt", "10", "depth_vis"], DEMO)
"""),
        md("""
## 4. 查看深度热力图（越亮越近）
"""),
        code("""
import glob

depth_imgs = sorted(glob.glob(os.path.join(DEMO, "depth_vis/*")))
print(f"生成 {len(depth_imgs)} 张深度图")
show_images(depth_imgs[:5], max_width=320)
"""),
        md("""
## 小结

- 大模型可以拆成多个子模型分别转换、流水线运行，这是 RKNN 部署超大模型的常见手段。
- 换自己的照片时注意：必须恰好 10 个视角，且尽量是同一场景不同角度。
"""),
    ]
    return notebook(cells)


# ================================================================ 07 Chinese-CLIP

def build_07():
    files = repo_files("chinese-clip-rknn3-rk1828", "demos/chinese_clip")
    cells = [
        cpp_intro("RK1828 Chinese-CLIP 图文匹配 Demo",
                  """
[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)（达摩院）中文图文匹配：
给定一张图和几条中文候选描述，计算匹配概率。文本编码器和视觉编码器分别转换、
全部跑在 RK1828 上。示例图上正确描述的得分约 0.96 vs 0.04。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 运行图文匹配 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/chinese_clip"])
    cells += [
        run_helper_code(),
        md("""
## 3. 运行图文匹配

输入示例图 `model/demo.jpg`，demo 内置几条中文候选描述，输出每条的概率。
"""),
        code("""
show_images(["demos/chinese_clip/model/demo.jpg"], max_width=400)
"""),
        code("""
DEMO = "demos/chinese_clip"
run_demo(["./rknn_qaclipvit_demo",
          "model/chinese_clip_text.rknn", "model/chinese_clip_text.weight",
          "model/chinese_clip_text.embed.bin", "model/chinese_clip_text.tokenizer.gguf",
          "model/chinese_clip_vision.rknn", "model/chinese_clip_vision.weight",
          "model/demo.jpg", "1"], DEMO)
"""),
        md("""
## 小结

- CLIP 类模型把图文映射到同一向量空间，"匹配"就是算相似度，可做零样本分类/以文搜图。
- 多模态模型的文本侧（tokenizer + embed）也需要一起部署，见 `model/chinese_clip_text.*`。
"""),
    ]
    return notebook(cells)


# ================================================================ 08 Qwen3-VL

def build_08():
    files_2b = repo_files("qwen3-vl-2b-rknn3-rk1828", "demos/qwen3_vl_2b")
    files_4b = repo_files("qwen3-vl-4b-rknn3-rk1828", "demos/qwen3_vl_4b")
    files = files_2b + files_4b
    cells = [
        cpp_intro("RK1828 Qwen3-VL 视觉语言大模型 Demo",
                  """
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)（通义千问）视觉语言模型：
看图回答 / 图像描述。LLM 部分 **GRQ W4A16 量化**，视觉编码器 fp16，全部跑在 RK1828 上。

| 规格 | 生成速度（实测） |
|---|---|
| Qwen3-VL-2B | 62.7 tokens/s |
| Qwen3-VL-4B | 38.7 tokens/s |

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型（可选 HF / HF-Mirror 源，可只下 2B 或 4B） |
| 2 | 解压预编译 demo |
| 3 | 输入提示词，看图说话 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(
        files,
        intro="两个规格共约 10GB。磁盘紧张的话，下载单元格可以改成只下载其中一个规格。")
    cells += extract_cells(["demos/qwen3_vl_2b", "demos/qwen3_vl_4b"])
    cells += [
        run_helper_code(),
        md("""
## 3. 选择模型规格和提示词
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(options=[("Qwen3-VL-2B（快）", "2b"),
                                    ("Qwen3-VL-4B（更聪明）", "4b")],
                           value="2b", description="模型:")
w_prompt = widgets.Textarea(value="Describe this image",
                            description="提示词:",
                            layout=widgets.Layout(width="70%", height="60px"))
display(w_model, w_prompt)
print("选好后继续往下运行 ▼")
"""),
        code("""
DEMO = f"demos/qwen3_vl_{w_model.value}"
tag = "Qwen3-VL-2B" if w_model.value == "2b" else "Qwen3-VL-4B"
show_images([f"{DEMO}/model/demo.jpg"], max_width=400)
run_demo(["./rknn_qwen3_vl_demo",
          f"{tag}-vision.rknn", f"{tag}-vision.weight",
          f"{tag}-llm.rknn", f"{tag}-llm.weight",
          f"{tag}-llm.tokenizer.gguf", f"{tag}-llm.embed.bin",
          "0xff", "0xff", "model/demo.jpg", w_prompt.value, "384", "384"], DEMO)
"""),
        md("""
## 小结

- VLM = 视觉编码器（把图片转成 token 嵌入）+ LLM（自回归生成文字），两部分都是独立 RKNN 模型。
- `0xff 0xff` 是设备/会话参数占位；`384 384` 是视觉输入分辨率。
- 换提示词试试中文提问、数数图里有几个人、识别文字等。
"""),
    ]
    return notebook(cells)


# ================================================================ 09 Qwen3-ASR

def build_09():
    files = repo_files("qwen3-asr-rknn3-rk1828", "demos/qwen3_asr") + ASR_WAVS
    cells = [
        cpp_intro("RK1828 Qwen3-ASR 语音识别 Demo",
                  """
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)（通义千问）支持 **52 种语言和方言**，
自动判断语种。拆成两部分，全部跑在 NPU 上：

```
音频(wav) -> [Audio Encoder] 提取音频特征 -> [LLM] 自回归生成文字 -> 识别结果
             encoder.rknn (fp16)            llm.rknn (W4A16 量化)
```

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型和示例音频（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 选择音频并试听 |
| 4 | 识别 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files, intro="示例音频来自其他 ASR 仓库，放在 `datasets/asr/` 共用。")
    cells += extract_cells(["demos/qwen3_asr"])
    cells += [
        run_helper_code(),
        md("""
## 3. 选择音频并试听
"""),
        wav_dropdown("test_zh.wav"),
        AUDIO_PLAY_MD,
        AUDIO_PLAY_CODE,
        md("""
## 4. 识别

`0xff 0xff` 是设备/会话参数占位。实测一段 8.9 秒中文音频可以完整正确转写。
"""),
        code("""
import os

DEMO = "demos/qwen3_asr"
wav = os.path.abspath(w_wav.value)
run_demo(["./rknn_qwen3_asr_demo",
          "model/encoder.rknn", "model/encoder.weight",
          "model/llm.rknn", "model/llm.weight",
          "model/llm.tokenizer.gguf", "model/llm.embed.bin",
          "0xff", "0xff", wav], DEMO)
"""),
        md("""
## 小结

- 编码器 + LLM 解耦：音频特征提取和文字生成分别在两个 RKNN 模型里完成。
- 压缩包里还有一个 `rknn_qwen3_asr_demo_online`（流式版），可以试试看。
"""),
    ]
    return notebook(cells)


# ================================================================ 10 Whisper

def build_10():
    files = repo_files("whisper-base-rknn3-rk1828", "demos/whisper_base") + ASR_WAVS
    cells = [
        cpp_intro("RK1828 Whisper-base 语音识别 Demo",
                  """
OpenAI [whisper-base](https://github.com/openai/whisper)，中英文语音识别。
拆成三个模型：**编码器** + **首 token 解码器（decode0）** + **自回归解码器（decode1）**，
实测中文示例 RTF 0.073（比实时快约 14 倍）。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型和示例音频（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 选择语种和音频 |
| 4 | 识别 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/whisper_base"])
    cells += [
        run_helper_code(),
        md("""
## 3. 选择语种和音频
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_lang = widgets.Dropdown(options=[("中文", "zh"), ("English", "en")],
                          value="zh", description="语种:")
w_wav = widgets.Dropdown(options=sorted(glob.glob("datasets/asr/*.wav")),
                         description="音频:", layout=widgets.Layout(width="60%"))
if any(o.endswith('/test_zh.wav') for o in w_wav.options):
    w_wav.value = [o for o in w_wav.options if o.endswith('/test_zh.wav')][0]
display(w_lang, w_wav)
print("选好后继续往下运行 ▼")
"""),
        AUDIO_PLAY_MD,
        AUDIO_PLAY_CODE,
        md("""
## 4. 识别
"""),
        code("""
import os

DEMO = "demos/whisper_base"
wav = os.path.abspath(w_wav.value)
run_demo(["./rknn_whisper_raw_demo",
          "model/whisper_encoder_base.rknn", "model/whisper_encoder_base.weight",
          "model/whisper_decode0_base.rknn", "model/whisper_decode0_base.weight",
          "model/whisper_decode1_base.rknn", "model/whisper_decode1_base.weight",
          w_lang.value, wav, "model/token_embed.bin",
          "0xff", "0xff", "0xff", "--model-type", "base"], DEMO)
"""),
        md("""
## 小结

- Whisper 的自回归解码被拆成"首 token"和"后续 token"两个模型，是 NPU 部署 Transformer
  decoder 的常见优化（首 token 不需要 KV cache）。
- 长音频（如 `long_en.wav`）会被自动切分处理。
"""),
    ]
    return notebook(cells)


# ================================================================ 11 Paraformer

def build_11():
    files = repo_files("paraformer-rknn3-rk1828", "demos/paraformer") + ASR_WAVS
    cells = [
        cpp_intro("RK1828 Paraformer 中文语音识别 Demo（带热词）",
                  """
阿里 [Paraformer](https://github.com/modelscope/FunASR)**非自回归**中文 ASR：
一次前向直接输出全部文字（不像 Whisper 逐 token 生成），实测 RTF 0.046（约 21 倍实时）。
还支持**热词增强**：给定的词汇会被优先识别（asf_scorer + 热词解码器）。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型和示例音频（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 选择音频（可选热词模式） |
| 4 | 识别 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/paraformer"])
    cells += [
        run_helper_code(),
        md("""
## 3. 选择音频和模式

- **普通模式**：`paraformer_encoder` + `paraformer_decoder`
- **热词模式**：改用 `paraformer_decoder_hotword` + `asf_scorer`，按 `model/hotwords.txt`
  里的热词增强识别（示例热词音频：datasets/asr/asr_example_hotword.wav）
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_wav = widgets.Dropdown(options=sorted(glob.glob("datasets/asr/*.wav")),
                         description="音频:", layout=widgets.Layout(width="60%"))
if any(o.endswith('/asr_example.wav') for o in w_wav.options):
    w_wav.value = [o for o in w_wav.options if o.endswith('/asr_example.wav')][0]
w_hot = widgets.Checkbox(value=False, description="热词模式")
display(w_wav, w_hot)
print("选好后继续往下运行 ▼")
"""),
        AUDIO_PLAY_MD,
        AUDIO_PLAY_CODE,
        md("""
## 4. 识别
"""),
        code("""
import os

DEMO = "demos/paraformer"
wav = os.path.abspath(w_wav.value)
if w_hot.value:
    print(open(os.path.join(DEMO, "model/hotwords.txt")).read())
    run_demo(["./rknn_paraformer_demo",
              "--encoder_path", "model/paraformer_encoder.rknn",
              "--decoder_path", "model/paraformer_decoder_hotword.rknn",
              "--tokens_path", "model/tokens.json",
              "--audio_path", wav,
              "--seaco_path", "model/seaco_decoder.rknn",
              "--hotwords_path", "model/hotwords.txt",
              "--asf_path", "model/asf_scorer.rknn",
              "--asf_nfilter", "50"], DEMO)
else:
    run_demo(["./rknn_paraformer_demo",
              "--encoder_path", "model/paraformer_encoder.rknn",
              "--decoder_path", "model/paraformer_decoder.rknn",
              "--tokens_path", "model/tokens.json",
              "--audio_path", wav], DEMO)
"""),
        md("""
## 小结

- 非自回归 = 并行输出全部 token，比自回归 ASR 快一个数量级，适合实时场景。
- 热词机制适合行业术语、人名地名等长尾词的识别纠偏。
"""),
    ]
    return notebook(cells)


# ================================================================ 12 SenseVoice

def build_12():
    files = repo_files("sensevoice-small-rknn3-rk1828", "demos/sensevoice") + ASR_WAVS
    cells = [
        cpp_intro("RK1828 SenseVoice-Small 多语言语音识别 Demo",
                  """
阿里 [SenseVoice-Small](https://github.com/FunAudioLLM/SenseVoice) 多语言 ASR，
输出里除了文字还**内联语种、情感和声音事件标签**（如 `<|zh|><|HAPPY|><|Speech|>`），
实测单条约 292ms。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型和示例音频（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 选择音频 |
| 4 | 识别（看富标签输出） |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/sensevoice"])
    cells += [
        run_helper_code(),
        md("""
## 3. 选择音频
"""),
        wav_dropdown("sensevoice_zh.wav"),
        AUDIO_PLAY_MD,
        AUDIO_PLAY_CODE,
        md("""
## 4. 识别
"""),
        code("""
import os

DEMO = "demos/sensevoice"
wav = os.path.abspath(w_wav.value)
run_demo(["./rknn_sensevoice_demo",
          "model/sensevoice_fix_10s.rknn", "model/sensevoice_fix_10s.weight",
          wav, "0x1"], DEMO)
"""),
        md("""
## 小结

- SenseVoice 把"语种/情感/事件"当成特殊 token 直接生成，一次推理得到转写 + 副语言信息。
- 模型输入固定 10s，短音频会自动补齐。
"""),
    ]
    return notebook(cells)


# ================================================================ 13 强制对齐

def build_13():
    files = repo_files("qwen3-forced-aligner-rknn3-rk1828", "demos/forced_aligner")
    cells = [
        cpp_intro("RK1828 Qwen3-ForcedAligner 强制对齐 Demo",
                  """
[Qwen3-ForcedAligner](https://github.com/QwenLM/Qwen3-ForcedAligner)：
给定**音频 + 已知文本**，输出每个字/词的时间戳（TSV），中英文均可，实测 RTF 0.041。
典型用途：字幕生成、歌词对齐、语音数据标注。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型和示例音频（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 输入文本并运行 |
| 4 | 查看时间戳结果 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/forced_aligner"])
    cells += [
        run_helper_code(),
        md("""
## 3. 输入要对齐的文本

仓库自带示例 `audio.wav`（中文："欢迎大家来到么哒社区进行体验"），
也可以换自己的 wav + 对应文本。
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_wav = widgets.Text(value="audio.wav", description="音频:",
                     layout=widgets.Layout(width="60%"))
w_text = widgets.Textarea(value="欢迎大家来到么哒社区进行体验",
                          description="文本:",
                          layout=widgets.Layout(width="70%", height="50px"))
w_lang = widgets.Dropdown(options=["Chinese", "English"], description="语言:")
display(w_wav, w_text, w_lang)
print("选好后继续往下运行 ▼")
"""),
        code("""
DEMO = "demos/forced_aligner"
before = snapshot(DEMO)
run_demo(["./rknn_qwen3_forced_aligner_demo",
          "--model_path", "./model",
          "--audio_path", w_wav.value,
          "--text", w_text.value,
          "--language", w_lang.value,
          "--mel_filters_path", "./mel_128_filters.txt",
          "--output_path", "./result.tsv"], DEMO)
"""),
        md("""
## 4. 查看时间戳（TSV：开始秒 / 结束秒 / 字）
"""),
        code("""
import csv

rows = list(csv.reader(open("demos/forced_aligner/result.tsv"), delimiter="\\t"))
if rows and rows[0][0].lower().startswith("start"):
    rows = rows[1:]  # 去掉表头 start_ms / end_ms / text
rows = [(float(r[0]) / 1000, float(r[1]) / 1000, r[2]) for r in rows]  # ms -> s
print(f"{'开始(s)':>8} {'结束(s)':>8}  字")
for s, e, t in rows:
    print(f"{s:8.2f} {e:8.2f}  {t}")
"""),
        code("""
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from vision_post import setup_cjk_font

setup_cjk_font()

wav_path = "demos/forced_aligner/audio.wav"
data, sr = sf.read(wav_path)
t = np.arange(len(data)) / sr
plt.figure(figsize=(12, 3))
plt.plot(t, data, lw=0.3, color="#888")
for s, e, txt in rows:
    plt.axvspan(s, e, alpha=0.25)
    plt.text((s + e) / 2, 0.8, txt, ha="center", fontsize=9)
plt.title("音频波形上的对齐结果")
plt.xlabel("秒")
plt.tight_layout()
plt.show()
"""),
        md("""
## 小结

- 对齐不是"识别"：文本已知，模型只负责把每个字**钉到时间轴上**。
- 波形图上可以直观看到每段 span 与发音的对应关系。
"""),
    ]
    return notebook(cells)


# ================================================================ 14 VITS TTS

def build_14():
    files = repo_files("vits-tts-rknn3-rk1828", "demos/vits_tts")
    cells = [
        cpp_intro("RK1828 VITS 语音合成 Demo",
                  """
[VITS](https://github.com/jaywalnut310/vits) 端到端英文 TTS。两个预训练模型：

| 模型 | 说话人 |
|---|---|
| LJSpeech | 单一女声 |
| VCTK | 109 个说话人（用 `--speaker_id` 选择） |

每个模型拆成 step1（文本编码 + 时长）和 step2（flow + 声码器）两个 RKNN 模型，
滑窗推理，LJSpeech 实测 RTF 0.345。音素数据依赖 `model/espeak-ng-data/`。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 输入英文文本并合成 |
| 4 | 试听 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/vits_tts"])
    cells += [
        run_helper_code(),
        md("""
## 3. 输入文本并合成
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(options=[("LJSpeech 单女声", "ljs"),
                                    ("VCTK 多说话人", "vctk")],
                           description="模型:")
w_speaker = widgets.IntSlider(value=4, min=0, max=108,
                              description="说话人ID:",
                              style={"description_width": "80px"})
w_text = widgets.Textarea(value="VITS is awesome! Have a nice day.",
                          description="文本:",
                          layout=widgets.Layout(width="70%", height="50px"))
display(w_model, w_speaker, w_text)
print("选好后继续往下运行 ▼（说话人ID 仅 VCTK 有效）")
"""),
        code("""
DEMO = "demos/vits_tts"
before = snapshot(DEMO)
args = ["./rknn_vits_demo",
        "--step1_model", f"model/vits_{w_model.value}_step1_slid_fp.rknn",
        "--step1_weight", f"model/vits_{w_model.value}_step1_slid_fp.weight",
        "--step2_model", f"model/vits_{w_model.value}_step2_slid_fp.rknn",
        "--step2_weight", f"model/vits_{w_model.value}_step2_slid_fp.weight",
        "--text", w_text.value,
        "--output", "out.wav"]
if w_model.value == "vctk":
    args += ["--speaker_id", str(w_speaker.value)]
run_demo(args, DEMO)
"""),
        md("""
## 4. 试听合成结果
"""),
        code("""
show_audio("demos/vits_tts/out.wav")
"""),
        md("""
## 小结

- TTS 流水线：文本 -> 音素（espeak-ng）-> step1 出时长/声学特征 -> step2 出声波。
- 换几个 speaker_id 对比不同说话人的音色。
"""),
    ]
    return notebook(cells)


# ================================================================ 15 Embedding

def build_15():
    files = repo_files("qwen3-embedding-0.6b-rknn3-rk1828", "demos/qwen3_embedding")
    cells = [
        cpp_intro("RK1828 Qwen3-Embedding 文本向量 Demo",
                  """
[Qwen3-Embedding-0.6B](https://github.com/QwenLM/Qwen3-Embedding)：
把文本编码成 **1024 维向量**，用于语义搜索 / RAG 检索。GRQ W4A16 量化，
实测单条 prefill 76.7ms； paraphrase 余弦相似度 0.958 vs 无关文本 0.283。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 输入若干句子并批量编码 |
| 4 | 余弦相似度矩阵 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/qwen3_embedding"])
    cells += [
        run_helper_code(),
        md("""
## 3. 输入句子并编码

每行一句。demo 每次编码一句，把 1024 维 float32 向量写到 `embedding_out.bin`。
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_texts = widgets.Textarea(
    value="How do I reset my password?\\n"
          "I forgot my password, how can I change it?\\n"
          "What's the best way to cook rice?\\n"
          "如何重置我的密码？",
    description="句子(每行一句):", layout=widgets.Layout(width="75%", height="120px"))
display(w_texts)
print("选好后继续往下运行 ▼")
"""),
        code("""
import numpy as np

DEMO = "demos/qwen3_embedding"
texts = [t.strip() for t in w_texts.value.splitlines() if t.strip()]
embs = []
for t in texts:
    run_demo(["./rknn_qwen3_embedding_demo",
              "model/Qwen3-Embedding-0.6B.rknn", "model/Qwen3-Embedding-0.6B.weight",
              "model/Qwen3-Embedding-0.6B.tokenizer.gguf",
              "model/Qwen3-Embedding-0.6B.embed.bin",
              "0xff", t], DEMO)
    embs.append(np.fromfile(f"{DEMO}/embedding_out.bin", dtype=np.float32))
embs = np.stack(embs)
print("向量矩阵:", embs.shape)
"""),
        md("""
## 4. 余弦相似度矩阵

语义相近的句子（含跨语言）相似度应该接近 1，无关的接近 0。
"""),
        code("""
import matplotlib.pyplot as plt

norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
sim = norm @ norm.T

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(sim, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(len(texts)), [t[:18] for t in texts], rotation=45, ha="right")
ax.set_yticks(range(len(texts)), [t[:18] for t in texts])
for i in range(len(texts)):
    for j in range(len(texts)):
        ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                color="white" if sim[i, j] > 0.6 else "black")
plt.colorbar(im)
plt.title("余弦相似度")
plt.tight_layout()
plt.show()
"""),
        md("""
## 小结

- Embedding 模型只看"语义距离"，与语种无关 —— 可以试试中英混排的同义句。
- 这是 RAG 系统的检索侧组件；配合 16 号 notebook 的 Reranker 就是完整的检索 + 精排。
"""),
    ]
    return notebook(cells)


# ================================================================ 16 Reranker

def build_16():
    files = repo_files("qwen3-reranker-0.6b-rknn3-rk1828", "demos/qwen3_reranker")
    cells = [
        cpp_intro("RK1828 Qwen3-Reranker 文档重排序 Demo",
                  """
[Qwen3-Reranker-0.6B](https://github.com/QwenLM/Qwen3-Embedding)：
给（query, document）对打 0~1 的相关性分数，用于 RAG 精排。GRQ W4A16 量化，
实测单对 prefill 132ms；相关 0.9985 vs 无关 0.0000。

| 步骤 | 内容 |
|---|---|
| 1 | 下载模型（可选 HF / HF-Mirror 源） |
| 2 | 解压预编译 demo |
| 3 | 输入 query 和候选文档 |
| 4 | 打分排序 |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/qwen3_reranker"])
    cells += [
        run_helper_code(),
        md("""
## 3. 输入 query 和候选文档（每行一篇）
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_query = widgets.Text(value="What are gravitational waves?",
                       description="Query:", layout=widgets.Layout(width="75%"))
w_docs = widgets.Textarea(
    value="Gravitational waves are ripples in spacetime caused by accelerating masses.\\n"
          "The best chocolate chip cookie recipe uses brown butter.\\n"
          "LIGO first detected gravitational waves in 2015, confirming Einstein's prediction.\\n"
          "Python is a popular programming language for data science.",
    description="文档(每行一篇):", layout=widgets.Layout(width="75%", height="120px"))
display(w_query, w_docs)
print("选好后继续往下运行 ▼")
"""),
        md("""
## 4. 打分排序
"""),
        code("""
import re

import numpy as np

DEMO = "demos/qwen3_reranker"
docs = [d.strip() for d in w_docs.value.splitlines() if d.strip()]
scores = []
for d in docs:
    p = run_demo(["./rknn_qwen3_reranker_demo",
                  "model/Qwen3-Reranker-0.6B.rknn", "model/Qwen3-Reranker-0.6B.weight",
                  "model/Qwen3-Reranker-0.6B.tokenizer.gguf",
                  "model/Qwen3-Reranker-0.6B.embed.bin",
                  "0xff", w_query.value, d], DEMO)
    nums = re.findall(r"(?:score|Score|relevance)[:\\s]+([0-9]*\\.?[0-9]+)", p.stdout)
    if not nums:  # 兜底：取输出中最后一个 0~1 的小数
        nums = re.findall(r"(?<![\\d.])(0?\\.[0-9]+|1\\.0+)(?![\\d.])", p.stdout)
    scores.append(float(nums[-1]))
    print(f"  -> {scores[-1]:.4f}")

order = np.argsort(scores)[::-1]
print("\\n按相关性排序:")
for i in order:
    print(f"  {scores[i]:.4f}  {docs[i][:60]}")
"""),
        code("""
import matplotlib.pyplot as plt

labels = [d[:28] for d in docs]
order = np.argsort(scores)
plt.figure(figsize=(8, 3.5))
plt.barh([labels[i] for i in order], [scores[i] for i in order], color="#3b82f6")
plt.xlim(0, 1)
plt.title(f"query: {w_query.value}")
plt.tight_layout()
plt.show()
"""),
        md("""
## 小结

- Reranker 是 RAG 的"精排"：先用 Embedding 粗排出 top-K，再用它逐个精排提准。
- 相关与不相关的分数差非常极端（0.99 vs 0.00），说明模型区分度很好。
"""),
    ]
    return notebook(cells)


# ================================================================ main

BUILDERS = {
    "01_classification.ipynb": build_01,
    "02_yolo_detection.ipynb": build_02,
    "03_FastSAM.ipynb": build_03,
    "04_yolo26_pose_segment.ipynb": build_04,
    "05_RAFT_Stereo.ipynb": build_05,
    "06_DepthAnythingV3.ipynb": build_06,
    "07_QAClip.ipynb": build_07,
    "08_Qwen3_VL.ipynb": build_08,
    "09_Qwen3_ASR.ipynb": build_09,
    "10_whisper.ipynb": build_10,
    "11_paraformer.ipynb": build_11,
    "12_sensevoice.ipynb": build_12,
    "13_forced_aligner.ipynb": build_13,
    "14_vits_tts.ipynb": build_14,
    "15_Qwen3_Embedding.ipynb": build_15,
    "16_Qwen3_Reranker.ipynb": build_16,
}


def main():
    for name, builder in BUILDERS.items():
        nb = builder()
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("ok", name, f"({len(nb['cells'])} cells)")
    print("total", len(BUILDERS))


if __name__ == "__main__":
    main()
