# -*- coding: utf-8 -*-
"""Regenerate all 17 notebooks under rk1828-notebook (English edition).

Usage: python3 _build_notebooks.py
Depends on the qhddz HF repos under /home/ubuntu/hugging_face/ (used to
enumerate each repo's file list for the "download models" first step).

Every notebook shares the same structure:
  0. Environment check
  1. Download models (from huggingface.co; existing files are skipped)
  2. ... demo-specific steps
"""
import json
import os

HF_DIR = os.environ.get("HF_DIR", "/home/ubuntu/hugging_face")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

EXCLUDE = {".gitattributes", "README.md", "output.npy", "out.wav",
           "convert_left.jpg", "convert_right.jpg"}


# ---------------------------------------------------------------- helpers

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
    """Enumerate HF repo files -> (repo, remote, local) list."""
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


# ---------------------------------------------------------------- shared cells

ENV_MD = md("""
## 0. Environment check

Make sure the RKNN3 runtime (`librknn3_api.so`) and the RK1828 device are ready.
If this fails, install the driver/runtime first (the `rknn3-rk182x` DEB package)
and check with `sudo rknn-smi info` that the RK1828 shows up.
""")

ENV_CODE = code("""
from rknn3 import get_lib, find_devices

_ = get_lib()  # loads /usr/lib/librknn3_api.so; failure means the runtime is not installed

devices = find_devices()
if not devices:
    raise RuntimeError("No RK1828 device found! Check: sudo rknn-smi info")
for dev_id, dev_type in devices:
    print(f"Found device: id={dev_id}  interface={dev_type}")
""")


def download_cells(files, intro=None):
    """Step 1: download models (from huggingface.co; existing files skipped)."""
    md_src = """
## 1. Download models

The models for this demo are hosted on Hugging Face (account `qhddz`) and are
downloaded directly from **huggingface.co**. Run the next cell to start.

> Files that already exist (and are non-empty) are **skipped automatically** —
> re-running never downloads them twice.
> If your network cannot reach huggingface.co, set up a proxy first, e.g.
> `export https_proxy=http://<your-proxy>:<port>` before starting Jupyter.
"""
    if intro:
        md_src += "\n\n" + intro
    return [
        md(md_src),
        code(f"""
# (HF repo, path in repo, local path)
FILES = {files_literal(files)}
"""),
        code("""
from nb_download import ensure_files

ensure_files(FILES, "hf")
"""),
    ]


EXTRACT_MD = md("""
## 2. Extract the prebuilt demo

`demo_x86_64.tar.gz` contains the official x86_64 C++ demo and its runtime
libraries (`lib/`). Hugging Face does not host raw executables, so they are
shipped as a tarball; extract once (re-running skips automatically).

> **Why a C++ demo instead of pure Python?**
> Notebooks 01-05 (classification / detection / FastSAM / pose / segmentation)
> are a single model with one forward pass and simple post-processing, so they
> walk through everything in Python (`rknn3.py`). This demo's pipeline is far
> more complex — chained models, tokenizer, KV cache, sliding-window /
> streaming state, hundreds of lines of orchestration — reimplementing it in
> Python would be costly and risk diverging from the official behavior.
> So we run the official `rknn3-model-zoo` C++ demo directly; source code:
> [github.com/airockchip/rknn3-model-zoo](https://github.com/airockchip/rknn3-model-zoo)
> under `examples/`.
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


# ================================================================ 01 classification

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
# RK1828 Image Classification Demo (MobileNetV2 / ResNet50-v2)

This notebook runs ImageNet classification on the **RK1828 AI coprocessor**
(PCIe-attached, 20 TOPS NPU) and walks through the full inference pipeline:

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Pick a model and an image |
| 3 | Pre-processing (turn the image into a tensor) |
| 4 | RKNN inference (on the NPU) |
| 5 | Post-processing (dequantize + Softmax, Top-5) |
| 6 | Multi-model comparison (2 models x 2 quantizations) |

> Background: **i8** quantizes the weights to 8-bit integers (smaller, faster);
> **fp16** is the 16-bit float version. Run both and compare.
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(
        files,
        intro="This demo downloads MobileNetV2 and ResNet50-v2 (i8 and fp16 each), "
              "plus the ImageNet labels and a sample image.")
    cells += [
        md("""
## 2. Pick a model and an image

- **Model**: `mobilenet_v2` (lightweight, 3.4M params) or `resnet50_v2` (classic, 25M params)
- **Quantization**: `i8` (int8) or `fp16` (half precision)
- **Image**: samples under `datasets/` (you can drop your own ~224x224 image there)
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(options=["mobilenet_v2", "resnet50_v2"],
                           description="Model:")
w_quant = widgets.Dropdown(options=["i8", "fp16"], description="Quant:")
w_image = widgets.Dropdown(options=sorted(glob.glob("datasets/*.jpg")),
                           description="Image:", layout=widgets.Layout(width="60%"))
display(w_model, w_quant, w_image)

LABELS = open("datasets/imagenet_labels.txt").read().splitlines()
print(f"{len(LABELS)} classes. Then keep running downwards ▼")
"""),
        md("""
## 3. Pre-processing

The model takes **224x224 NHWC uint8 RGB** (normalization is baked into the
model — no mean/std needed). We resize the image to 224x224 and add a batch dim.
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
print("input tensor:", x.shape, x.dtype)

# show every pre-processing stage: original -> resize -> input tensor
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
for ax, im, title in [
    (axes[0], img_orig, f"1. original {img_orig.size[0]}x{img_orig.size[1]}"),
    (axes[1], img, "2. resize 224x224"),
    (axes[2], x[0], f"3. model input {x.shape}\\nuint8 NHWC"),
]:
    ax.imshow(np.asarray(im))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## 4. RKNN inference

Load the `.rknn` (compute graph) + `.weight` (weights) onto the RK1828 and run
one timed inference.
"""),
        code("""
import time
from rknn3 import Rknn3Model

model_path = f"models/cnn/{w_model.value}/{w_quant.value}/{w_model.value}.rknn"
model = Rknn3Model(model_path)
print("input attrs:", model.input_attrs)
print("output attrs:", model.output_attrs)

_ = model.infer([x])  # warmup
t0 = time.time()
outputs = model.infer([x])
print(f"inference time: {(time.time()-t0)*1000:.1f} ms")
"""),
        md("""
## 5. Post-processing: dequantize + Softmax

The i8 model outputs int8; restore float logits with the output tensor's own
`scale` / `zero_point`, then Softmax and take the Top-5.
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
axes[1].set_title("Top-5 probability")
axes[1].set_xlim(0, 1)
plt.tight_layout()
plt.show()

model.close()
"""),
        md("""
## 6. Multi-model comparison (i8 + fp16, full sweep)

Same image, running **2 models x 2 quantizations (i8 / fp16) = 4 combos**.
Progress is printed live; afterwards you get a comparison table, a latency
bar chart, and a Top-5 mosaic. Each model is timed 3 times and the best is
kept (warm-up jitter excluded).
"""),
        code("""
import time
from rknn3 import Rknn3Model
from vision_post import dequant

ALL_MODELS = ["mobilenet_v2", "resnet50_v2"]
QUANTS = ["i8", "fp16"]

compare = []  # (model, quant, latency_ms, top5_idx, top5_prob)
total = len(ALL_MODELS) * len(QUANTS)
step = 0
for name in ALL_MODELS:
    for quant in QUANTS:
        step += 1
        print(f"[{step}/{total}] loading {name} ({quant}) ...", flush=True)
        m = Rknn3Model(f"models/cnn/{name}/{quant}/{name}.rknn")
        m.infer([x])  # warmup
        best, outs = float("inf"), None
        for _ in range(3):
            t0 = time.time()
            outs = m.infer([x])
            best = min(best, (time.time() - t0) * 1000)
        logits = dequant(outs[0], m.output_attrs[0]).ravel()
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top5_i = p.argsort()[::-1][:5]
        compare.append((name, quant, best, top5_i, p[top5_i]))
        m.close()
        print(f"      -> {best:6.1f} ms, Top-1: {LABELS[top5_i[0]]} "
              f"({p[top5_i[0]]*100:.1f}%)", flush=True)
print("\\nAll done ✅", flush=True)
"""),
        md("""
### Comparison table
"""),
        code("""
from IPython.display import HTML

rows = "".join(
    f"<tr><td>{n}</td><td>{q}</td><td>{lat:.1f}</td>"
    f"<td>{', '.join(f'{LABELS[i]}({pp*100:.1f}%)' for i, pp in list(zip(t, pr))[:5])}</td></tr>"
    for n, q, lat, t, pr in compare)
HTML("<table style='text-align:left'>"
     "<tr><th>Model</th><th>Quant</th><th>Latency (ms)</th><th>Top-5 (prob)</th></tr>"
     + rows + "</table>")
"""),
        md("""
### Latency bar chart (i8 vs fp16)
"""),
        code("""
import matplotlib.pyplot as plt
from vision_post import setup_cjk_font

setup_cjk_font()
colors = {"i8": "#3b82f6", "fp16": "#f59e0b"}
xs = np.arange(len(ALL_MODELS))
width = 0.38
fig, ax = plt.subplots(figsize=(8, 4.5))
for k, q in enumerate(QUANTS):
    lats = [lat for n, qq, lat, *_ in compare if qq == q]
    tops = [LABELS[t[0]] for n, qq, _, t, _ in compare if qq == q]
    bars = ax.bar(xs + (k - 0.5) * width, lats, width, label=q, color=colors[q])
    for bar, lat, top in zip(bars, lats, tops):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{lat:.0f}ms\\n{top}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs, ALL_MODELS)
ax.set_ylabel("inference latency (ms)")
ax.set_title("Classification models (same image, i8 vs fp16)")
ax.set_ylim(0, max(lat for _, _, lat, *_ in compare) * 1.35)
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""
### Top-5 mosaic
"""),
        code("""
import matplotlib.pyplot as plt
from vision_post import setup_cjk_font

setup_cjk_font()
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, (n, q, lat, t, pr) in zip(axes.flat, compare):
    names = [LABELS[i] for i in t][::-1]
    ax.barh(names, list(pr)[::-1], color="#3b82f6" if q == "i8" else "#f59e0b")
    ax.set_xlim(0, 1)
    ax.set_title(f"{n} ({q})  {lat:.0f}ms", fontsize=11)
plt.tight_layout()
plt.show()
"""),
        md("""
## Summary

- Full pipeline: **download -> pre-process -> NPU inference -> dequantize/Softmax -> Top-5**
- What step 6 shows:
  - **mobilenet_v2** (lightweight, 3.4M params) is clearly faster than
    **resnet50_v2** (classic, 25M params); ResNet50-v2 is more accurate
    (ImageNet Top-1 ~72% vs ~75.6%, paper figures).
  - For the same model, i8 is usually faster than fp16 with nearly identical
    Top-5 — quantization costs very little accuracy.
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
# RK1828 YOLO Object Detection Demo (YOLOv5/v6/v8 + YOLO26)

Run 6 YOLO models in one go and understand the two output formats:

| Model | Output | Post-processing |
|---|---|---|
| YOLOv5 / v6 / v8 | `[1,K,6]` (score, class, x1, y1, x2, y2) | decode + NMS baked into the model; only coordinate restore needed |
| YOLO26 (n/s/m) | 6 raw heads (box + score at strides 8/16/32) | host-side sigmoid + box decode + per-class NMS |

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Pick a model and an image |
| 3 | Pre-processing (letterbox: aspect-preserving resize + gray pad) |
| 4 | RKNN inference |
| 5 | Post-processing and visualization |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(
        files,
        intro="This demo downloads YOLOv5/v6/v8 and YOLO26 n/s/m (i8 and fp16 each), "
              "plus the COCO 80-class labels and sample images.")
    cells += [
        md("""
## 2. Pick a model and an image

Model files live under `models/yolo/<name>/<quant>/`; feel free to use your own image.

**Actual variant of each model** (judged from weight sizes):

| Directory | Actual variant | Params | Note |
|---|---|---|---|
| `yolov5` | **YOLOv5s** | 7.2M | i8/fp16 directories match their content |
| `yolov6` | **YOLOv6s** | 17.2M | ⚠️ `i8`/`fp16` directories **appear swapped** (the "i8" dir has the larger, slower weights — measured) |
| `yolov8` | **YOLOv8s** | 11.2M | ⚠️ same as above, `i8`/`fp16` **appear swapped** |
| `yolo26n/s/m` | YOLO26n / 26s / 26m | 2.4M / 9.5M / 20.4M | directories match their content |

> The v6/v8 directory issue comes from the official Rockchip SDK conversion
> package itself (the i8 build embeds an fp16 NMS post-processing head, making
> it bigger and slower); trust the measurements, not the directory names.
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(
    options=["yolov5", "yolov6", "yolov8", "yolo26n", "yolo26s", "yolo26m"],
    value="yolov8", description="Model:")
w_quant = widgets.Dropdown(options=["i8", "fp16"], description="Quant:")
w_image = widgets.Dropdown(options=sorted(glob.glob("datasets/*.jpg")),
                           description="Image:", layout=widgets.Layout(width="60%"))
display(w_model, w_quant, w_image)

LABELS = open("datasets/coco_80_labels_list.txt").read().splitlines()
print(f"{len(LABELS)} COCO classes. Then keep running downwards ▼")
"""),
        md("""
## 3. Pre-processing: letterbox

YOLO expects a 640x640 square input. Stretching would distort aspect ratio, so
we **resize proportionally and pad with gray**, remembering the scale and
padding — the post-processing needs them to map boxes back to original
coordinates.
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
print(f"original {img.size} -> input {x.shape}, scale={SCALE:.3f}, pad=({PAD_X},{PAD_Y})")

# show every pre-processing stage: original -> letterbox -> input tensor
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, im, title in [
    (axes[0], img, f"1. original {img.size[0]}x{img.size[1]}"),
    (axes[1], img_lb, "2. letterbox 640x640\\n(aspect resize + gray pad)"),
    (axes[2], x[0], f"3. model input {x.shape}\\nuint8 NHWC"),
]:
    ax.imshow(np.asarray(im))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## 4. RKNN inference
"""),
        code("""
import time
from rknn3 import Rknn3Model

model_path = f"models/yolo/{w_model.value}/{w_quant.value}/{w_model.value}.rknn"
model = Rknn3Model(model_path)
print(f"{len(model.output_attrs)} outputs:")
for a in model.output_attrs:
    print("  ", a)

_ = model.infer([x])
t0 = time.time()
outputs = model.infer([x])
print(f"inference time: {(time.time()-t0)*1000:.1f} ms")
"""),
        md("""
## 5. Post-processing

- **YOLOv5/v6/v8**: output is already post-NMS `(score, class, x1, y1, x2, y2)`;
  just filter by confidence
- **YOLO26**: 6 heads (3 box heads `[1,4,h,w]` + 3 score heads `[1,80,h,w]`);
  sigmoid, box decode and per-class NMS run on the host; i8 outputs are
  dequantized first
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
print(f"{len(boxes)} objects detected:")
for b, c, s in zip(boxes, classes, scores):
    print(f"  {LABELS[c]:12s} {s:.2f}  box=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")
"""),
        md("""
## 6. Visualization
"""),
        code("""
from vision_post import draw_boxes

result = draw_boxes(img, boxes, classes, scores, LABELS)
model.close()
result
"""),
        md("""
## 7. Multi-model comparison (i8 + fp16, full sweep)

Same image, running **6 models x 2 quantizations (i8 / fp16) = 12 combos**.
Progress is printed live; afterwards you get a comparison table, a latency
bar chart, and a detection mosaic. Each model is timed 3 times and the best
is kept (warm-up jitter excluded).
"""),
        code("""
import time
from rknn3 import Rknn3Model
from vision_post import (decode_builtin_nms, decode_yolo26_heads,
                         unletterbox_boxes)

# directory -> actual variant: yolov5=YOLOv5s, yolov6=YOLOv6s, yolov8=YOLOv8s
# note: yolov6/yolov8 i8 and fp16 directories appear swapped (official SDK
# package, see the table above)
ALL_MODELS = ["yolov5", "yolov6", "yolov8", "yolo26n", "yolo26s", "yolo26m"]
QUANTS = ["i8", "fp16"]

compare = []  # (model, quant, latency_ms, boxes, classes, scores)
total = len(ALL_MODELS) * len(QUANTS)
step = 0
for name in ALL_MODELS:
    for quant in QUANTS:
        step += 1
        print(f"[{step:2d}/{total}] loading {name} ({quant}) ...", flush=True)
        m = Rknn3Model(f"models/yolo/{name}/{quant}/{name}.rknn")
        m.infer([x])  # warmup
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
        print(f"      -> {best:6.1f} ms, {len(b)} objects", flush=True)
print("\\nAll done ✅", flush=True)
"""),
        md("""
### Comparison table
"""),
        code("""
from IPython.display import HTML

rows = "".join(
    f"<tr><td>{n}</td><td>{q}</td>"
    f"<td>{lat:.1f}</td><td>{len(b)}</td>"
    f"<td>{', '.join(f'{LABELS[ci]}({si:.2f})' for ci, si in list(zip(c, s))[:5]) or '-'}</td></tr>"
    for n, q, lat, b, c, s in compare)
HTML("<table style='text-align:left'>"
     "<tr><th>Model</th><th>Quant</th><th>Latency (ms)</th><th>Objects</th><th>Top-5 detections</th></tr>"
     + rows + "</table>")
"""),
        md("""
### Latency bar chart (i8 vs fp16)
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
                f"{lat:.0f}ms\\n{cnt} obj", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xs, ALL_MODELS)
ax.set_ylabel("inference latency (ms)")
ax.set_title("YOLO multi-model comparison (same image, i8 vs fp16)")
ax.set_ylim(0, max(lat for _, _, lat, *_ in compare) * 1.3)
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""
### Detection mosaic (top row i8, bottom row fp16)
"""),
        code("""
import matplotlib.pyplot as plt
from vision_post import draw_boxes

fig, axes = plt.subplots(2, 6, figsize=(20, 7))
for k, q in enumerate(QUANTS):
    row = [(n, lat, b, c, s) for n, qq, lat, b, c, s in compare if qq == q]
    for ax, (n, lat, b, c, s) in zip(axes[k], row):
        ax.imshow(draw_boxes(img, b, c, s, LABELS))
        ax.set_title(f"{n} ({q})  {lat:.0f}ms  {len(b)} obj", fontsize=10)
        ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## Summary

- YOLOv5/v6/v8 bake decode + NMS into the model, so the host does almost no
  post-processing; YOLO26 emits raw heads for deployment flexibility, with
  post-processing on the host (see `vision_post.decode_yolo26_heads`).
- General rule: i8 beats fp16; bigger models are slower but detect more;
  official reference mAP@0.5:0.95: 26n 40.9 / 26s 48.6 / 26m 53.1 (PyTorch FP32).
- Variants: v5/v6/v8 are the official Rockchip SDK pre-converted models,
  released by family name. Judging by weight size they are all **s variants**:
  `yolov5`=YOLOv5s (7.2M params), `yolov6`=YOLOv6s (17.2M),
  `yolov8`=YOLOv8s (11.2M) — **none of them is an n variant**.
  Also the v6/v8 `i8`/`fp16` directories appear swapped (the "i8" dir has the
  larger, slower weights), because the i8 build embeds an fp16 NMS head;
  trust measurements when picking a quantization.
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
# RK1828 FastSAM-s Instance Segmentation Demo

[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) (CASIA) is a YOLOv8-seg-based
**promptable instance segmentation** model. This demo uses the "segment
everything" mode: one forward pass finds every object and its mask.

The model has two outputs:

| Output | Shape | Meaning |
|---|---|---|
| `output0` | `[1,37,8400]` | cx, cy, w, h, confidence + 32 mask coefficients (decoded) |
| `output1` | `[1,32,160,160]` | mask prototypes |

Post-processing: confidence filter -> NMS -> `sigmoid(coeffs @ prototypes)`.

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Pick an image |
| 3 | Pre-processing (letterbox) |
| 4 | RKNN inference |
| 5 | Post-processing (NMS + mask assembly) and visualization |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files, intro="This demo downloads the FastSAM-s model and sample images.")
    cells += [
        md("""
## 2. Pick an image
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_image = widgets.Dropdown(options=sorted(glob.glob("datasets/*.jpg")),
                           description="Image:", layout=widgets.Layout(width="60%"))
w_conf = widgets.FloatSlider(value=0.5, min=0.1, max=0.9, step=0.05,
                             description="Conf:", style={"description_width": "80px"})
display(w_image, w_conf)
print("Then keep running downwards ▼")
"""),
        md("""
## 3. Pre-processing: letterbox
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
print(f"original {img.size} -> input {x.shape}, scale={SCALE:.3f}, pad=({PAD_X},{PAD_Y})")

# show every pre-processing stage: original -> letterbox -> input tensor
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, im, title in [
    (axes[0], img, f"1. original {img.size[0]}x{img.size[1]}"),
    (axes[1], img_lb, "2. letterbox 640x640\\n(aspect resize + gray pad)"),
    (axes[2], x[0], f"3. model input {x.shape}\\nuint8 NHWC"),
]:
    ax.imshow(np.asarray(im))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## 4. RKNN inference
"""),
        code("""
import time
from rknn3 import Rknn3Model

model = Rknn3Model("models/fastsam/fastsam_s.rknn")
print("outputs:", model.output_attrs)

_ = model.infer([x])
t0 = time.time()
outputs = model.infer([x])
print(f"inference time: {(time.time()-t0)*1000:.1f} ms")
"""),
        md("""
## 5. Post-processing: NMS + mask assembly

`sigmoid(coeffs @ prototypes)` yields 160x160 masks; upscale to input size,
crop by box, then map back to the original image. Fragment masks smaller
than 0.2% of the image are dropped.
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
print(f"{len(boxes)} objects (after fragment filtering)")
"""),
        md("""
## 6. Visualization
"""),
        code("""
from vision_post import draw_boxes, draw_masks

overlay = draw_masks(img, masks_full)
result = draw_boxes(overlay, boxes, np.zeros(len(boxes)), scores)
model.close()
result
"""),
        md("""
## Summary

- FastSAM is **class-agnostic**: it only tells "is this an object" plus the
  mask, with no class label.
- Mask = `sigmoid(32 coefficients x 32-channel prototypes)` — the standard
  approach of the YOLOv8-seg family.
"""),
    ]
    return notebook(cells)


# ================================================================ 04 YOLO26 pose

def build_04():
    repo = "yolo26-pose-rknn3-rk1828"
    files = []
    for m, quants in (("n", ("i8", "fp16")), ("s", ("i8", "fp16")), ("m", ("i8",))):
        for q in quants:
            for ext in ("rknn", "weight"):
                files.append((repo, f"model/yolo26{m}-pose_{q}.{ext}",
                              f"models/yolo26_pose/yolo26{m}-pose_{q}.{ext}"))
    files.append((repo, "model/little_boy.jpg", "datasets/little_boy.jpg"))

    cells = [
        md("""
# RK1828 YOLO26-pose Human Pose Estimation Demo

Ultralytics **YOLO26-pose**: detects people and regresses **17 keypoints**
(nose/eyes/ears/shoulders/elbows/wrists/hips/knees/ankles). End-to-end
NMS-free design (we still run a light NMS on the host for dedup), 9 output
heads:

| Head | Shape (one set per stride 8/16/32) | Meaning |
|---|---|---|
| `box_*` | `[1,4,80,80]` `[1,4,40,40]` `[1,4,20,20]` | box regression (l,t,r,b), no DFL |
| `score_*` | `[1,1,...]` | person confidence (single class) |
| `pos_*` | `[1,51,...]` | 17 keypoints x 3 (x, y, confidence logit) |

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Pick a model and an image |
| 3 | Pre-processing (letterbox: aspect resize + gray pad) |
| 4 | RKNN inference |
| 5 | Post-processing (box + keypoint decode + NMS) |
| 6 | Skeleton visualization |

> Variants: **n/s come in i8 and fp16; m only ships i8** — the m-pose fp16
> weight in the official SDK package is corrupted (a byte-copy of the .rknn
> file), so it is not included.
> Reference: yolo26m-pose i8 ~158 ms/frame, person confidence 0.93 on the sample.
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(
        files,
        intro="This demo downloads YOLO26-pose n/s/m (m is i8-only) and the sample little_boy.jpg.")
    cells += [
        md("""
## 2. Pick a model and an image
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(
    options=[("yolo26n-pose i8", "yolo26n-pose_i8"),
             ("yolo26n-pose fp16", "yolo26n-pose_fp16"),
             ("yolo26s-pose i8", "yolo26s-pose_i8"),
             ("yolo26s-pose fp16", "yolo26s-pose_fp16"),
             ("yolo26m-pose i8 (only this one)", "yolo26m-pose_i8")],
    value="yolo26m-pose_i8", description="Model:")
w_image = widgets.Dropdown(options=sorted(glob.glob("datasets/*.jpg")),
                           description="Image:", layout=widgets.Layout(width="60%"))
if any(o.endswith("/little_boy.jpg") for o in w_image.options):
    w_image.value = [o for o in w_image.options if o.endswith("/little_boy.jpg")][0]
display(w_model, w_image)
print("Then keep running downwards ▼")
"""),
        md("""
## 3. Pre-processing: letterbox

YOLO expects a 640x640 square input. Stretching would distort body proportions,
so we **resize proportionally and pad with gray**, remembering the scale and
padding — the post-processing needs them to map boxes and keypoints back.
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
print(f"original {img.size} -> input {x.shape}, scale={SCALE:.3f}, pad=({PAD_X},{PAD_Y})")

# show every pre-processing stage: original -> letterbox -> input tensor
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, im, title in [
    (axes[0], img, f"1. original {img.size[0]}x{img.size[1]}"),
    (axes[1], img_lb, "2. letterbox 640x640\\n(aspect resize + gray pad)"),
    (axes[2], x[0], f"3. model input {x.shape}\\nuint8 NHWC"),
]:
    ax.imshow(np.asarray(im))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## 4. RKNN inference
"""),
        code("""
import time
from rknn3 import Rknn3Model

model = Rknn3Model(f"models/yolo26_pose/{w_model.value}.rknn")
print(f"{len(model.output_attrs)} output heads:")
for a in model.output_attrs:
    print("  ", a)

_ = model.infer([x])
t0 = time.time()
outputs = model.infer([x])
print(f"inference time: {(time.time()-t0)*1000:.1f} ms")
"""),
        md("""
## 5. Post-processing: box + keypoint decode

- Each grid cell: the box head gives (l,t,r,b) distances, the score head the
  person confidence
- Keypoints: `x = (raw_x + grid_x + 0.5) * stride`; confidence goes through sigmoid
- Collect all cells, run one NMS, then map coordinates from letterbox space
  back to the original image
"""),
        code("""
from vision_post import (decode_yolo26_pose, unletterbox_boxes,
                         unletterbox_keypoints)

boxes, scores, keypoints = decode_yolo26_pose(outputs, model.output_attrs,
                                              conf_thresh=0.4)
boxes = unletterbox_boxes(boxes, SCALE, PAD_X, PAD_Y, *img.size)
keypoints = unletterbox_keypoints(keypoints, SCALE, PAD_X, PAD_Y)

print(f"{len(boxes)} person(s) detected:")
for b, s, k in zip(boxes, scores, keypoints):
    vis = (k[:, 2] >= 0.5).sum()
    print(f"  person {s:.2f}  box=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})"
          f"  visible keypoints {vis}/17")
"""),
        md("""
## 6. Skeleton visualization

COCO 17 keypoints connected by the skeleton (shoulder-elbow-wrist,
hip-knee-ankle, etc.); white dots = keypoints with confidence >= 0.5.
"""),
        code("""
from vision_post import draw_pose

result = draw_pose(img, boxes, scores, keypoints)
model.close()
result
"""),
        md("""
## Summary

- YOLO26-pose outputs keypoints and boxes in the same forward pass (the
  `pos_*` head's 51 channels = 17 points x 3) — no second pose model needed.
- Decode lives in `vision_post.decode_yolo26_pose` / `unletterbox_keypoints`,
  drawing in `vision_post.draw_pose` (`COCO_SKELETON` defines the 17 bones).
- Try n/s for speed, or a multi-person photo to see the skeletons.
"""),
    ]
    return notebook(cells)


# ================================================================ 05 YOLO26 segmentation

def build_05():
    repo = "yolo26-segment-rknn3-rk1828"
    files = []
    for m in ("n", "s", "m"):
        for q in ("i8", "fp16"):
            for ext in ("rknn", "weight"):
                files.append((repo, f"model/yolo26{m}-seg_{q}.{ext}",
                              f"models/yolo26_seg/yolo26{m}-seg_{q}.{ext}"))
    files.append((repo, "model/coco_80_labels_list.txt",
                  "datasets/coco_80_labels_list.txt"))
    files.append((repo, "model/bus.jpg", "datasets/bus.jpg"))

    cells = [
        md("""
# RK1828 YOLO26-seg Instance Segmentation Demo

Ultralytics **YOLO26-seg**: detection boxes + 80-class scores + **per-instance
pixel masks**, 10 output heads:

| Head | Shape | Meaning |
|---|---|---|
| `box_*` x3 | `[1,4,80,80]` `[1,4,40,40]` `[1,4,20,20]` | box regression (l,t,r,b) |
| `score_*` x3 | `[1,80,...]` | 80-class confidence |
| mask coeffs x3 | `[1,32,...]` | 32-dim mask coefficients per candidate |
| `proto` | `[1,32,160,160]` | mask prototypes |

Instance mask = `sigmoid(32-dim coeffs . proto)`, cropped by its box and
mapped back to the original image. (Same idea as FastSAM in notebook 03,
plus 80 classes.)

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Pick a model and an image |
| 3 | Pre-processing (letterbox: aspect resize + gray pad) |
| 4 | RKNN inference |
| 5 | Post-processing (decode + NMS + mask assembly) |
| 6 | Visualization |

> Reference: yolo26m-seg i8 ~250 ms/frame, 4 persons + 1 bus on the sample.
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(
        files,
        intro="This demo downloads YOLO26-seg n/s/m (i8 and fp16 each), the COCO labels and bus.jpg.")
    cells += [
        md("""
## 2. Pick a model and an image
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(
    options=[f"yolo26{m}-seg_{q}" for m in ("n", "s", "m") for q in ("i8", "fp16")],
    value="yolo26m-seg_i8", description="Model:")
w_image = widgets.Dropdown(options=sorted(glob.glob("datasets/*.jpg")),
                           description="Image:", layout=widgets.Layout(width="60%"))
if any(o.endswith("/bus.jpg") for o in w_image.options):
    w_image.value = [o for o in w_image.options if o.endswith("/bus.jpg")][0]
display(w_model, w_image)

LABELS = open("datasets/coco_80_labels_list.txt").read().splitlines()
print(f"{len(LABELS)} COCO classes. Then keep running downwards ▼")
"""),
        md("""
## 3. Pre-processing: letterbox

Aspect-preserving resize + gray padding to 640x640, remembering the scale and
padding for coordinate restore.
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
print(f"original {img.size} -> input {x.shape}, scale={SCALE:.3f}, pad=({PAD_X},{PAD_Y})")

# show every pre-processing stage: original -> letterbox -> input tensor
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, im, title in [
    (axes[0], img, f"1. original {img.size[0]}x{img.size[1]}"),
    (axes[1], img_lb, "2. letterbox 640x640\\n(aspect resize + gray pad)"),
    (axes[2], x[0], f"3. model input {x.shape}\\nuint8 NHWC"),
]:
    ax.imshow(np.asarray(im))
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## 4. RKNN inference
"""),
        code("""
import time
from rknn3 import Rknn3Model

model = Rknn3Model(f"models/yolo26_seg/{w_model.value}.rknn")
print(f"{len(model.output_attrs)} output heads:")
for a in model.output_attrs:
    print("  ", a)

_ = model.infer([x])
t0 = time.time()
outputs = model.infer([x])
print(f"inference time: {(time.time()-t0)*1000:.1f} ms")
"""),
        md("""
## 5. Post-processing: decode + NMS + mask assembly

1. Per grid cell: decode the box and 80-class scores, keep the cell's 32-dim
   mask coefficients
2. Per-class NMS for dedup
3. `sigmoid(coeffs . proto)` -> 160x160 masks -> upscale to input size, crop
   by box -> map back to the original image
"""),
        code("""
from vision_post import (decode_yolo26_seg, scale_masks, unletterbox_boxes)

boxes, classes, scores, masks = decode_yolo26_seg(outputs, model.output_attrs,
                                                  conf_thresh=0.4)
masks_full = scale_masks(masks, boxes, (img.height, img.width), SCALE, PAD_X, PAD_Y)
boxes = unletterbox_boxes(boxes, SCALE, PAD_X, PAD_Y, *img.size)

print(f"{len(boxes)} objects detected:")
for b, c, s in zip(boxes, classes, scores):
    print(f"  {LABELS[c]:12s} {s:.2f}  box=({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f})")
"""),
        md("""
## 6. Visualization: mask overlay + boxes
"""),
        code("""
from vision_post import draw_boxes, draw_masks

overlay = draw_masks(img, masks_full)
result = draw_boxes(overlay, boxes, classes, scores, LABELS)
model.close()
result
"""),
        md("""
## Summary

- `sigmoid(coeffs . proto)` mask assembly is the shared recipe of YOLOv8-seg /
  FastSAM / YOLO26-seg; proto is computed once and shared by all instances.
- Decode lives in `vision_post.decode_yolo26_seg`, mask mapping in
  `vision_post.scale_masks`.
- Compared with 03 FastSAM: FastSAM is class-agnostic (segment everything),
  YOLO26-seg outputs 80 classes.
"""),
    ]
    return notebook(cells)


# ================================================================ C++ demo shared

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
    """Audio picker cell for the ASR notebooks; `default` pre-selects a file name."""
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
                         description="Audio:", layout=widgets.Layout(width="60%"))
display(w_wav){val}print("Then keep running downwards ▼")
""")

AUDIO_PLAY_MD = md("""
Listen to the audio first:
""")

AUDIO_PLAY_CODE = code("""
from nb_demo import show_audio

show_audio(w_wav.value)
""")


def cpp_intro(title, body):
    return md(f"# {title}\n\n{body}")


# ================================================================ 06 RAFT stereo

def build_06():
    files = repo_files("raft-stereo-rknn3-rk1828", "demos/raft_stereo")
    cells = [
        cpp_intro("RK1828 RAFT-Stereo Disparity Demo",
                  """
[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) (Princeton Vision &
Learning Lab) estimates a **disparity map** from a stereo pair: brighter means
closer. This is the realtime variant, 160x240 input, int8 quantized, about
90 ms/pair on the RK1828.

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Run and view the disparity map |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/raft_stereo"])
    cells += [
        run_helper_code(),
        md("""
## 3. Run the demo

Input: `model/xt_left.jpg` / `model/xt_right.jpg` (a sample stereo pair)
Output: `out.png` disparity heatmap
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
## Summary

- Disparity = horizontal shift of the same point between left and right
  images; larger disparity means closer. It is the core of stereo depth.
- Try your own stereo photos (keep 160x240, or let the demo scale them).
"""),
    ]
    return notebook(cells)


# ================================================================ 07 DA3 depth

def build_07():
    files = repo_files("depth-anything-v3-rknn3-rk1828", "demos/depth_anything_v3")
    cells = [
        cpp_intro("RK1828 Depth Anything V3 Multi-view Depth Demo",
                  """
[Depth Anything V3](https://github.com/ByteDance-Seed/depth-anything-3)
(ByteDance) multi-view depth estimation. The model is split into three
sub-models (local / global / head) with a fixed input of **10 views**, producing
one depth heatmap per view. About 12 s for all 10 views on the RK1828.

| Step | Content |
|---|---|
| 1 | Download models and the 10 sample frames (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Build the image list and run |
| 4 | View the depth maps |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/depth_anything_v3"])
    cells += [
        run_helper_code(),
        md("""
## 3. Build the 10-view image list and run

The repo ships 10 sample frames of the same scene (`model/frame_01.jpg` ~
`frame_10.jpg`); the demo wants a list file with exactly 10 image paths.
"""),
        code("""
import glob, os

DEMO = "demos/depth_anything_v3"
frames = sorted(glob.glob(os.path.join(DEMO, "model/frame_*.jpg")))
assert len(frames) == 10, f"need exactly 10 frames, got {len(frames)}"
with open(os.path.join(DEMO, "image_list.txt"), "w") as f:
    f.write("\\n".join(os.path.abspath(p) for p in frames))  # demo needs absolute paths
show_images(frames[:5], max_width=320)
print(f"{len(frames)} frames, list written to image_list.txt")
"""),
        code("""
run_demo(["./rknn_da3_base_demo",
          "model/da3_base_local.rknn", "model/da3_base_global.rknn",
          "model/da3_base_head.rknn",
          "image_list.txt", "10", "depth_vis"], DEMO)
"""),
        md("""
## 4. View the depth heatmaps (brighter = closer)
"""),
        code("""
import glob

depth_imgs = sorted(glob.glob(os.path.join(DEMO, "depth_vis/*")))
print(f"{len(depth_imgs)} depth maps generated")
show_images(depth_imgs[:5], max_width=320)
"""),
        md("""
## Summary

- Splitting a big model into sub-models, converted and pipelined separately,
  is the standard RKNN approach for very large models.
- With your own photos: exactly 10 views, ideally the same scene from
  different angles.
"""),
    ]
    return notebook(cells)


# ================================================================ 08 Chinese-CLIP

def build_08():
    files = repo_files("chinese-clip-rknn3-rk1828", "demos/chinese_clip")
    cells = [
        cpp_intro("RK1828 Chinese-CLIP Image-Text Matching Demo",
                  """
[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP) (DAMO Academy) matches
an image against several Chinese text candidates and scores them. The text and
vision encoders are converted separately and both run on the RK1828. On the
sample, the correct caption scores ~0.96 vs ~0.04.

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Run image-text matching |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/chinese_clip"])
    cells += [
        run_helper_code(),
        md("""
## 3. Run image-text matching

Input: sample image `model/demo.jpg`; the demo has several built-in Chinese
candidate captions and prints a probability for each.
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
## Summary

- CLIP-style models map images and text into the same vector space; "matching"
  is just similarity — usable for zero-shot classification / text-to-image search.
- The text side of a multimodal model (tokenizer + embed) must be deployed too,
  see `model/chinese_clip_text.*`.
"""),
    ]
    return notebook(cells)


# ================================================================ 09 Qwen3-VL

def build_09():
    files_2b = repo_files("qwen3-vl-2b-rknn3-rk1828", "demos/qwen3_vl_2b")
    files_4b = repo_files("qwen3-vl-4b-rknn3-rk1828", "demos/qwen3_vl_4b")
    files = files_2b + files_4b
    cells = [
        cpp_intro("RK1828 Qwen3-VL Vision-Language Model Demo",
                  """
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) (Tongyi Qianwen): image QA /
captioning. The LLM part is **GRQ W4A16 quantized**, the vision encoder is
fp16, both run on the RK1828.

| Variant | Generation speed (measured) |
|---|---|
| Qwen3-VL-2B | 62.7 tokens/s |
| Qwen3-VL-4B | 38.7 tokens/s |

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co; you may fetch only 2B or 4B) |
| 2 | Extract the prebuilt demo |
| 3 | Enter a prompt and chat with the image |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(
        files,
        intro="Both variants total ~10GB. If disk is tight, edit the download cell to fetch only one variant.")
    cells += extract_cells(["demos/qwen3_vl_2b", "demos/qwen3_vl_4b"])
    cells += [
        run_helper_code(),
        md("""
## 3. Pick a model variant and a prompt
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(options=[("Qwen3-VL-2B (faster)", "2b"),
                                    ("Qwen3-VL-4B (smarter)", "4b")],
                           value="2b", description="Model:")
w_prompt = widgets.Textarea(value="Describe this image",
                            description="Prompt:",
                            layout=widgets.Layout(width="70%", height="60px"))
display(w_model, w_prompt)
print("Then keep running downwards ▼")
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
## Summary

- A VLM = vision encoder (image -> token embeddings) + LLM (autoregressive
  text); both are separate RKNN models.
- `0xff 0xff` are device/session placeholders; `384 384` is the vision input
  resolution.
- Try Chinese questions, counting people in the image, OCR, etc.
"""),
    ]
    return notebook(cells)


# ================================================================ 10 Qwen3-ASR

def build_10():
    files = repo_files("qwen3-asr-rknn3-rk1828", "demos/qwen3_asr") + ASR_WAVS
    cells = [
        cpp_intro("RK1828 Qwen3-ASR Speech Recognition Demo",
                  """
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) (Tongyi Qianwen) supports
**52 languages and dialects** with automatic language detection. Split into two
parts, both on the NPU:

```
audio(wav) -> [Audio Encoder] features -> [LLM] autoregressive text -> transcript
              encoder.rknn (fp16)       llm.rknn (W4A16 quantized)
```

| Step | Content |
|---|---|
| 1 | Download models and sample audio (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Pick an audio clip and listen |
| 4 | Transcribe |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files, intro="Sample audio comes from the other ASR repos, shared under `datasets/asr/`.")
    cells += extract_cells(["demos/qwen3_asr"])
    cells += [
        run_helper_code(),
        md("""
## 3. Pick an audio clip and listen
"""),
        wav_dropdown("test_zh.wav"),
        AUDIO_PLAY_MD,
        AUDIO_PLAY_CODE,
        md("""
## 4. Transcribe

`0xff 0xff` are device/session placeholders. An 8.9 s Chinese clip was
transcribed fully and correctly in our tests.
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
## Summary

- Encoder + LLM decoupled: acoustic features and text generation live in two
  separate RKNN models.
- The tarball also ships `rknn_qwen3_asr_demo_online` (streaming) — give it a try.
"""),
    ]
    return notebook(cells)


# ================================================================ 11 Whisper

def build_11():
    files = repo_files("whisper-base-rknn3-rk1828", "demos/whisper_base") + ASR_WAVS
    cells = [
        cpp_intro("RK1828 Whisper-base Speech Recognition Demo",
                  """
OpenAI [whisper-base](https://github.com/openai/whisper), Chinese/English
recognition. Split into three models: **encoder** + **first-token decoder
(decode0)** + **autoregressive decoder (decode1)**. Measured RTF 0.073 on the
Chinese sample (~14x realtime).

| Step | Content |
|---|---|
| 1 | Download models and sample audio (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Pick language and audio |
| 4 | Transcribe |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/whisper_base"])
    cells += [
        run_helper_code(),
        md("""
## 3. Pick language and audio
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_lang = widgets.Dropdown(options=[("Chinese", "zh"), ("English", "en")],
                          value="zh", description="Language:")
w_wav = widgets.Dropdown(options=sorted(glob.glob("datasets/asr/*.wav")),
                         description="Audio:", layout=widgets.Layout(width="60%"))
if any(o.endswith('/test_zh.wav') for o in w_wav.options):
    w_wav.value = [o for o in w_wav.options if o.endswith('/test_zh.wav')][0]
display(w_lang, w_wav)
print("Then keep running downwards ▼")
"""),
        AUDIO_PLAY_MD,
        AUDIO_PLAY_CODE,
        md("""
## 4. Transcribe
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
## Summary

- Whisper's autoregressive decoder is split into a "first token" model and a
  "later tokens" model — a common NPU optimization for Transformer decoders
  (the first token needs no KV cache).
- Long audio (e.g. `long_en.wav`) is chunked automatically.
"""),
    ]
    return notebook(cells)


# ================================================================ 12 Paraformer

def build_12():
    files = repo_files("paraformer-rknn3-rk1828", "demos/paraformer") + ASR_WAVS
    cells = [
        cpp_intro("RK1828 Paraformer Chinese ASR Demo (with hotwords)",
                  """
Alibaba [Paraformer](https://github.com/modelscope/FunASR) **non-autoregressive**
Chinese ASR: one forward pass emits the whole transcript (no token-by-token
generation like Whisper). Measured RTF 0.046 (~21x realtime). It also supports
**hotword boosting** — given words are recognized preferentially
(asf_scorer + hotword decoder).

| Step | Content |
|---|---|
| 1 | Download models and sample audio (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Pick audio (hotword mode optional) |
| 4 | Transcribe |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/paraformer"])
    cells += [
        run_helper_code(),
        md("""
## 3. Pick audio and mode

- **Normal mode**: `paraformer_encoder` + `paraformer_decoder`
- **Hotword mode**: `paraformer_decoder_hotword` + `asf_scorer` instead,
  boosting the words listed in `model/hotwords.txt`
  (sample hotword audio: datasets/asr/asr_example_hotword.wav)
"""),
        code("""
import glob
import ipywidgets as widgets
from IPython.display import display

w_wav = widgets.Dropdown(options=sorted(glob.glob("datasets/asr/*.wav")),
                         description="Audio:", layout=widgets.Layout(width="60%"))
if any(o.endswith('/asr_example.wav') for o in w_wav.options):
    w_wav.value = [o for o in w_wav.options if o.endswith('/asr_example.wav')][0]
w_hot = widgets.Checkbox(value=False, description="Hotword mode")
display(w_wav, w_hot)
print("Then keep running downwards ▼")
"""),
        AUDIO_PLAY_MD,
        AUDIO_PLAY_CODE,
        md("""
## 4. Transcribe
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
## Summary

- Non-autoregressive = all tokens in parallel, an order of magnitude faster
  than autoregressive ASR — great for realtime.
- Hotwords help with domain terms, names and other long-tail words.
"""),
    ]
    return notebook(cells)


# ================================================================ 13 SenseVoice

def build_13():
    files = repo_files("sensevoice-small-rknn3-rk1828", "demos/sensevoice") + ASR_WAVS
    cells = [
        cpp_intro("RK1828 SenseVoice-Small Multilingual ASR Demo",
                  """
Alibaba [SenseVoice-Small](https://github.com/FunAudioLLM/SenseVoice)
multilingual ASR. Besides the transcript, the output **inlines language,
emotion and sound-event tags** (e.g. `<|zh|><|HAPPY|><|Speech|>`). Measured
~292 ms per clip.

| Step | Content |
|---|---|
| 1 | Download models and sample audio (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Pick an audio clip |
| 4 | Transcribe (see the rich tags) |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/sensevoice"])
    cells += [
        run_helper_code(),
        md("""
## 3. Pick an audio clip
"""),
        wav_dropdown("sensevoice_zh.wav"),
        AUDIO_PLAY_MD,
        AUDIO_PLAY_CODE,
        md("""
## 4. Transcribe
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
## Summary

- SenseVoice generates "language/emotion/event" as special tokens, so one
  inference gives you the transcript plus paralinguistic info.
- The model input is fixed at 10 s; shorter audio is padded automatically.
"""),
    ]
    return notebook(cells)


# ================================================================ 14 forced aligner

def build_14():
    files = repo_files("qwen3-forced-aligner-rknn3-rk1828", "demos/forced_aligner")
    cells = [
        cpp_intro("RK1828 Qwen3-ForcedAligner Demo",
                  """
[Qwen3-ForcedAligner](https://github.com/QwenLM/Qwen3-ForcedAligner): given
**audio + known text**, output a timestamp per character/word (TSV), Chinese
and English, measured RTF 0.041. Typical uses: subtitle generation, lyrics
alignment, speech-data labeling.

| Step | Content |
|---|---|
| 1 | Download models and sample audio (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Enter the text and run |
| 4 | View the timestamps |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/forced_aligner"])
    cells += [
        run_helper_code(),
        md("""
## 3. Enter the text to align

The repo ships a sample `audio.wav` (Chinese). You can also use your own
wav + matching transcript.
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_wav = widgets.Text(value="audio.wav", description="Audio:",
                     layout=widgets.Layout(width="60%"))
w_text = widgets.Textarea(value="欢迎大家来到么哒社区进行体验",
                          description="Text:",
                          layout=widgets.Layout(width="70%", height="50px"))
w_lang = widgets.Dropdown(options=["Chinese", "English"], description="Language:")
display(w_wav, w_text, w_lang)
print("Then keep running downwards ▼")
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
## 4. View the timestamps (TSV: start s / end s / token)
"""),
        code("""
import csv

rows = list(csv.reader(open("demos/forced_aligner/result.tsv"), delimiter="\\t"))
if rows and rows[0][0].lower().startswith("start"):
    rows = rows[1:]  # drop the header start_ms / end_ms / text
rows = [(float(r[0]) / 1000, float(r[1]) / 1000, r[2]) for r in rows]  # ms -> s
print(f"{'start(s)':>8} {'end(s)':>8}  token")
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
plt.title("alignment over the waveform")
plt.xlabel("seconds")
plt.tight_layout()
plt.show()
"""),
        md("""
## Summary

- Alignment is not recognition: the text is known, the model only **pins each
  token onto the timeline**.
- The waveform view makes the span-to-sound correspondence clearly visible.
"""),
    ]
    return notebook(cells)


# ================================================================ 15 VITS TTS

def build_15():
    files = repo_files("vits-tts-rknn3-rk1828", "demos/vits_tts")
    cells = [
        cpp_intro("RK1828 VITS Text-to-Speech Demo",
                  """
[VITS](https://github.com/jaywalnut310/vits) end-to-end English TTS. Two
pretrained models:

| Model | Speakers |
|---|---|
| LJSpeech | single female voice |
| VCTK | 109 speakers (choose with `--speaker_id`) |

Each model is split into step1 (text encoding + durations) and step2
(flow + vocoder), running with a sliding window; LJSpeech measured RTF 0.345.
Phoneme data lives under `model/espeak-ng-data/`.

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Enter English text and synthesize |
| 4 | Listen |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/vits_tts"])
    cells += [
        run_helper_code(),
        md("""
## 3. Enter text and synthesize
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_model = widgets.Dropdown(options=[("LJSpeech single female voice", "ljs"),
                                    ("VCTK multi-speaker", "vctk")],
                           description="Model:")
w_speaker = widgets.IntSlider(value=4, min=0, max=108,
                              description="Speaker ID:",
                              style={"description_width": "80px"})
w_text = widgets.Textarea(value="VITS is awesome! Have a nice day.",
                          description="Text:",
                          layout=widgets.Layout(width="70%", height="50px"))
display(w_model, w_speaker, w_text)
print("Then keep running downwards ▼ (Speaker ID only works for VCTK)")
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
## 4. Listen to the result
"""),
        code("""
show_audio("demos/vits_tts/out.wav")
"""),
        md("""
## Summary

- TTS pipeline: text -> phonemes (espeak-ng) -> step1 durations/acoustic
  features -> step2 waveform.
- Try a few speaker_id values to compare voices.
"""),
    ]
    return notebook(cells)


# ================================================================ 16 Embedding

def build_16():
    files = repo_files("qwen3-embedding-0.6b-rknn3-rk1828", "demos/qwen3_embedding")
    cells = [
        cpp_intro("RK1828 Qwen3-Embedding Text Vector Demo",
                  """
[Qwen3-Embedding-0.6B](https://github.com/QwenLM/Qwen3-Embedding): encodes
text into **1024-dim vectors** for semantic search / RAG retrieval. GRQ W4A16
quantized, measured 76.7 ms prefill per sentence; paraphrase cosine similarity
0.958 vs 0.283 for an unrelated text.

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Enter sentences and batch-encode |
| 4 | Cosine similarity matrix |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/qwen3_embedding"])
    cells += [
        run_helper_code(),
        md("""
## 3. Enter sentences and encode

One sentence per line. The demo encodes one sentence per run and writes the
1024-dim float32 vector to `embedding_out.bin`.
"""),
        code("""
import ipywidgets as widgets
from IPython.display import display

w_texts = widgets.Textarea(
    value="How do I reset my password?\\n"
          "I forgot my password, how can I change it?\\n"
          "What's the best way to cook rice?\\n"
          "如何重置我的密码？",
    description="Sentences (one per line):", layout=widgets.Layout(width="75%", height="120px"))
display(w_texts)
print("Then keep running downwards ▼")
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
print("embedding matrix:", embs.shape)
"""),
        md("""
## 4. Cosine similarity matrix

Semantically close sentences (even across languages) should score near 1,
unrelated ones near 0.
"""),
        code("""
import matplotlib.pyplot as plt
from vision_post import setup_cjk_font

setup_cjk_font()
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
plt.title("cosine similarity")
plt.tight_layout()
plt.show()
"""),
        md("""
## Summary

- An embedding model only measures "semantic distance", regardless of
  language — try mixed Chinese/English paraphrases.
- This is the retrieval side of a RAG system; pair it with the Reranker in
  notebook 17 for the full retrieve + rerank pipeline.
"""),
    ]
    return notebook(cells)


# ================================================================ 17 Reranker

def build_17():
    files = repo_files("qwen3-reranker-0.6b-rknn3-rk1828", "demos/qwen3_reranker")
    cells = [
        cpp_intro("RK1828 Qwen3-Reranker Demo",
                  """
[Qwen3-Reranker-0.6B](https://github.com/QwenLM/Qwen3-Embedding): scores a
(query, document) pair with a 0~1 relevance score for RAG reranking. GRQ W4A16
quantized, measured 132 ms prefill per pair; relevant 0.9985 vs irrelevant 0.0000.

| Step | Content |
|---|---|
| 1 | Download models (from huggingface.co) |
| 2 | Extract the prebuilt demo |
| 3 | Enter a query and candidate documents |
| 4 | Score and rank |
"""),
        ENV_MD, ENV_CODE,
    ]
    cells += download_cells(files)
    cells += extract_cells(["demos/qwen3_reranker"])
    cells += [
        run_helper_code(),
        md("""
## 3. Enter a query and candidate documents (one per line)
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
    description="Docs (one per line):", layout=widgets.Layout(width="75%", height="120px"))
display(w_query, w_docs)
print("Then keep running downwards ▼")
"""),
        md("""
## 4. Score and rank
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
    if not nums:  # fallback: take the last 0~1 decimal in the output
        nums = re.findall(r"(?<![\\d.])(0?\\.[0-9]+|1\\.0+)(?![\\d.])", p.stdout)
    scores.append(float(nums[-1]))
    print(f"  -> {scores[-1]:.4f}")

order = np.argsort(scores)[::-1]
print("\\nRanked by relevance:")
for i in order:
    print(f"  {scores[i]:.4f}  {docs[i][:60]}")
"""),
        code("""
import matplotlib.pyplot as plt
from vision_post import setup_cjk_font

setup_cjk_font()
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
## Summary

- A reranker is the "fine ranking" stage of RAG: embeddings shortlist top-K,
  the reranker re-scores them one by one for precision.
- The score gap between relevant and irrelevant is extreme (0.99 vs 0.00),
  showing excellent discrimination.
"""),
    ]
    return notebook(cells)


# ================================================================ main

BUILDERS = {
    "01_classification.ipynb": build_01,
    "02_yolo_detection.ipynb": build_02,
    "03_FastSAM.ipynb": build_03,
    "04_yolo26_pose.ipynb": build_04,
    "05_yolo26_segment.ipynb": build_05,
    "06_RAFT_Stereo.ipynb": build_06,
    "07_DepthAnythingV3.ipynb": build_07,
    "08_QAClip.ipynb": build_08,
    "09_Qwen3_VL.ipynb": build_09,
    "10_Qwen3_ASR.ipynb": build_10,
    "11_whisper.ipynb": build_11,
    "12_paraformer.ipynb": build_12,
    "13_sensevoice.ipynb": build_13,
    "14_forced_aligner.ipynb": build_14,
    "15_vits_tts.ipynb": build_15,
    "16_Qwen3_Embedding.ipynb": build_16,
    "17_Qwen3_Reranker.ipynb": build_17,
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
