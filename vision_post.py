# -*- coding: utf-8 -*-
"""视觉模型（YOLO 系列 / FastSAM）的通用前后处理。

notebook 01-03 共用。所有函数都是纯 numpy/PIL 实现。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 绘图

def setup_cjk_font():
    """配置 matplotlib 中文字体（找不到 CJK 字体时保持默认）。"""
    import matplotlib
    from matplotlib import font_manager
    # matplotlib 对 .ttc 只注册第一个字模（通常是 JP），JP 字体也含简体中文字形
    for f in ("Noto Sans CJK JP", "Noto Serif CJK JP",
              "Noto Sans CJK SC", "Noto Serif CJK SC",
              "WenQuanYi Zen Hei", "WenQuanYi Micro Hei"):
        try:
            font_manager.findfont(f, fallback_to_default=False)
            matplotlib.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            break
        except Exception:  # noqa: BLE001
            continue
    matplotlib.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------- 前处理

def letterbox(img, size=640, pad=(114, 114, 114)):
    """等比缩放 + 灰边填充到 size×size。

    返回 (填充后的 PIL.Image, scale, pad_x, pad_y)。
    坐标还原: x_orig = (x_model - pad_x) / scale
    """
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(round(w * scale), 1), max(round(h * scale), 1)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), pad)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def to_input_array(img):
    """PIL.Image -> [1,H,W,3] uint8（NHWC，模型内置归一化）。"""
    return np.asarray(img.convert("RGB"))[None]


# ---------------------------------------------------------------- 反量化

def dequant(arr, attr):
    """按输出 tensor 的量化参数还原成 float32。非量化输出原样返回。"""
    arr = np.asarray(arr)
    if getattr(attr, "dtype", None) == 2 and attr.qnt_info.scale != 0:  # INT8
        return (arr.astype(np.float32) - attr.qnt_info.zero_point) * attr.qnt_info.scale
    return arr.astype(np.float32)


# ---------------------------------------------------------------- NMS

def box_iou(box, boxes):
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a + b - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes, scores, iou_thresh=0.45, classes=None):
    """NMS。classes 不为 None 时按类别分别做；返回按分数降序的下标。"""
    if classes is None:
        classes = np.zeros(len(boxes), dtype=np.int64)
    keep = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        order = idx[np.argsort(scores[idx])[::-1]]
        while order.size:
            cur = order[0]
            keep.append(cur)
            if order.size == 1:
                break
            iou = box_iou(boxes[cur], boxes[order[1:]])
            order = order[1:][iou <= iou_thresh]
    keep = np.array(keep, dtype=np.int64)
    return keep[np.argsort(scores[keep])[::-1]] if keep.size else keep


def unletterbox_boxes(boxes, scale, pad_x, pad_y, orig_w, orig_h):
    """把模型输入坐标系下的框映射回原图。"""
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / scale, 0, orig_w)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / scale, 0, orig_h)
    return boxes


# ---------------------------------------------------------------- YOLO 解码

def decode_builtin_nms(output, conf_thresh=0.4):
    """YOLOv5/v6/v8：模型内置 decode+NMS，输出 [1,K,6] = (score, class, x1, y1, x2, y2)。

    返回 (boxes[K,4], classes[K], scores[K])（模型输入坐标系）。
    """
    det = np.asarray(output, dtype=np.float32)
    if det.ndim == 3:
        det = det[0]
    if det.shape[0] == 6 and det.shape[1] != 6:
        det = det.T
    valid = det[:, 0] >= conf_thresh
    det = det[valid]
    if len(det) == 0:
        return np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0)
    return det[:, 2:6], det[:, 1].astype(np.int64), det[:, 0]


def decode_yolo26_heads(outputs, output_attrs, conf_thresh=0.4, nms_thresh=0.7,
                        num_classes=80):
    """YOLO26：6 个原始检测头（box + score，stride 8/16/32），host 端解码。

    返回 (boxes[K,4], classes[K], scores[K])（模型输入坐标系）。
    """
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    # 按网格大小分组: box 头 shape=[1,4,h,w]，score 头 shape=[1,80,h,w]
    branches = {}
    for arr, attr in zip(outputs, output_attrs):
        a = dequant(arr, attr)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        c, h, w = a.shape
        branches.setdefault(h, {})[c] = a  # c=4 -> box, c=80 -> score

    all_boxes, all_scores, all_classes = [], [], []
    for h in sorted(branches, reverse=True):
        pair = branches[h]
        box_head, score_head = pair[4], pair[num_classes]
        gh, gw = box_head.shape[1], box_head.shape[2]
        stride = 640 // gh
        row, col = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        # YOLO26 直接回归 (l,t,r,b)，无 DFL
        x1 = (-box_head[0] + col + 0.5) * stride
        y1 = (-box_head[1] + row + 0.5) * stride
        x2 = (box_head[2] + col + 0.5) * stride
        y2 = (box_head[3] + row + 0.5) * stride
        boxes = np.stack([x1, y1, x2, y2], -1).reshape(-1, 4)

        prob = score_head
        if prob.min() < 0 or prob.max() > 1:  # logits -> 概率
            prob = sigmoid(prob)
        prob = prob.transpose(1, 2, 0).reshape(-1, num_classes)
        scores = prob.max(1)
        classes = prob.argmax(1)
        keep = scores >= conf_thresh
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
        all_classes.append(classes[keep])

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)
    keep = nms(boxes, scores, nms_thresh, classes)
    return boxes[keep], classes[keep], scores[keep]


# ---------------------------------------------------------------- FastSAM 解码

def decode_fastsam(outputs, conf_thresh=0.4, nms_thresh=0.7, mask_thresh=0.5):
    """FastSAM-s 后处理。

    output0: [1,37,8400] = cx,cy,w,h,conf + 32 个 mask 系数（已解码到输入尺度）
    output1: [1,32,160,160] mask prototypes
    返回 (boxes[K,4] xyxy, scores[K], masks[K,160,160] float)（模型输入坐标系）。
    """
    det = np.asarray(outputs[0], dtype=np.float32)
    if det.ndim == 3:
        det = det[0]          # [37, 8400]
    det = det.T               # [8400, 37]
    protos = np.asarray(outputs[1], dtype=np.float32)
    if protos.ndim == 4:
        protos = protos[0]    # [32, 160, 160]

    keep = det[:, 4] >= conf_thresh
    det = det[keep]
    if len(det) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, *protos.shape[1:]))
    cx, cy, w, h = det[:, 0], det[:, 1], det[:, 2], det[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    scores = det[:, 4]
    coeffs = det[:, 5:]                                   # [K, 32]
    idx = nms(boxes, scores, nms_thresh)
    boxes, scores, coeffs = boxes[idx], scores[idx], coeffs[idx]

    masks = 1.0 / (1.0 + np.exp(-np.clip(coeffs @ protos.reshape(32, -1), -60, 60)))
    masks = masks.reshape(len(boxes), *protos.shape[1:])  # [K,160,160]
    return boxes, scores, masks


def scale_masks(masks, boxes, out_shape, scale, pad_x, pad_y, input_size=640):
    """把 [K,160,160] 的 mask 放大到输入尺寸、按框裁剪，再映射回原图尺寸。"""
    from cv2 import resize as cv2_resize
    k = len(masks)
    up = np.stack([cv2_resize(m, (input_size, input_size)) for m in masks]) if k else masks
    out = np.zeros((k, out_shape[0], out_shape[1]), dtype=np.float32)
    for i in range(k):
        x1, y1, x2, y2 = boxes[i]
        crop = np.zeros((input_size, input_size), dtype=np.float32)
        xi1, yi1 = int(max(x1, 0)), int(max(y1, 0))
        xi2, yi2 = int(min(x2, input_size)), int(min(y2, input_size))
        crop[yi1:yi2, xi1:xi2] = up[i][yi1:yi2, xi1:xi2]
        # 去掉 letterbox 灰边再缩放回原图
        ox1, oy1 = pad_x, pad_y
        ox2 = input_size - (input_size - int(round(out_shape[1] * scale)) - pad_x)
        oy2 = input_size - (input_size - int(round(out_shape[0] * scale)) - pad_y)
        body = crop[oy1:oy2, ox1:ox2]
        out[i] = cv2_resize(body, (out_shape[1], out_shape[0]))
    return out


# ---------------------------------------------------------------- 画图

_PALETTE = [(255, 56, 56), (255, 157, 51), (255, 225, 52), (87, 210, 60),
            (0, 210, 210), (62, 120, 255), (168, 62, 255), (255, 62, 168)]


def draw_boxes(img, boxes, classes, scores, labels=None):
    """在 PIL.Image 上画检测框 + 标签，返回新图。"""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        color = _PALETTE[int(cls) % len(_PALETTE)]
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = labels[int(cls)] if labels is not None and int(cls) < len(labels) else str(int(cls))
        text = f"{name} {score:.2f}"
        tw = draw.textlength(text, font=font)
        draw.rectangle([x1, y1 - 18, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - 16), text, fill=(255, 255, 255), font=font)
    return img


def draw_masks(img, masks, alpha=0.45):
    """把二值/概率 mask 以半透明色块叠加到 PIL.Image 上。"""
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    for i, m in enumerate(masks):
        color = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)
        region = m > 0.5
        base[region] = base[region] * (1 - alpha) + color * alpha
    return Image.fromarray(base.astype(np.uint8))
