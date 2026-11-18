# -*- coding: utf-8 -*-
"""Shared pre/post-processing for the vision models (YOLO family / FastSAM).

Used by notebooks 01-05. Everything is pure numpy/PIL.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- plotting

def setup_cjk_font():
    """Configure a CJK-capable matplotlib font (keeps default if none found)."""
    import matplotlib
    from matplotlib import font_manager
    # matplotlib registers only the first face of a .ttc (usually JP);
    # JP fonts also contain simplified-Chinese glyphs
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


# ---------------------------------------------------------------- pre-processing

def letterbox(img, size=640, pad=(114, 114, 114)):
    """Aspect-preserving resize + gray padding to size x size.

    Returns (padded PIL.Image, scale, pad_x, pad_y).
    Coordinate restore: x_orig = (x_model - pad_x) / scale
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
    """PIL.Image -> [1,H,W,3] uint8 (NHWC; normalization is baked into the model)."""
    return np.asarray(img.convert("RGB"))[None]


# ---------------------------------------------------------------- dequant

def dequant(arr, attr):
    """Restore a float32 tensor using the tensor's quantization params.
    Non-quantized tensors are returned as float32 unchanged."""
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
    """NMS. Per-class when `classes` is given; returns indices sorted by score desc."""
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
    """Map boxes from the model-input coordinate system back to the original image."""
    boxes = boxes.copy().astype(np.float32)
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / scale, 0, orig_w)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / scale, 0, orig_h)
    return boxes


# ---------------------------------------------------------------- YOLO decoding

def _group_heads(outputs, output_attrs):
    """Group YOLO26 output heads by grid size: {grid_h: {channels: array[c,h,w]}}."""
    branches = {}
    for arr, attr in zip(outputs, output_attrs):
        a = dequant(arr, attr)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        c, h, w = a.shape
        branches.setdefault(h, {})[c] = a
    return branches


def _decode_boxes_ltrb(box_head):
    """YOLO26 regresses (l,t,r,b) directly (no DFL); restore to input-space xyxy [h*w,4]."""
    gh, gw = box_head.shape[1], box_head.shape[2]
    stride = 640 // gh
    row, col = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    x1 = (-box_head[0] + col + 0.5) * stride
    y1 = (-box_head[1] + row + 0.5) * stride
    x2 = (box_head[2] + col + 0.5) * stride
    y2 = (box_head[3] + row + 0.5) * stride
    return np.stack([x1, y1, x2, y2], -1).reshape(-1, 4)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def decode_builtin_nms(output, conf_thresh=0.4):
    """YOLOv5/v6/v8: decode+NMS is built into the model; output is
    [1,K,6] = (score, class, x1, y1, x2, y2).

    Returns (boxes[K,4], classes[K], scores[K]) in model-input coordinates.
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
    """YOLO26: 6 raw heads (box + score at strides 8/16/32), decoded on the host.

    Returns (boxes[K,4], classes[K], scores[K]) in model-input coordinates.
    """
    branches = _group_heads(outputs, output_attrs)
    all_boxes, all_scores, all_classes = [], [], []
    for h in sorted(branches, reverse=True):
        pair = branches[h]
        box_head, score_head = pair[4], pair[num_classes]
        boxes = _decode_boxes_ltrb(box_head)

        prob = score_head
        if prob.min() < 0 or prob.max() > 1:  # logits -> probabilities
            prob = _sigmoid(prob)
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


def decode_yolo26_pose(outputs, output_attrs, conf_thresh=0.4, nms_thresh=0.7,
                       num_keypoints=17):
    """YOLO26-pose: per branch box[1,4,h,w] + score[1,1,h,w] + kpt[1,K*3,h,w].

    Returns (boxes[K,4] xyxy, scores[K], keypoints[K,17,3]) in model-input
    coordinates; keypoints[...,2] is the sigmoided keypoint confidence.
    Pose detects a single class (person).
    """
    kc = num_keypoints * 3
    branches = _group_heads(outputs, output_attrs)
    all_boxes, all_scores, all_kpts = [], [], []
    for h in sorted(branches, reverse=True):
        pair = branches[h]
        if 4 not in pair or 1 not in pair or kc not in pair:
            continue
        box_head, score_head, kpt_head = pair[4], pair[1], pair[kc]
        gh, gw = box_head.shape[1], box_head.shape[2]
        stride = 640 // gh
        row, col = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")

        boxes = _decode_boxes_ltrb(box_head)
        prob = score_head[0]
        if prob.min() < 0 or prob.max() > 1:  # logits -> probabilities
            prob = _sigmoid(prob)
        keep = prob.ravel() >= conf_thresh

        # keypoints: channels 3k/3k+1 are offsets from the grid center,
        # 3k+2 is the confidence logit
        kx = (kpt_head[0::3] + col + 0.5) * stride            # [17,h,w]
        ky = (kpt_head[1::3] + row + 0.5) * stride
        kp = _sigmoid(kpt_head[2::3])
        kpts = np.stack([kx, ky, kp], -1)                     # [17,h,w,3]
        kpts = kpts.transpose(1, 2, 0, 3).reshape(-1, num_keypoints, 3)

        all_boxes.append(boxes[keep])
        all_scores.append(prob.ravel()[keep])
        all_kpts.append(kpts[keep])

    if not all_boxes:
        return (np.zeros((0, 4)), np.zeros(0),
                np.zeros((0, num_keypoints, 3)))
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    kpts = np.concatenate(all_kpts)
    keep = nms(boxes, scores, nms_thresh)
    return boxes[keep], scores[keep], kpts[keep]


def decode_yolo26_seg(outputs, output_attrs, conf_thresh=0.4, nms_thresh=0.7,
                      num_classes=80, mask_dim=32):
    """YOLO26-seg: per branch box[1,4,h,w] + score[1,80,h,w] + mask
    coefficients[1,32,h,w], plus one proto head [1,32,160,160].

    Returns (boxes[K,4], classes[K], scores[K], masks[K,160,160]) in
    model-input coordinates.
    """
    branches = _group_heads(outputs, output_attrs)
    proto = None
    for h, pair in branches.items():
        if mask_dim in pair and 4 not in pair:  # only the proto head has no box
            proto = pair[mask_dim]
    if proto is None:
        raise ValueError("mask proto head [1,32,160,160] not found")

    all_boxes, all_scores, all_classes, all_coeffs = [], [], [], []
    for h in sorted(branches, reverse=True):
        pair = branches[h]
        if 4 not in pair or num_classes not in pair or mask_dim not in pair:
            continue
        box_head, score_head, mask_head = pair[4], pair[num_classes], pair[mask_dim]

        boxes = _decode_boxes_ltrb(box_head)
        prob = score_head
        if prob.min() < 0 or prob.max() > 1:  # logits -> probabilities
            prob = _sigmoid(prob)
        prob = prob.transpose(1, 2, 0).reshape(-1, num_classes)
        scores = prob.max(1)
        classes = prob.argmax(1)
        keep = scores >= conf_thresh

        coeffs = mask_head.transpose(1, 2, 0).reshape(-1, mask_dim)
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
        all_classes.append(classes[keep])
        all_coeffs.append(coeffs[keep])

    if not all_boxes:
        return (np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0),
                np.zeros((0, *proto.shape[1:])))
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_classes)
    coeffs = np.concatenate(all_coeffs)
    keep = nms(boxes, scores, nms_thresh, classes)
    boxes, scores, classes, coeffs = boxes[keep], scores[keep], classes[keep], coeffs[keep]

    masks = _sigmoid(coeffs @ proto.reshape(mask_dim, -1))
    masks = masks.reshape(len(boxes), *proto.shape[1:])  # [K,160,160]
    return boxes, classes, scores, masks


def unletterbox_keypoints(kpts, scale, pad_x, pad_y):
    """Map keypoints from model-input coordinates back to the original image
    (kpts [...,3], last dim = x,y,conf)."""
    kpts = kpts.copy().astype(np.float32)
    kpts[..., 0] = (kpts[..., 0] - pad_x) / scale
    kpts[..., 1] = (kpts[..., 1] - pad_y) / scale
    return kpts


# ---------------------------------------------------------------- FastSAM decoding

def decode_fastsam(outputs, conf_thresh=0.4, nms_thresh=0.7, mask_thresh=0.5):
    """FastSAM-s post-processing.

    output0: [1,37,8400] = cx,cy,w,h,conf + 32 mask coefficients (decoded,
    input scale)
    output1: [1,32,160,160] mask prototypes
    Returns (boxes[K,4] xyxy, scores[K], masks[K,160,160] float) in
    model-input coordinates.
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
    """Upscale [K,160,160] masks to input size, crop to the box, then map
    back to the original image size."""
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
        # strip the letterbox padding, then resize to the original image
        ox1, oy1 = pad_x, pad_y
        ox2 = input_size - (input_size - int(round(out_shape[1] * scale)) - pad_x)
        oy2 = input_size - (input_size - int(round(out_shape[0] * scale)) - pad_y)
        body = crop[oy1:oy2, ox1:ox2]
        out[i] = cv2_resize(body, (out_shape[1], out_shape[0]))
    return out


# ---------------------------------------------------------------- drawing

_PALETTE = [(255, 56, 56), (255, 157, 51), (255, 225, 52), (87, 210, 60),
            (0, 210, 210), (62, 120, 255), (168, 62, 255), (255, 62, 168)]

# COCO 17 keypoints: 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear,
# 5 left shoulder, 6 right shoulder, 7 left elbow, 8 right elbow, 9 left wrist,
# 10 right wrist, 11 left hip, 12 right hip, 13 left knee, 14 right knee,
# 15 left ankle, 16 right ankle
COCO_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
                 (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                 (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def draw_boxes(img, boxes, classes, scores, labels=None):
    """Draw detection boxes + labels on a PIL.Image; returns a new image."""
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


def draw_pose(img, boxes, scores, keypoints, kpt_thresh=0.5):
    """Draw person boxes + COCO 17-keypoint skeletons on a PIL.Image."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    for i, (box, score, kpts) in enumerate(zip(boxes, scores, keypoints)):
        color = _PALETTE[i % len(_PALETTE)]
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"person {score:.2f}"
        tw = draw.textlength(text, font=font)
        draw.rectangle([x1, y1 - 18, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - 16), text, fill=(255, 255, 255), font=font)
        for a, b in COCO_SKELETON:
            if kpts[a, 2] >= kpt_thresh and kpts[b, 2] >= kpt_thresh:
                draw.line([kpts[a, 0], kpts[a, 1], kpts[b, 0], kpts[b, 1]],
                          fill=color, width=3)
        for x, y, c in kpts:
            if c >= kpt_thresh:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3],
                             fill=(255, 255, 255), outline=color, width=2)
    return img


def draw_masks(img, masks, alpha=0.45):
    """Overlay binary/probability masks as translucent colors on a PIL.Image."""
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    for i, m in enumerate(masks):
        color = np.array(_PALETTE[i % len(_PALETTE)], dtype=np.float32)
        region = m > 0.5
        base[region] = base[region] * (1 - alpha) + color * alpha
    return Image.fromarray(base.astype(np.uint8))
