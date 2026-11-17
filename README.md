# rk1828-notebook

在 **Rockchip RK1828 AI 协处理器**（PCIe，20 TOPS NPU）上运行的 16 个交互式
Jupyter notebook demo，覆盖视觉、语音和多模态模型，全部为 **RKNN3 格式**
（`.rknn` 计算图 + `.weight` 权重分离），开箱即跑。

## 快速开始

```bash
# 1. 安装 RKNN3 驱动/运行时（rknn3-rk182x DEB 包），验证:
sudo rknn-smi info        # 应能看到 RK1828

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 启动 Jupyter，打开任意 notebook 从上到下运行
jupyter lab   # 或 jupyter notebook
```

每个 notebook 的**第一步都是下载模型**：下拉框选择下载源
（**HF-Mirror 国内镜像** 或 **Hugging Face 官方**），
已存在的文件自动跳过、不会重复下载。
模型全部托管在 [huggingface.co/qhddz](https://huggingface.co/qhddz)。

也可以在命令行一次性下载全部 notebook 的模型（约 25GB）：

```bash
bash download_models.sh            # 默认 HF-Mirror 源
bash download_models.sh hf         # Hugging Face 官方源
```

## Notebook 列表

| # | Notebook | 模型 | 类型 |
|---|---|---|---|
| 01 | 01_classification | MobileNetV2 / ResNet50-v2 (i8/fp16) | 图像分类（Python 推理） |
| 02 | 02_yolo_detection | YOLOv5/v6/v8, YOLO26 n/s/m | 目标检测（Python 推理） |
| 03 | 03_FastSAM | FastSAM-s | 实例分割（Python 推理） |
| 04 | 04_yolo26_pose_segment | YOLO26-pose / YOLO26-seg | 姿态估计 + 分割 |
| 05 | 05_RAFT_Stereo | RAFT-Stereo realtime | 双目视差 |
| 06 | 06_DepthAnythingV3 | Depth Anything V3 (base) | 多视角深度估计 |
| 07 | 07_QAClip | Chinese-CLIP | 中文图文匹配 |
| 08 | 08_Qwen3_VL | Qwen3-VL-2B / 4B | 视觉语言大模型 |
| 09 | 09_Qwen3_ASR | Qwen3-ASR-0.6B | 语音识别（52 语种） |
| 10 | 10_whisper | whisper-base | 语音识别（中英文） |
| 11 | 11_paraformer | Paraformer | 中文 ASR（热词） |
| 12 | 12_sensevoice | SenseVoice-Small | 多语言 ASR（情感/事件标签） |
| 13 | 13_forced_aligner | Qwen3-ForcedAligner-0.6B | 强制对齐（字级时间戳） |
| 14 | 14_vits_tts | VITS (LJSpeech / VCTK) | 英文语音合成 |
| 15 | 15_Qwen3_Embedding | Qwen3-Embedding-0.6B | 文本向量（RAG 检索） |
| 16 | 16_Qwen3_Reranker | Qwen3-Reranker-0.6B | 文档重排序（RAG 精排） |

01-03 用 Python 直接调用运行时（`rknn3.py`，ctypes 封装），逐步展示
前处理/推理/后处理；04-16 调用官方 `rknn3-model-zoo` 流程编译的 x86_64 C++ demo。

## 目录结构

```
├── 01_classification.ipynb ... 16_Qwen3_Reranker.ipynb
├── rknn3.py            # librknn3_api.so 的 ctypes 封装（Rknn3Model）
├── nb_download.py      # 模型下载（HF / HF-Mirror 源选择，跳过已存在）
├── nb_demo.py          # C++ demo 运行/结果展示辅助
├── vision_post.py      # YOLO/FastSAM 前后处理（letterbox/NMS/解码/掩码）
├── download_models.sh  # 命令行批量下载全部模型
├── _build_notebooks.py # notebook 生成器（改动后重新生成所有 .ipynb）
├── models/             # 模型文件（下载后生成）
├── datasets/           # 示例图片/音频（下载后生成）
└── demos/              # C++ demo 程序 + 模型（下载后生成）
```

## 模型来源与许可

模型为官方 `rknn3-model-zoo` 流程转换的 RKNN3 产物，已在真实 RK1828 上验证。
各模型的许可以其上游仓库为准（YOLO 系为 AGPL-3.0，Qwen 系为 Apache-2.0，
SenseVoice/VITS 为 MIT，RAFT-Stereo 为 BSD-3-Clause 等），
转换成 RKNN3 格式不改变其许可。详见各 HF 仓库的 README。
