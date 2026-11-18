# rk1828-notebook

[English](README.md) | 中文

在 **Rockchip RK1828 AI 协处理器**（PCIe，20 TOPS NPU）上运行的 16 个交互式
Jupyter notebook demo，覆盖视觉、语音和多模态模型。所有模型均为 **RKNN3 格式**
（`.rknn` 计算图 + `.weight` 权重分离），已在真实 RK1828 硬件上逐个验证，开箱即跑。

## 环境安装

### 1. 硬件要求

- x86_64 主机（Linux，建议 Ubuntu 20.04+）
- RK1820 / RK1828 AI 协处理器卡（PCIe 接口：SODIMM、M.2 2280 或 M.2 2242 版型）
- 磁盘空间：全部模型约 **25GB**（也可只下载单个 notebook 需要的模型）

### 2. 安装 RKNN3 驱动 / 固件 / 运行时

RK1828 通过 PCIe 与主机通信，需要先安装 DKMS 驱动、固件和 `librknn3_api` 运行时。
这些都在瑞芯微官方 SDK（`rknn3-rk182x`）中：

```bash
# 方式一（推荐）：SDK 自带的一键脚本，自动装依赖 + DEB + 固件版型自适应
sudo /path/to/rk1828-x86-v1.2.0/install_all.sh

# 方式二：手动安装 x86_64 DEB 包（含驱动/固件/Runtime/SMI）
sudo dpkg -i rknn3-rk182x-sodimm_v1.2.0_x86_64.deb || sudo apt-get -f install -y
sudo modprobe pcie-rkep
sudo systemctl restart rknn3.service rknn-mdns.service
```

> 如果设备未上线，可能是固件版型不匹配：SDK 里的 `install_all.sh` 会自动
> 依次尝试 SODIMM / M.2 2280 / M.2 2242 固件。

### 3. 验证设备

```bash
sudo rknn-smi info
# 能看到 "RK1828" 字样和 Online 状态即正常
```

注意：固件未启动时 rknn-smi 也可能显示 Online，**必须以芯片名 RK182x 为准**。

### 4. 安装 Python 依赖

建议用虚拟环境（Python 3.10+）：

```bash
cd rk1828-notebook
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

依赖：`numpy / matplotlib / pillow / opencv-python / soundfile / ipywidgets / jupyter`。

### 5. 启动 Jupyter

```bash
jupyter lab   # 或 jupyter notebook
```

打开任意 notebook，**从上到下顺序运行**。

## 模型下载

每个 notebook 的**第一步都是下载模型**：下拉框选择下载源，已存在的文件自动跳过：

| 下载源 | 地址 | 适用 |
|---|---|---|
| **HF-Mirror 国内镜像**（默认推荐） | `hf-mirror.com` | 国内直连，速度快 |
| **Hugging Face 官方** | `huggingface.co` | 海外或已有代理 |

模型全部托管在 [huggingface.co/qhddz](https://huggingface.co/qhddz)。
也可以命令行一次性下载全部模型（约 25GB）：

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

## 常见问题

- **`librknn3_api.so` 加载失败**：第 2 步的 Runtime 未安装，或安装后未重启服务
- **找不到 RK1828 设备**：`sudo rknn-smi info` 检查；无 `/dev/pcie-rkep-*` 节点说明
  DKMS 驱动没装上（`dkms status | grep pcie-rkep`）
- **core mask 报错**：RK1828 只有 1 个 NPU 核，脚本里 core mask 用 `0x1`
- **磁盘不足**：不必全量下载，每个 notebook 的第一步只下载自己需要的模型
- **下载中断/失败**：重跑下载单元格即可，已完成的文件会自动跳过

## 模型来源与许可

模型为官方 `rknn3-model-zoo` 流程转换的 RKNN3 产物，已在真实 RK1828 上验证。
各模型的许可以其上游仓库为准（YOLO 系为 AGPL-3.0，Qwen 系为 Apache-2.0，
SenseVoice/VITS 为 MIT，RAFT-Stereo 为 BSD-3-Clause 等），
转换成 RKNN3 格式不改变其许可。详见各 HF 仓库的 README。
