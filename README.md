# rk1828-notebook

English | [中文](README_CN.md)

16 interactive Jupyter notebook demos running on the **Rockchip RK1828 AI
coprocessor** (PCIe, 20 TOPS NPU), covering vision, speech and multimodal
models. All models are in **RKNN3 format** (weight-separated: `.rknn` graph +
`.weight` weights) and have been verified end-to-end on real RK1828 hardware.

## Environment Setup

### 1. Hardware requirements

- x86_64 Linux host (Ubuntu 20.04+ recommended)
- An RK1820 / RK1828 AI coprocessor card (PCIe: SODIMM, M.2 2280 or M.2 2242)
- Disk space: ~**25 GB** for all models (or download only what one notebook needs)

### 2. Install the RKNN3 driver / firmware / runtime

The RK1828 talks to the host over PCIe, so the DKMS driver, firmware and the
`librknn3_api` runtime must be installed first. They ship in the official
Rockchip SDK (`rknn3-rk182x`):

```bash
# Option A (recommended): the SDK one-click script — installs dependencies,
# the DEB package, and auto-detects the board variant firmware
sudo /path/to/rk1828-x86-v1.2.0/install_all.sh

# Option B: manual install of the x86_64 DEB (driver/firmware/runtime/SMI)
sudo dpkg -i rknn3-rk182x-sodimm_v1.2.0_x86_64.deb || sudo apt-get -f install -y
sudo modprobe pcie-rkep
sudo systemctl restart rknn3.service rknn-mdns.service
```

> If the device does not come online, the firmware variant may not match your
> board: `install_all.sh` automatically tries SODIMM / M.2 2280 / M.2 2242
> firmware images in turn.

### 3. Verify the device

```bash
sudo rknn-smi info
# You should see "RK1828" with Online status
```

Note: rknn-smi can show "Online" even before the firmware is up — always
check for the **RK182x chip name**.

### 4. Install Python dependencies

A virtual environment is recommended (Python 3.10+):

```bash
cd rk1828-notebook
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `numpy / matplotlib / pillow / opencv-python / soundfile /
ipywidgets / jupyter`.

### 5. Launch Jupyter

```bash
jupyter lab   # or: jupyter notebook
```

Open any notebook and **run the cells top to bottom**.

## Model download

The **first step of every notebook downloads the models**: pick a source from
the dropdown; files that already exist are skipped automatically:

| Source | Host | Best for |
|---|---|---|
| **HF-Mirror** (default) | `hf-mirror.com` | Direct access in China, faster |
| **Hugging Face** | `huggingface.co` | Overseas or behind a proxy |

All models are hosted at [huggingface.co/qhddz](https://huggingface.co/qhddz).
To download everything from the command line (~25 GB):

```bash
bash download_models.sh            # default: HF-Mirror
bash download_models.sh hf         # Hugging Face official
```

## Notebook list

| # | Notebook | Model | Task |
|---|---|---|---|
| 01 | 01_classification | MobileNetV2 / ResNet50-v2 (i8/fp16) | Image classification (Python inference) |
| 02 | 02_yolo_detection | YOLOv5/v6/v8, YOLO26 n/s/m | Object detection (Python inference) |
| 03 | 03_FastSAM | FastSAM-s | Instance segmentation (Python inference) |
| 04 | 04_yolo26_pose_segment | YOLO26-pose / YOLO26-seg | Pose estimation + segmentation |
| 05 | 05_RAFT_Stereo | RAFT-Stereo realtime | Stereo disparity |
| 06 | 06_DepthAnythingV3 | Depth Anything V3 (base) | Multi-view depth estimation |
| 07 | 07_QAClip | Chinese-CLIP | Chinese image-text matching |
| 08 | 08_Qwen3_VL | Qwen3-VL-2B / 4B | Vision-language model |
| 09 | 09_Qwen3_ASR | Qwen3-ASR-0.6B | Speech recognition (52 languages) |
| 10 | 10_whisper | whisper-base | Speech recognition (zh/en) |
| 11 | 11_paraformer | Paraformer | Chinese ASR (hotwords) |
| 12 | 12_sensevoice | SenseVoice-Small | Multilingual ASR (emotion/event tags) |
| 13 | 13_forced_aligner | Qwen3-ForcedAligner-0.6B | Forced alignment (word timestamps) |
| 14 | 14_vits_tts | VITS (LJSpeech / VCTK) | English text-to-speech |
| 15 | 15_Qwen3_Embedding | Qwen3-Embedding-0.6B | Text embeddings (RAG retrieval) |
| 16 | 16_Qwen3_Reranker | Qwen3-Reranker-0.6B | Document reranking (RAG) |

Notebooks 01-03 call the runtime directly from Python (`rknn3.py`, a ctypes
wrapper) and walk through pre-processing / inference / post-processing step by
step; 04-16 drive prebuilt x86_64 C++ demos from the official
`rknn3-model-zoo` pipelines.

## Repository layout

```
├── 01_classification.ipynb ... 16_Qwen3_Reranker.ipynb
├── rknn3.py            # ctypes wrapper of librknn3_api.so (Rknn3Model)
├── nb_download.py      # model download (HF / HF-Mirror picker, skip existing)
├── nb_demo.py          # C++ demo runner / result display helpers
├── vision_post.py      # YOLO/FastSAM pre/post-processing (letterbox/NMS/decode/masks)
├── download_models.sh  # batch download of all models (CLI)
├── _build_notebooks.py # notebook generator (regenerates all .ipynb files)
├── models/             # model files (created by downloads)
├── datasets/           # sample images/audio (created by downloads)
└── demos/              # C++ demo binaries + models (created by downloads)
```

## Troubleshooting

- **`librknn3_api.so` fails to load**: the runtime from step 2 is not
  installed, or the services were not restarted after installation
- **No RK1828 device found**: check `sudo rknn-smi info`; missing
  `/dev/pcie-rkep-*` nodes mean the DKMS driver is not loaded
  (`dkms status | grep pcie-rkep`)
- **Core mask errors**: RK1828 has a single NPU core — use `0x1`
- **Low disk space**: no need to download everything — each notebook's step 1
  only fetches the models it needs
- **Interrupted downloads**: just re-run the download cell; completed files
  are skipped automatically

## Model sources and licenses

All models are RKNN3 conversions produced with the official
`rknn3-model-zoo` pipelines and verified on real RK1828 hardware. Each model
keeps the license of its upstream repository (YOLO family: AGPL-3.0, Qwen
family: Apache-2.0, SenseVoice/VITS: MIT, RAFT-Stereo: BSD-3-Clause, etc.).
Converting weights to RKNN3 format does not change their license. See each
Hugging Face repository's README for details.
