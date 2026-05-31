# VLM_engagement_pillar

VLM wrapper to detect engagement using the webcam, for the PILLAR project.

## Requirements

- Python 3.
- OpenCV.
  - opencv-python==4.13.0.92
- (Recommended) CUDA-capable GPU.
- torch==2.10.0
- torchcodec==0.10.0
- torchvision==0.25.0
- transformers==5.4.0
- qwen-vl-utils==0.0.14
- pandas==3.0.1

## Installation

```bash
# git clone this repo
git clone <REPO_PATH>.git
cd VLM_engagement_pillar/
python3 -m pip install -r requirements.txt
```

## Usage

```bash
cd VLM_engagement_pillar/VLM_engagement_pillar_core
# To run on CUDA-capable GPUs
python3 ./vlm_webcam.py --display --device cuda --interval 10
# To run on CPU
python3 ./vlm_webcam.py --display --device cpu --interval 10

## (Optional) to select a different camera index:
python3 ./vlm_webcam.py --display --device cuda --interval 10 --camera_index 4
```
