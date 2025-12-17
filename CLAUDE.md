# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Depth Anything is a video depth estimation model from ByteDance (CVPR 2025 highlight). It performs temporally consistent depth estimation for arbitrarily long videos. This is a cloned/vendored dependency used for dental tool perception tasks.

## Commands

```bash
# Offline depth inference using parent project venv (run from this directory)
uv run --project .. python run.py --input_video ../data/pen_and_face.mp4 --output_dir ./outputs --encoder vits

# Offline depth inference (relative depth)
python3 run.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ./outputs --encoder vitl

# Offline depth inference (metric depth)
python3 run.py --input_video ./video.mp4 --output_dir ./outputs --encoder vitl --metric

# Streaming depth inference (experimental)
python3 run_streaming.py --input_video ./video.mp4 --output_dir ./outputs --encoder vitl

# Launch Gradio web interface
python3 app.py

# Download model weights
bash get_weights.sh
```

### Key Arguments for run.py / run_streaming.py
- `--encoder`: Model size (vits, vitb, vitl)
- `--metric`: Use metric depth models instead of relative
- `--input_size`: Input resolution (default 518, must be multiple of 14)
- `--max_res`: Max resolution cap (default 1280)
- `--max_len`: Max frames to process (-1 = all)
- `--target_fps`: Target FPS (-1 = original)
- `--fp32`: Use float32 instead of fp16
- `--save_npz`: Save raw depth as compressed NPZ
- `--save_exr`: Save depth as EXR frames
- `--grayscale`: Output grayscale depth (no colormap)

## Architecture

### Core Model Pipeline
1. **DINOv2 Backbone** (`video_depth_anything/dinov2.py`): Pre-trained ViT feature extractor (vits/vitb/vitl variants)
2. **DPT Decoder** (`video_depth_anything/dpt.py`): Feature pyramid fusion for depth prediction
3. **Temporal Module** (`video_depth_anything/motion_module/`): Frame-to-frame consistency via temporal attention across 32-frame windows

### Inference Classes
- `VideoDepthAnything` (`video_depth_anything/video_depth.py`): Offline batch processing with 32-frame chunks and 10-frame overlap
- `VideoDepthAnythingStream` (`video_depth_anything/video_depth_stream.py`): Single-frame streaming with cached temporal states

### Key Constants (in video_depth.py, not recommended to change)
- `INFER_LEN = 32`: Frames per processing chunk
- `OVERLAP = 10`: Frame overlap between chunks
- `KEYFRAMES`: Specific frames used for scale/shift alignment
- `INTERP_LEN = 8`: Interpolated frames at chunk boundaries

### Utilities
- `utils/dc_utils.py`: Video I/O (read_video, save_video)
- `utils/util.py`: Scale/shift computation, interpolation helpers
- `loss/loss.py`: Training losses (VideoDepthLoss, TrimmedProcrustesLoss, TemporalGradientMatchingLoss)

## Model Configuration

```python
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
```

Checkpoints expected at:
- `checkpoints/video_depth_anything_{vits|vitb|vitl}.pth` (relative depth)
- `checkpoints/metric_video_depth_anything_{vits|vitb|vitl}.pth` (metric depth)

## Benchmarking

```bash
# Extract dataset for evaluation
cd benchmark/dataset_extract
python3 dataset_extract_scannet.py

# Run inference on dataset
python3 benchmark/infer/infer.py --infer_path ${out_path} --json_file ${json_path} --datasets scannet

# Evaluate results
bash benchmark/eval/eval.sh ${out_path} benchmark/dataset_extract/dataset
```

Supported datasets: Sintel, KITTI, Bonn, ScanNet, NYUv2

## Notes

- FP16 precision used by default (saves ~50% VRAM)
- Input images normalized with ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Videos with aspect ratio >16:9 are automatically downsampled
- Point cloud generation requires camera intrinsics via `--focal-length-x/y` (default 470.4)
- Streaming mode has ~8-10% accuracy drop vs offline due to training/testing mismatch
