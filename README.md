## AI Sport Coach - Badminton Pose Pipeline

This repo contains a modular pipeline to:

- Download a badminton video from a YouTube URL
- Auto-select an interval that best captures the player's form
- Filter out background via person segmentation to focus on the player
- Extract the player's pose (skeleton keypoints)
- Visualize skeleton movement as a video overlay

### 1) Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- macOS Apple Silicon can leverage Metal (MPS) with `torch>=2.1`. If you already have a working PyTorch install, you can comment out the `torch` lines in `requirements.txt` and keep your local build.
- Ensure `ffmpeg` is installed and available on your PATH (e.g., `brew install ffmpeg`).

### 2) Run the end-to-end pipeline

```bash
python pipeline.py "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \
  --out outputs \
  --seconds 6 \
  --stride 2 \
  --device auto \
  --pose yolov8n-pose.pt \
  --seg yolov8n-seg.pt
```

Outputs are written to `outputs/<video_id>/`:

- `segmented.mp4`: background-filtered clip focusing on the player
- `poses.csv`: extracted keypoints across frames
- `skeleton_overlay.mp4`: video with skeleton drawn over frames
- `run_meta.json`: metadata for the run

### 3) Implementation notes

- Interval selection: uses Ultralytics pose model scores (keypoint visibility/confidence and subject size) to choose the best N-second window.
- Segmentation: uses Ultralytics instance segmentation to mask the main player and blur the background.
- Pose extraction: outputs per-frame COCO-17 keypoints (x, y, conf), with optional smoothing.
- Visualization: renders a skeleton overlay using the extracted keypoints.

### 4) Troubleshooting

- If models fail to download the first time, try re-running. You can also pre-download models using Ultralytics CLI.
- For CUDA GPUs, ensure an appropriate PyTorch version is installed for your driver.
- For macOS without `ffmpeg`, install via Homebrew: `brew install ffmpeg`.
