import os
from typing import List, Dict

import pandas as pd
from rich import print as rprint
from scipy.signal import savgol_filter
from ultralytics import YOLO

from .video_io import iterate_video_frames


COCO17_SKELETON = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def extract_pose_to_csv(
    video_path: str,
    start_s: float,
    end_s: float,
    output_csv_path: str,
    device: str = "auto",
    pose_model_name: str = "yolov8n-pose.pt",
    smooth: bool = True,
) -> str:
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    model = YOLO(pose_model_name)
    if device != "auto":
        model.to(device)

    rows: List[Dict] = []

    for frame_idx, t, frame in iterate_video_frames(video_path, start_s, end_s, stride=1):
        results = model.predict(source=frame, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            continue
        r = results[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            continue
        # Access underlying keypoints tensor via .data
        kp = r.keypoints.data[0].cpu().numpy()
        for k_idx, (x, y, c) in enumerate(kp):
            rows.append({
                "frame": frame_idx,
                "time_s": t,
                "keypoint": int(k_idx),
                "x": float(x),
                "y": float(y),
                "conf": float(c),
            })

    if not rows:
        raise RuntimeError("No poses found in selected interval.")

    df = pd.DataFrame(rows)
    df.sort_values(["frame", "keypoint"], inplace=True)

    if smooth:
        smoothed = []
        for k in sorted(df.keypoint.unique()):
            sub = df[df.keypoint == k].copy()
            if len(sub) >= 7:
                win = min(21, len(sub))
                if win % 2 == 0:
                    win -= 1
                sub["x"] = savgol_filter(sub["x"].values, window_length=win, polyorder=2)
                sub["y"] = savgol_filter(sub["y"].values, window_length=win, polyorder=2)
            smoothed.append(sub)
        df = pd.concat(smoothed, ignore_index=True)

    df.to_csv(output_csv_path, index=False)
    rprint(f"[bold green]Pose CSV saved:[/bold green] {output_csv_path}  (rows={len(df)})")
    return output_csv_path
