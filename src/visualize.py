import os

import cv2
import numpy as np
import pandas as pd
from rich import print as rprint

from .video_io import get_video_meta, iterate_video_frames, write_video
from .pose_extraction import COCO17_SKELETON


COLORS = {
    "skeleton": (0, 255, 0),
    "joint": (0, 128, 255),
}


def _draw_skeleton(frame: np.ndarray, keypoints_xyc: np.ndarray) -> np.ndarray:
    img = frame.copy()
    for (x, y, c) in keypoints_xyc:
        if c > 0.2:
            cv2.circle(img, (int(x), int(y)), 3, COLORS["joint"], -1)
    for i, j in COCO17_SKELETON:
        x1, y1, c1 = keypoints_xyc[i]
        x2, y2, c2 = keypoints_xyc[j]
        if c1 > 0.2 and c2 > 0.2:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), COLORS["skeleton"], 2)
    return img


def render_skeleton_overlay_video(
    video_path: str,
    start_s: float,
    end_s: float,
    poses_csv_path: str,
    output_video_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    width, height, fps, total = get_video_meta(video_path)
    writer = write_video(output_video_path, width, height, fps)

    df = pd.read_csv(poses_csv_path)
    grouped = df.groupby("frame")
    frame_to_kps = {int(k): g.sort_values("keypoint")[['x', 'y', 'conf']].values for k, g in grouped}

    for frame_idx, t, frame in iterate_video_frames(video_path, start_s, end_s, stride=1):
        kps = frame_to_kps.get(frame_idx)
        if kps is not None:
            frame = _draw_skeleton(frame, kps)
        writer.write(frame)

    writer.release()
    rprint(f"[bold green]Skeleton overlay saved:[/bold green] {output_video_path}")
    return output_video_path
