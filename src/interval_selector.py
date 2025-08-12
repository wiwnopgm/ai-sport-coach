from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
from scipy.signal import savgol_filter
from rich import print as rprint

from ultralytics import YOLO

from .video_io import get_video_meta, iterate_video_frames


@dataclass
class IntervalSelection:
    start_s: float
    end_s: float
    scores: List[Tuple[float, float]]


def _frame_pose_quality(result) -> Optional[Tuple[float, float, float]]:
    if result.keypoints is None or result.boxes is None:
        return None

    boxes_obj = result.boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return None

    boxes_xyxy = boxes_obj.xyxy.cpu().numpy()
    boxes_conf = boxes_obj.conf.squeeze().cpu().numpy()
    best_idx = int(np.argmax(boxes_conf))

    # Ultralytics v8: Keypoints is an object; access underlying tensor via .data
    # Shape: (num_dets, num_keypoints, 3 [x,y,conf])
    kps = result.keypoints.data[best_idx].cpu().numpy()

    vis_mask = kps[:, 2] > 0.25
    if vis_mask.sum() == 0:
        return None

    avg_conf = float(kps[vis_mask, 2].mean())
    visible_fraction = float(vis_mask.mean())

    x1, y1, x2, y2 = boxes_xyxy[best_idx].reshape(-1)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    area = float(w * h)

    H, W = result.orig_shape
    norm_area = area / float(W * H)
    return avg_conf, visible_fraction, norm_area


def select_best_interval(
    video_path: str,
    interval_seconds: float = 6.0,
    stride: int = 2,
    device: str = "auto",
    pose_model_name: str = "yolov8n-pose.pt",
) -> IntervalSelection:
    width, height, fps, total = get_video_meta(video_path)

    model = YOLO(pose_model_name)
    if device != "auto":
        model.to(device)

    timestamps: List[float] = []
    scores: List[float] = []

    for frame_idx, t, frame in iterate_video_frames(video_path, 0.0, None, stride=stride):
        results = model.predict(source=frame, verbose=False)
        if len(results) == 0:
            continue
        q = _frame_pose_quality(results[0])
        if q is None:
            timestamps.append(t)
            scores.append(0.0)
            continue
        avg_conf, visible_frac, norm_area = q
        score = 0.6 * visible_frac + 0.3 * avg_conf + 0.1 * min(1.0, norm_area * 4.0)
        timestamps.append(t)
        scores.append(score)

    if not scores:
        return IntervalSelection(start_s=0.0, end_s=min(interval_seconds, total / fps), scores=[])

    ts = np.array(timestamps)
    sc = np.array(scores)

    if len(sc) >= 7:
        win = min(9, len(sc) - (1 - len(sc) % 2))
        if win >= 5 and win % 2 == 1:
            sc = savgol_filter(sc, window_length=win, polyorder=2)

    best_start, best_end = 0.0, interval_seconds
    best_mean = -1.0

    for s in ts:
        e = s + interval_seconds
        mask = (ts >= s) & (ts <= e)
        if mask.sum() < 2:
            continue
        mean_score = float(sc[mask].mean())
        if mean_score > best_mean:
            best_mean = mean_score
            best_start, best_end = float(s), float(min(e, ts[-1]))

    rprint(f"[bold green]Selected interval:[/bold green] {best_start:.2f}s → {best_end:.2f}s (score={best_mean:.3f})")
    return IntervalSelection(start_s=best_start, end_s=best_end, scores=list(zip(ts.tolist(), sc.tolist())))
