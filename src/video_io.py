import os
from typing import Generator, Tuple, Optional

import cv2


def get_video_meta(video_path: str) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, total


def iterate_video_frames(
    video_path: str,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    stride: int = 1,
) -> Generator[Tuple[int, float, 'cv2.Mat'], None, None]:
    width, height, fps, total = get_video_meta(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps) if end_s is not None else total

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    frame_idx = max(0, start_frame)

    while True:
        if frame_idx >= end_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % stride == 0:
            t = frame_idx / fps
            yield frame_idx, t, frame
        frame_idx += 1

    cap.release()


def write_video(
    output_path: str,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
