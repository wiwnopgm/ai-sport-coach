from typing import Optional

import cv2
import numpy as np
from rich import print as rprint
from ultralytics import YOLO

from .video_io import get_video_meta, iterate_video_frames, write_video


class SimpleTracker:
    def __init__(self):
        self.prev_box: Optional[np.ndarray] = None

    def update(self, boxes_xyxy: np.ndarray) -> int:
        if boxes_xyxy.size == 0:
            return -1
        if self.prev_box is None:
            areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            idx = int(np.argmax(areas))
            self.prev_box = boxes_xyxy[idx]
            return idx
        ious = self._iou(self.prev_box, boxes_xyxy)
        idx = int(np.argmax(ious))
        self.prev_box = boxes_xyxy[idx]
        return idx

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter + 1e-6
        return inter / union


def blur_background(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    mask3 = np.repeat(mask[..., None], 3, axis=2)
    return np.where(mask3 > 0, frame, blurred)


def segment_main_player(
    video_path: str,
    start_s: float,
    end_s: float,
    output_video_path: str,
    device: str = "auto",
    seg_model_name: str = "yolov8n-seg.pt",
) -> str:
    width, height, fps, total = get_video_meta(video_path)

    model = YOLO(seg_model_name)
    if device != "auto":
        model.to(device)

    writer = write_video(output_video_path, width, height, fps)

    tracker = SimpleTracker()

    for frame_idx, t, frame in iterate_video_frames(video_path, start_s, end_s, stride=1):
        results = model.predict(source=frame, verbose=False)
        if len(results) == 0:
            writer.write(frame)
            continue

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4), dtype=np.float32)
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None

        if boxes.shape[0] == 0 or masks is None:
            writer.write(frame)
            continue

        idx = tracker.update(boxes)
        if idx < 0:
            writer.write(frame)
            continue

        mask = masks[idx].astype(np.uint8)
        mask = (mask > 0).astype(np.uint8)
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        composed = blur_background(frame, mask_resized)
        writer.write(composed)

    writer.release()
    rprint(f"[bold green]Segmented video saved:[/bold green] {output_video_path}")
    return output_video_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--out", type=str, default="outputs/segmented.mp4")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    args = parser.parse_args()

    segment_main_player(args.video, args.start, args.end, args.out, args.device, args.model)
