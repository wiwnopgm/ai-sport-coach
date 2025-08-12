import os
import json

from rich import print as rprint

from src.downloader import download_youtube_video, ensure_ffmpeg_available, DownloadResult
from src.interval_selector import select_best_interval
from src.segmentation import segment_main_player
from src.pose_extraction import extract_pose_to_csv
from src.visualize import render_skeleton_overlay_video


DEFAULTS = {
    "interval_seconds": 6.0,
    "frame_stride": 2,
    "device": "auto",
    "pose_model": "yolov8n-pose.pt",
    "seg_model": "yolov8n-seg.pt",
}


def run_pipeline(video_or_url: str, out_root: str = "outputs", **kwargs):
    ensure_ffmpeg_available()

    # Accept either a YouTube URL or a local video file path
    if os.path.exists(video_or_url):
        video_path = video_or_url
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        dl = DownloadResult(video_id=video_id, output_path=video_path, title=os.path.basename(video_path), duration_sec=None)
    else:
        dl = download_youtube_video(
            video_or_url,
            os.path.join(out_root, "downloads"),
            cookies_from_browser=kwargs.get("cookies_browser"),
            cookies_file=kwargs.get("cookies_file"),
            user_agent=kwargs.get("user_agent"),
        )
        video_path = dl.output_path
        video_id = dl.video_id

    os.makedirs(out_root, exist_ok=True)
    run_dir = os.path.join(out_root, video_id)
    os.makedirs(run_dir, exist_ok=True)

    interval_seconds = float(kwargs.get("interval_seconds", DEFAULTS["interval_seconds"]))
    frame_stride = int(kwargs.get("frame_stride", DEFAULTS["frame_stride"]))
    device = kwargs.get("device", DEFAULTS["device"]) or "auto"
    pose_model = kwargs.get("pose_model", DEFAULTS["pose_model"]) or DEFAULTS["pose_model"]
    seg_model = kwargs.get("seg_model", DEFAULTS["seg_model"]) or DEFAULTS["seg_model"]

    sel = select_best_interval(video_path, interval_seconds, frame_stride, device, pose_model)

    seg_out = os.path.join(run_dir, "segmented.mp4")
    segment_main_player(video_path, sel.start_s, sel.end_s, seg_out, device, seg_model)

    poses_csv = os.path.join(run_dir, "poses.csv")
    extract_pose_to_csv(video_path, sel.start_s, sel.end_s, poses_csv, device, pose_model)

    overlay_out = os.path.join(run_dir, "skeleton_overlay.mp4")
    render_skeleton_overlay_video(video_path, sel.start_s, sel.end_s, poses_csv, overlay_out)

    meta_path = os.path.join(run_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "video_id": video_id,
            "video_title": dl.title,
            "video_path": video_path,
            "interval": {"start_s": sel.start_s, "end_s": sel.end_s},
            "outputs": {
                "segmented_video": seg_out,
                "poses_csv": poses_csv,
                "skeleton_overlay": overlay_out,
            }
        }, f, indent=2)

    rprint(f"[bold green]Pipeline completed.[/bold green] Outputs → {run_dir}")
    return run_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Sport Coach - Badminton Pose Pipeline")
    parser.add_argument("url", type=str, help="YouTube URL or local video path")
    parser.add_argument("--out", type=str, default="outputs", help="Output root directory")
    parser.add_argument("--seconds", type=float, default=DEFAULTS["interval_seconds"], help="Interval duration in seconds")
    parser.add_argument("--stride", type=int, default=DEFAULTS["frame_stride"], help="Frame sampling stride for interval selection")
    parser.add_argument("--device", type=str, default=DEFAULTS["device"], help="Device: auto|cpu|cuda|mPS")
    parser.add_argument("--pose", type=str, default=DEFAULTS["pose_model"], help="Ultralytics pose model name")
    parser.add_argument("--seg", type=str, default=DEFAULTS["seg_model"], help="Ultralytics segmentation model name")
    parser.add_argument("--cookies-browser", type=str, default=None, help="Browser to import cookies from (safari|chrome|edge|firefox)")
    parser.add_argument("--cookies-file", type=str, default=None, help="Path to cookies.txt (Netscape format)")
    parser.add_argument("--user-agent", type=str, default=None, help="Override HTTP User-Agent")

    args = parser.parse_args()

    run_pipeline(
        video_or_url=args.url,
        out_root=args.out,
        interval_seconds=args.seconds,
        frame_stride=args.stride,
        device=args.device,
        pose_model=args.pose,
        seg_model=args.seg,
        cookies_browser=args.cookies_browser,
        cookies_file=args.cookies_file,
        user_agent=args.user_agent,
    )
