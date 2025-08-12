import os
from dataclasses import dataclass
from typing import Optional, Tuple

from rich import print as rprint
from yt_dlp import YoutubeDL


@dataclass
class DownloadResult:
    video_id: str
    output_path: str
    title: Optional[str]
    duration_sec: Optional[float]


def download_youtube_video(
    youtube_url: str,
    output_dir: str,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> DownloadResult:
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "extractor_retries": 10,
        "noprogress": True,
        "http_headers": {
            "User-Agent": user_agent
            or "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
        # Prefer TV first to avoid iOS 400 errors and recent web client restrictions
        "extractor_args": {"youtube": {"player_client": ["tv", "web", "web_creator"]}},
    }

    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id")
        title = info.get("title")
        duration = info.get("duration")
        out_path = os.path.join(output_dir, f"{video_id}.mp4")

    rprint(f"[bold green]Downloaded:[/bold green] {title} → {out_path}")
    return DownloadResult(video_id=video_id, output_path=out_path, title=title, duration_sec=duration)


def ensure_ffmpeg_available() -> None:
    import shutil

    if shutil.which("ffmpeg") is None:
        rprint("[yellow]Warning: ffmpeg not found in PATH. Video trimming will be slower or unavailable.[/yellow]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str)
    parser.add_argument("--out", type=str, default="downloads")
    parser.add_argument("--cookies-browser", type=str, default=None, help="Browser to import cookies from (safari|chrome|edge|firefox)")
    parser.add_argument("--cookies-file", type=str, default=None, help="Path to cookies.txt (Netscape format)")
    parser.add_argument("--user-agent", type=str, default=None, help="Override HTTP User-Agent")
    args = parser.parse_args()

    ensure_ffmpeg_available()
    res = download_youtube_video(
        args.url,
        args.out,
        cookies_from_browser=args.cookies_browser,
        cookies_file=args.cookies_file,
        user_agent=args.user_agent,
    )
    rprint(res)
