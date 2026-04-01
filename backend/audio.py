"""
Audio extraction module using yt-dlp.
Downloads audio from video URLs and handles cleanup of temporary files.
"""

import os
import uuid
import tempfile
import yt_dlp


def download_audio(video_url: str, output_dir: str | None = None) -> str:
    """
    Downloads audio-only from a video URL using yt-dlp.

    Args:
        video_url: URL of the video to extract audio from.
        output_dir: Directory to save the audio file. Defaults to system temp dir.

    Returns:
        Absolute path to the downloaded .wav audio file.

    Raises:
        ValueError: If the URL is empty or invalid.
        RuntimeError: If yt-dlp fails to download the audio.
    """
    if not video_url or not video_url.strip():
        raise ValueError("Video URL cannot be empty.")

    if output_dir is None:
        output_dir = tempfile.gettempdir()

    os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename to avoid collisions
    unique_id = uuid.uuid4().hex[:10]
    output_path = os.path.join(output_dir, f"asklecture_{unique_id}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        raise RuntimeError(f"Failed to download audio from URL: {e}")

    wav_path = f"{output_path}.wav"

    if not os.path.exists(wav_path):
        raise RuntimeError(
            "Audio download completed but .wav file was not found. "
            "Ensure ffmpeg is installed on your system."
        )

    return wav_path
