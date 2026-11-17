"""Video/audio -> mono 16 kHz 16-bit PCM WAV (YAMNet's required format)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .utils import get_logger

log = get_logger(__name__)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
TARGET_SR = 16000


class FFmpegNotFoundError(RuntimeError):
    pass


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise FFmpegNotFoundError(
            "ffmpeg was not found on PATH. Install it first:\n"
            "  Windows: winget install ffmpeg   (or https://ffmpeg.org/download.html)\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg"
        )
    return exe


def _run_ffmpeg(input_path: str, output_wav: str) -> None:
    exe = _require_ffmpeg()
    cmd = [
        exe, "-i", input_path,
        "-vn",                 # ignore any video stream
        "-ac", "1",            # mono
        "-ar", str(TARGET_SR), # 16 kHz
        "-sample_fmt", "s16",  # PCM 16-bit
        output_wav, "-y",
    ]
    log.debug("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        raise RuntimeError(
            f"ffmpeg failed on '{input_path}' (exit {proc.returncode}). "
            f"Does the file contain an audio track?\n{tail}"
        )


def _normalize_wav_without_ffmpeg(input_path: str, output_wav: str) -> None:
    """Fallback for .wav input when ffmpeg is unavailable."""
    import librosa
    import soundfile as sf

    y, _ = librosa.load(input_path, sr=TARGET_SR, mono=True)
    sf.write(output_wav, y, TARGET_SR, subtype="PCM_16")


def extract_audio(input_path: str, output_wav: str) -> str:
    """Convert `input_path` to mono / 16 kHz / PCM-16 WAV at `output_wav`.

    - .wav input is re-encoded to the target format (ffmpeg if present, else librosa).
    - Any other container (.mp4 .mov .mkv .avi, or audio like .mp3/.m4a) goes through ffmpeg.
    Returns the path to the WAV file.
    """
    src = Path(input_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".wav":
        if shutil.which("ffmpeg"):
            _run_ffmpeg(str(src), output_wav)
        else:
            log.warning("ffmpeg not found; normalising WAV with librosa instead.")
            _normalize_wav_without_ffmpeg(str(src), output_wav)
    else:
        if src.suffix.lower() not in VIDEO_EXTS:
            log.info("Unrecognised extension '%s'; trying ffmpeg anyway.", src.suffix)
        _run_ffmpeg(str(src), output_wav)

    log.info("Audio ready: %s", output_wav)
    return output_wav
