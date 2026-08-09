import os
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4


TARGET_LUFS = -18.0
TRUE_PEAK_DBTP = -2.0
LOUDNESS_RANGE_LU = 7.0


def loudness_filter() -> str:
    """Return the EBU R128 normalization used for audiobook speech chunks."""

    return (
        f"loudnorm=I={TARGET_LUFS:g}:TP={TRUE_PEAK_DBTP:g}:"
        f"LRA={LOUDNESS_RANGE_LU:g}"
    )


def normalize_speech_wav(
    path: str | Path,
    sample_rate: int,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Normalize a speech WAV in place without risking the original on failure."""

    source = Path(path)
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to normalize audiobook audio")

    temporary = source.with_name(
        f".{source.stem}.normalized-{uuid4().hex}{source.suffix}"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        loudness_filter(),
        "-ar",
        str(int(sample_rate)),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr.strip() or "unknown FFmpeg error"
            raise RuntimeError(f"FFmpeg failed to normalize speech audio: {detail}")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("FFmpeg did not produce normalized speech audio")
        os.replace(temporary, source)
    finally:
        temporary.unlink(missing_ok=True)
    return source
