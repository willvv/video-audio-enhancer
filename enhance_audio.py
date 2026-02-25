"""
Audio enhancement script for video files.
Uses: ffmpeg (audio extraction/muxing), noisereduce + scipy (processing).

Pipeline per video:
  1. Extract audio as 48kHz WAV
  2. Noise reduction (stationary + non-stationary passes)
  3. High-pass filter at 80Hz (remove low-frequency rumble/hum)
  4. Gentle de-essing / high shelving
  5. RMS normalization to -14 LUFS equivalent
  6. True-peak limiter
  7. Mux processed audio back into a copy of the video
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy import signal

VIDEO_DIR = Path(r"C:\code\video")
OUTPUT_SUFFIX = "_enhanced"
SAMPLE_RATE = 48000


def run(cmd, **kwargs):
    print(f"  > {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr.strip()}")
        raise RuntimeError(f"Command failed: {cmd[0]}")
    return result


def extract_audio(video_path: Path, wav_path: Path):
    """Extract audio from video to 48kHz float32 WAV."""
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_f32le",
        str(wav_path)
    ])


def high_pass_filter(audio: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """Remove rumble/hum below cutoff_hz."""
    sos = signal.butter(4, cutoff_hz / (sr / 2), btype="high", output="sos")
    if audio.ndim == 1:
        return signal.sosfilt(sos, audio)
    return np.stack([signal.sosfilt(sos, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)


def high_shelf_filter(audio: np.ndarray, sr: int, freq_hz: float = 8000.0,
                      gain_db: float = -2.5) -> np.ndarray:
    """Gentle high-frequency shelf to tame harshness / de-essing."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq_hz / sr
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2 * 0.7)  # Q = 0.7

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])

    if audio.ndim == 1:
        return signal.lfilter(b, a, audio)
    return np.stack([signal.lfilter(b, a, audio[:, ch]) for ch in range(audio.shape[1])], axis=1)


def rms_normalize(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    """Normalize to a target RMS level (~-22 dBFS, comfortable for voice)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    gain = target_rms / rms
    # Don't boost more than +20 dB to avoid amplifying silence
    gain = min(gain, 10.0)
    return audio * gain


def true_peak_limit(audio: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Hard limiter to prevent clipping."""
    peak = np.max(np.abs(audio))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return audio


def apply_noise_reduction(audio: np.ndarray, sr: int) -> np.ndarray:
    """Two-pass noise reduction: stationary then non-stationary."""
    is_stereo = audio.ndim == 2
    if is_stereo:
        channels = [audio[:, ch] for ch in range(audio.shape[1])]
    else:
        channels = [audio]

    processed = []
    for ch_data in channels:
        # Pass 1: stationary noise (constant hum, AC, fans)
        # Use first 0.5s as noise profile sample
        noise_sample_len = min(int(sr * 0.5), len(ch_data) // 4)
        p1 = nr.reduce_noise(
            y=ch_data,
            sr=sr,
            y_noise=ch_data[:noise_sample_len],
            stationary=True,
            prop_decrease=0.85,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_std_thresh_stationary=1.5,
        )

        # Pass 2: non-stationary noise (variable background)
        p2 = nr.reduce_noise(
            y=p1,
            sr=sr,
            stationary=False,
            prop_decrease=0.75,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            time_constant_s=2.0,
        )
        processed.append(p2)

    if is_stereo:
        return np.stack(processed, axis=1)
    return processed[0]


def process_video(video_path: Path) -> Path:
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*60}")

    output_path = video_path.with_stem(video_path.stem + OUTPUT_SUFFIX)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        raw_wav = tmp / "raw_audio.wav"
        processed_wav = tmp / "processed_audio.wav"

        # Step 1: Extract audio
        print("\n[1/5] Extracting audio...")
        extract_audio(video_path, raw_wav)

        # Step 2: Load audio
        print("[2/5] Loading and processing audio...")
        audio, sr = sf.read(str(raw_wav), dtype="float32")
        print(f"      Duration: {len(audio)/sr:.1f}s, Channels: {audio.ndim if audio.ndim == 1 else audio.shape[1]}, SR: {sr}Hz")

        # Step 3: Process
        print("[3/5] Applying noise reduction (this may take a moment)...")
        audio = apply_noise_reduction(audio, sr)

        print("[4/5] Applying EQ, normalization, limiting...")
        audio = high_pass_filter(audio, sr, cutoff_hz=80.0)
        audio = high_shelf_filter(audio, sr, freq_hz=8000.0, gain_db=-2.0)
        audio = rms_normalize(audio, target_rms=0.10)
        audio = true_peak_limit(audio, ceiling=0.95)

        # Step 4: Write processed audio
        sf.write(str(processed_wav), audio, sr, subtype="PCM_24")

        # Step 5: Mux back into video
        print(f"[5/5] Writing output video: {output_path.name}")
        run([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(processed_wav),
            "-c:v", "copy",           # copy video stream unchanged
            "-map", "0:v:0",          # video from original
            "-map", "1:a:0",          # audio from processed
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", str(SAMPLE_RATE),
            str(output_path)
        ])

    print(f"\nDone: {output_path}")
    return output_path


def main():
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    # Exclude already-enhanced files
    videos = [v for v in videos if OUTPUT_SUFFIX not in v.stem]

    if not videos:
        print("No .mp4 files found.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to process:")
    for v in videos:
        print(f"  - {v.name}")

    for video in videos:
        process_video(video)

    print("\nAll done! Enhanced videos saved alongside originals.")


if __name__ == "__main__":
    main()
