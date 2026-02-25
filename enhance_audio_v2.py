"""
Audio enhancement v2 using SpeechBrain MetricGAN+.
MetricGAN+ is a neural network trained specifically for speech enhancement.
It removes noise while preserving speech clarity much better than spectral methods.

Outputs per video:
  - *_enhanced_v2.mp4  : video copy with enhanced audio
  - *_enhanced_v2.wav  : standalone audio file for easy comparison
"""

import subprocess
import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import huggingface_hub

# Patch 1: torchaudio 2.x removed list_audio_backends; speechbrain 1.0.3 still calls it
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# Patch 2: huggingface_hub 1.x removed use_auth_token; speechbrain 1.0.3 still passes it
_orig_hf_download = huggingface_hub.hf_hub_download
def _patched_hf_download(*args, **kwargs):
    kwargs.pop("use_auth_token", None)
    return _orig_hf_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_hf_download

from speechbrain.inference.enhancement import SpectralMaskEnhancement

VIDEO_DIR = Path(r"C:\code\video")
OUTPUT_SUFFIX = "_enhanced_v2"
MODEL_ID = "speechbrain/metricgan-plus-voicebank"
MODEL_SR = 16000   # MetricGAN+ expects 16kHz mono
OUTPUT_SR = 48000  # final output sample rate


def run(cmd):
    print(f"  > {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-2000:].strip()}")
        raise RuntimeError(f"Command failed: {cmd[0]}")
    return result


def extract_audio(video_path: Path, wav_path: Path, sr: int):
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", str(sr),
        "-ac", "2",            # keep stereo
        "-acodec", "pcm_f32le",
        str(wav_path)
    ])


def load_model():
    print("\n  Loading MetricGAN+ model...")
    # Use the local HuggingFace cache path directly to avoid Windows symlink privilege errors
    from huggingface_hub import snapshot_download
    try:
        local_path = snapshot_download(MODEL_ID, local_files_only=True)
    except Exception:
        print("  Model not cached locally — downloading now (~40MB)...")
        local_path = snapshot_download(MODEL_ID)
    print(f"  Model path: {local_path}")
    model = SpectralMaskEnhancement.from_hparams(
        source=local_path,
        savedir=local_path,   # same dir = no symlinks needed
        run_opts={"device": "cpu"},
    )
    return model


def enhance_channel(model, mono_audio: np.ndarray, sr: int) -> np.ndarray:
    """Enhance a single mono channel through MetricGAN+."""
    # Resample to 16kHz if needed
    if sr != MODEL_SR:
        audio_t = torch.from_numpy(mono_audio).float().unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, MODEL_SR)
    else:
        audio_t = torch.from_numpy(mono_audio).float().unsqueeze(0)

    # MetricGAN+ needs shape [batch, time]
    lengths = torch.tensor([1.0])
    with torch.no_grad():
        enhanced = model.enhance_batch(audio_t, lengths)

    result = enhanced.squeeze(0).numpy()

    # Resample back to original SR
    if sr != MODEL_SR:
        result_t = torch.from_numpy(result).float().unsqueeze(0)
        result_t = torchaudio.functional.resample(result_t, MODEL_SR, sr)
        result = result_t.squeeze(0).numpy()

    return result


def peaking_eq(audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    """Biquad peaking EQ filter (boost or cut at a center frequency)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0,     a1 / a0, a2 / a0])

    from scipy import signal
    if audio.ndim == 1:
        return signal.lfilter(b, a, audio).astype(np.float32)
    return np.stack(
        [signal.lfilter(b, a, audio[:, ch]).astype(np.float32) for ch in range(audio.shape[1])],
        axis=1,
    )


def sharpen_voice(audio: np.ndarray, sr: int) -> np.ndarray:
    """Three-band presence EQ for voice clarity:
      - Cut  ~300 Hz  (-2 dB, Q=1.0)  — reduce low-mid muddiness/boxiness
      - Boost ~3 kHz  (+4 dB, Q=1.2)  — presence/intelligibility
      - Boost ~10 kHz (+2.5 dB, Q=0.8) — air/sparkle/crispness
    """
    audio = peaking_eq(audio, sr, freq=300,   gain_db=-2.0, q=1.0)
    audio = peaking_eq(audio, sr, freq=3000,  gain_db=+4.0, q=1.2)
    audio = peaking_eq(audio, sr, freq=10000, gain_db=+2.5, q=0.8)
    return audio


def loudnorm_wav(src: Path, dst: Path):
    """Apply EBU R128 loudness normalization via ffmpeg loudnorm filter.
    Target: -16 LUFS (standard for speech/podcasts), true-peak -1.5 dBTP.
    Two-pass: first pass measures, second pass applies precise correction.
    """
    import json, re

    # Pass 1: measure loudness
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-"
    ], capture_output=True, text=True)

    # loudnorm stats come on stderr
    m = re.search(r'\{[^}]+\}', result.stderr, re.DOTALL)
    if not m:
        # fallback: single-pass if measurement fails
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src),
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", str(OUTPUT_SR), str(dst)
        ], check=True, capture_output=True)
        return

    stats = json.loads(m.group())

    # Pass 2: apply with measured values for accurate correction
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-af", (
            f"loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:"
            f"measured_I={stats['input_i']}:"
            f"measured_TP={stats['input_tp']}:"
            f"measured_LRA={stats['input_lra']}:"
            f"measured_thresh={stats['input_thresh']}:"
            f"offset={stats['target_offset']}:"
            f"print_format=summary"
        ),
        "-ar", str(OUTPUT_SR),
        str(dst)
    ], check=True, capture_output=True)


def process_video(video_path: Path, model, export_audio: bool = True) -> tuple[Path, Path]:
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*60}")

    video_out = video_path.with_stem(video_path.stem + OUTPUT_SUFFIX)
    audio_out = video_path.with_stem(video_path.stem + OUTPUT_SUFFIX).with_suffix(".wav")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        raw_wav = tmp / "raw.wav"
        processed_wav = tmp / "processed.wav"

        # 1. Extract audio
        print("\n[1/4] Extracting audio...")
        extract_audio(video_path, raw_wav, OUTPUT_SR)

        # 2. Load
        print("[2/4] Loading audio...")
        audio, sr = sf.read(str(raw_wav), dtype="float32")
        is_stereo = audio.ndim == 2
        print(f"      {len(audio)/sr:.1f}s, {'stereo' if is_stereo else 'mono'}, {sr}Hz")

        # 3. Enhance each channel
        print("[3/4] Enhancing with MetricGAN+ neural model...")
        if is_stereo:
            ch_l = enhance_channel(model, audio[:, 0], sr)
            ch_r = enhance_channel(model, audio[:, 1], sr)
            # Align lengths (resampling can shift by 1 sample)
            min_len = min(len(ch_l), len(ch_r))
            enhanced = np.stack([ch_l[:min_len], ch_r[:min_len]], axis=1)
        else:
            enhanced = enhance_channel(model, audio, sr)

        print("      Sharpening voice (presence EQ)...")
        enhanced = sharpen_voice(enhanced, sr)

        # 4. Write outputs
        print("[4/4] Writing output files...")
        # Write enhanced (pre-loudnorm) to temp WAV first
        sf.write(str(processed_wav), enhanced, sr, subtype="PCM_24")

        # Loudnorm → final standalone WAV
        loudnorm_normalized = tmp / "normalized.wav"
        print("      Applying EBU R128 loudness normalization (-16 LUFS)...")
        loudnorm_wav(processed_wav, loudnorm_normalized)

        if export_audio:
            import shutil
            shutil.copy(str(loudnorm_normalized), str(audio_out))
            print(f"      Audio: {audio_out.name}")

        # Enhanced + normalized video
        run([
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(loudnorm_normalized),
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", str(OUTPUT_SR),
            str(video_out)
        ])
        print(f"      Video: {video_out.name}")

    return video_out, audio_out if export_audio else None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhance audio in video files.")
    parser.add_argument("--input", metavar="FILE", help="Process a single video file.")
    parser.add_argument("--export-audio", action="store_true", help="Also export a standalone .wav file.")
    args = parser.parse_args()

    if args.input:
        videos = [Path(args.input)]
        if not videos[0].exists():
            print(f"File not found: {args.input}")
            sys.exit(1)
    else:
        # Folder mode: process all non-enhanced mp4s in the script's directory
        videos = sorted([
            v for v in VIDEO_DIR.glob("*.mp4")
            if "_enhanced" not in v.stem
        ])
        if not videos:
            print("No source .mp4 files found.")
            sys.exit(1)

    print(f"Found {len(videos)} video(s):")
    for v in videos:
        print(f"  {v.name}")

    print("\nLoading neural enhancement model...")
    model = load_model()

    for video in videos:
        process_video(video, model, export_audio=args.export_audio)

    print("\nAll done!")


if __name__ == "__main__":
    main()
