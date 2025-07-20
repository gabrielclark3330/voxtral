import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import time
import subprocess as sp
import math
from typing import Optional


def ffmpeg_read_audio(
    path: str,
    sampling_rate: int = 16000,
    ffmpeg_bin: str = "ffmpeg",
    use_soxr: bool = True,
    seek_s: float = 0.0,
    dur_s: Optional[float] = None,
    float_output: bool = True,
    accurate_seek: bool = True,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """
    Decode `path` to a mono audio tensor at `sampling_rate` using ffmpeg.

    Args:
        path: Input media path (any format ffmpeg can read).
        sampling_rate: Target sample rate (Hz).
        ffmpeg_bin: Path to ffmpeg binary.
        use_soxr: Use high-quality soxr resampler (aresample). If False, ffmpeg default.
        seek_s: Optional start offset in seconds.
        dur_s: Optional duration (seconds) to read after seek.
        float_output: If True, request pcm_f32le; otherwise pcm_s16le then convert to float.
        accurate_seek: If True, place -ss *after* -i for sample-accurate seek. If False,
                       place -ss *before* -i (fast but may land on keyframe).
        device: Optional torch device (e.g. "cuda", torch.device("cuda:0")).

    Returns:
        1-D float32 torch.Tensor of audio samples in range [-1, 1].
    """

    codec = "pcm_f32le" if float_output else "pcm_s16le"
    bytes_per_sample = 4 if float_output else 2

    cmd = [ffmpeg_bin, "-hide_banner", "-nostdin", "-loglevel", "error"]

    if seek_s > 0 and not accurate_seek:
        cmd += ["-ss", f"{seek_s}"]

    cmd += ["-i", path]

    if seek_s > 0 and accurate_seek:
        cmd += ["-ss", f"{seek_s}"]

    if dur_s is not None:
        cmd += ["-t", f"{dur_s}"]

    cmd += ["-ac", "1"]

    if use_soxr:
        cmd += ["-af", f"aresample={sampling_rate}:resampler=soxr"]
        cmd += ["-ar", f"{sampling_rate}"]
    else:
        cmd += ["-ar", f"{sampling_rate}"]

    cmd += [
        "-f", codec.replace("pcm_", "").replace("le", "le"),
        "-acodec", codec,
        "-"
    ]

    try:
        proc = sp.run(cmd, check=True, capture_output=True)
    except sp.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed (exit {e.returncode}): {e.stderr.decode(errors='ignore')}"
        ) from e

    raw = proc.stdout
    if len(raw) == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    if float_output:
        audio = torch.frombuffer(raw, dtype=torch.float32)
    else:
        audio_i16 = torch.frombuffer(raw, dtype=torch.int16)
        audio = audio_i16.to(torch.float32) / 32768.0

    if device is not None:
        audio = audio.to(device)

    if dur_s is not None:
        expected = int(round(dur_s * sampling_rate))
        tol = int(0.01 * sampling_rate)  # 10 ms tolerance
        if abs(audio.numel() - expected) > tol:
            pass

    return audio.contiguous()


path = "mohamedcapellaen.mp3"
path = "iamrussian.mp3"
path = "realquiet.mp3"
path = "yellingattrumpet.mp3"
path = "lillsush.mp3"
path = "3min-kitchen-audio.mp3"
path = "whisper-synthetic-5.mp3"
path = "bestinterviewever.mp3"
path = "sluredwords.mp3"
path = "backgroundspeaker.mp3"
path = "/nfsdata/datasets/tts/raw/movies/Zootopia (2016) [1080p] [YTS.AG]/Zootopia.2016.1080p.BluRay.x264-[YTS.AG].m4a"

sampling_rate = 16000
show_activity_mask = True

model = load_silero_vad()

wav = ffmpeg_read_audio(path, sampling_rate=sampling_rate)
start = time.time()
speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
print(f"took {time.time()-start}s")

print("Detected segments (seconds):", speech_timestamps)

num_samples = wav.shape[0]
time = torch.arange(num_samples) / sampling_rate

if show_activity_mask:
    fig, (ax_wav, ax_mask) = plt.subplots(2, 1, sharex=True, figsize=(12, 6),
                                          gridspec_kw={"height_ratios": [3, 1]})
else:
    fig, ax_wav = plt.subplots(1, 1, figsize=(12, 4))

if show_activity_mask:
    ax = ax_wav
else:
    ax = ax_wav

ax.plot(time, wav.numpy(), linewidth=0.6)

for seg in speech_timestamps:
    start = seg['start']
    end = seg['end']
    ax.axvspan(start, end, alpha=0.25)

ax.set_ylabel("Amplitude")
ax.set_title(f"Waveform with Detected Speech Segments ({path})")

if show_activity_mask:
    mask = torch.zeros_like(time)
    for seg in speech_timestamps:
        s_idx = int(seg['start'] * sampling_rate)
        e_idx = int(seg['end'] * sampling_rate)
        mask[s_idx:e_idx] = 1.0
    ax_mask.plot(time, mask.numpy(), linewidth=0.8)
    ax_mask.set_yticks([0, 1])
    ax_mask.set_yticklabels(["silence", "speech"])
    ax_mask.set_xlabel("Time (s)")
    ax_mask.set_ylabel("VAD")

plt.tight_layout()
plt.savefig("waveform_vad.png", dpi=150)
