from __future__ import annotations
import argparse, asyncio, json, os, sys, itertools, zlib
from pathlib import Path
from typing import AsyncIterator, Dict, Sequence, Set, List, Optional
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio
import httpx
from openai.types.audio import Transcription, TranscriptionSegment
from tqdm.asyncio import tqdm_asyncio
import subprocess as sp
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError
import asyncio
import random
import torch
import json
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import time
import math
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data import DataLoader

OPENAI_API_KEY = "EMPTY"
AUDIO_EXTS: Set[str] = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".m4b", ".wma"}

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--rank", type=int, default=0, help="Rank index of this feeder")
ap.add_argument("-w", "--world-size", type=int, default=1, help="Total number of feeder processes")
ap.add_argument("--shard-seed", type=str, default=None, help="Run-specific seed for dataset sharding. If not provided, sharding is stable.")
args = ap.parse_args()

openai_api_key = "EMPTY"
openai_api_base = f"http://0.0.0.0:900{args.rank%8}/v1"
print(f"base {openai_api_base}")
OAI_MAX_RETRIES = 8
OAI_BASE_DELAY  = 1

client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

vad_model = load_silero_vad()
max_transcribe_chunk_duration = 29.0
transcribe_padding_seconds = 0.5
VAD_SR = 16_000          # 16 kHz
MAX_CHUNK = 29.0
PAD = 0.5

FOLDERS_TO_TRANSCRIBE = [
    "/nfsdata/datasets/tts/raw/Podcasts",
    #"/nfsdata/datasets/tts/raw/Podcasts_by_language",
    "/nfsdata/datasets/tts/raw/movies",
    "/nfsdata/datasets/tts/raw/downloaded_audiobooks",
    "/nfsdata/datasets/tts/raw/otheraudiobooks",
    #"/nfsdata/datasets/tts/raw/wyndlabs/",
    "/nfsdata/datasets/tts/raw/wyndlabs/min_filtered_v2",
    "/nfsdata/datasets/tts/raw/wyndlabs/minimal_channel_contents_2_updated",
]

def ffmpeg_read_audio_for_vad(
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

def split_long_vad_segments_for_padded_chunks(segments, max_total_duration, padding):
    max_content_duration = max_total_duration - 2 * padding
    if max_content_duration <= 0:
        raise ValueError("Padding too large for max_total_duration.")
    processed = []
    for seg in segments:
        dur = seg['end'] - seg['start']
        if dur <= max_content_duration:
            processed.append(seg)
        else:
            current = seg['start']
            while current < seg['end']:
                piece_end = min(current + max_content_duration, seg['end'])
                processed.append({'start': current, 'end': piece_end})
                current = piece_end
    return processed


def group_vad_segments_no_overlap(segments, max_total_duration, padding):
    """
    Group segments ensuring padded chunks never overlap.
    Returns list of chunk dicts with explicit padding info:
      {
        'content_start', 'content_end',
        'left_pad', 'right_pad',
        'padded_start', 'padded_end',
        'segments': [ ... original segment dicts ... ]
      }
    """
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s['start'])
    chunks = []
    current = [segs[0]]

    def finalize(cur):
        content_start = cur[0]['start']
        content_end = cur[-1]['end']
        left_pad = padding
        right_pad = padding
        padded_start = max(0.0, content_start - left_pad)
        padded_end = content_end + right_pad
        return {
            'segments': cur,
            'content_start': content_start,
            'content_end': content_end,
            'left_pad': left_pad,
            'right_pad': right_pad,
            'padded_start': padded_start,
            'padded_end': padded_end
        }

    for seg in segs[1:]:
        tentative = current + [seg]
        tentative_start = tentative[0]['start']
        tentative_end = tentative[-1]['end']
        padded_span = (tentative_end + padding) - (tentative_start - padding)
        if padded_span <= max_total_duration:
            current.append(seg)
        else:
            chunks.append(finalize(current))
            current = [seg]

    chunks.append(finalize(current))

    i = 0
    while i < len(chunks) - 1:
        a = chunks[i]
        b = chunks[i + 1]
        a['padded_start'] = a['content_start'] - a['left_pad']
        a['padded_end'] = a['content_end'] + a['right_pad']
        b['padded_start'] = b['content_start'] - b['left_pad']
        b['padded_end'] = b['content_end'] + b['right_pad']

        if a['padded_end'] > b['padded_start']:
            overlap = a['padded_end'] - b['padded_start']
            trim_prev_cap = a['right_pad']
            trim_next_cap = b['left_pad']

            trim_prev = min(trim_prev_cap, overlap / 2.0)
            overlap -= trim_prev
            trim_next = min(trim_next_cap, overlap)
            overlap -= trim_next

            a['right_pad'] -= trim_prev
            b['left_pad'] -= trim_next

            a['padded_end'] = a['content_end'] + a['right_pad']
            b['padded_start'] = b['content_start'] - b['left_pad']

            if a['padded_end'] > b['padded_start']:
                combined_padded_start = min(a['padded_start'], b['padded_start'])
                combined_padded_end = max(a['padded_end'], b['padded_end'])
                combined_span = combined_padded_end - combined_padded_start
                if combined_span <= max_total_duration:
                    merged_segments = a['segments'] + b['segments']
                    merged = {
                        'segments': merged_segments,
                        'content_start': a['content_start'],
                        'content_end': b['content_end'],
                        'left_pad': max(a['left_pad'], padding),
                        'right_pad': max(b['right_pad'], padding),
                    }
                    merged['padded_start'] = merged['content_start'] - merged['left_pad']
                    merged['padded_end'] = merged['content_end'] + merged['right_pad']
                    chunks[i] = merged
                    del chunks[i + 1]
                    continue
                else:
                    b['left_pad'] = 0.0
                    b['padded_start'] = b['content_start']
                    if a['padded_end'] > b['padded_start']:
                        a['right_pad'] = 0.0
                        a['padded_end'] = a['content_end']
        i += 1

    return chunks


def audio_file_to_wav16k_bytes_voxtral(
    path: str,
    off_s: float = 0.0,
    dur_s: float | None = None,
    ffmpeg_bin: str = "ffmpeg",
    use_soxr: bool = True,
    pcm_float: bool = False,
) -> bytes:
    """
    Convert an arbitrary audio file to 16 kHz mono WAV bytes.

    Args:
        path: Input audio file path.
        off_s: Optional start offset (seconds).
        dur_s: Optional duration (seconds) to read.
        ffmpeg_bin: Path to ffmpeg binary.
        use_soxr: If True, use high quality soxr resampler.
        pcm_float: If True, encode WAV as 32-bit float (pcm_f32le). Otherwise 16-bit PCM.

    Returns:
        WAV file bytes ready to put in an in-memory buffer for an API request.
    """
    codec = "pcm_f32le" if pcm_float else "pcm_s16le"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel", "error",
        "-ss", f"{off_s}",
        "-i", path,
        "-vn", "-sn",
        "-ac", "1",
    ]
    if dur_s is not None:
        cmd += ["-t", f"{dur_s}"]

    if use_soxr:
        cmd += ["-af", "aresample=16000:resampler=soxr"]
        cmd += ["-ar", "16000"]
    else:
        cmd += ["-ar", "16000"]

    cmd += [
        "-f", "wav",
        "-acodec", codec,
        "-"
    ]

    try:
        proc = sp.run(cmd, check=True, capture_output=True)
    except sp.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed ({e.returncode}): {e.stderr.decode(errors='ignore')}"
        ) from e

    return proc.stdout

async def write_json(dst, segments) -> None:
    """Converts segment objects to dicts and writes a JSON file."""
    out = {
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "words": None,
                "speaker": None,
            }
            for seg in segments
        ],
        "source": "mixtral-mini",
    }
    print(f"writing {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(dst.write_text, json.dumps(out, ensure_ascii=False, indent=2))

def _iter_audio_files(root: Path) -> Iterable[Path]:
    """
    Deterministic depth-first directory walk yielding audio file Paths.
    Each directory's entries are processed in sorted order to ensure reproducibility.
    Uses an explicit stack (no recursion) to control ordering.
    """
    if not root.is_dir():
        return
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                dirs: List[str] = []
                files: List[str] = []
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(entry.name)
                    elif entry.is_file():
                        if Path(entry.name).suffix.lower() in AUDIO_EXTS:
                            files.append(entry.name)
                # Yield files in sorted order
                for fname in sorted(files):
                    yield d / fname
                # Add subdirectories to stack in reverse sorted order (so forward sorted pops)
                for sub in sorted(dirs, reverse=True):
                    stack.append(d / sub)
        except (PermissionError, FileNotFoundError):
            continue


def drop_first_n(path, n):
    p = Path(path)
    parts = p.parts
    if p.is_absolute():
        kept = parts[1 + n:]
        return str(Path(*kept)) if kept else ""
    else:
        kept = parts[n:]
        return str(Path(*kept)) if kept else ""

def get_output_path(path):
    dst = Path("/nfsdata/gabrielc/tts-v2/data_processing") / Path(drop_first_n(path, 4)).with_suffix(".json")
    return dst


class VadChunkDataset(IterableDataset):
    """
    Streaming, deterministic iterable dataset producing per-file transcription
    chunk request arrays with disjoint partition over (rank, worker) without
    preloading all file paths into memory.

    Sharding strategy:
      - Every candidate audio file path is hashed via adler32 over its POSIX path string.
      - global_worker_id := rank * num_workers + worker_id
      - File is processed iff (hash % total_shards) == global_worker_id

    This keeps memory usage O(1) in number of files and is stable across runs
    (assuming directory contents unchanged).
    """

    def __init__(
        self,
        roots: Sequence[str | Path],
        *,
        rank: int = 0,
        world_size: int = 1,
        max_transcribe_chunk_duration: float = 30.0,
        transcribe_padding_seconds: float = 0.5,
        model_name: str = "mistralai/Voxtral-Mini-3B-2507",
        language: str = "en",
        temperature: float = 0.0,
        preload_vad_model: bool = True,
        shard_seed=None,
        hash_fn = zlib.adler32,  # Accept injectable hash for testing if desired
    ) -> None:
        super().__init__()
        self.roots = [Path(r) for r in roots]
        self.max_transcribe_chunk_duration = max_transcribe_chunk_duration
        self.transcribe_padding_seconds = transcribe_padding_seconds
        self.model_name = model_name
        self.language = language
        self.temperature = temperature
        self.vad_model = load_silero_vad() if preload_vad_model else None
        self.rank = rank
        self.world_size = world_size
        self.hash_fn = hash_fn
        self.shard_seed=shard_seed

    def _ensure_vad_model(self):
        if self.vad_model is None:
            self.vad_model = load_silero_vad()

    def _get_worker_info(self) -> Tuple[int, int]:
        wi = get_worker_info()
        if wi is None:
            return 0, 1
        return wi.id, wi.num_workers

    def _process_file(self, path: Path):
        """
        Produce (path, request_arr) for a single audio file.
        """
        wav_for_vad = ffmpeg_read_audio_for_vad(path, sampling_rate=VAD_SR)
        speech_timestamps = get_speech_timestamps(
            wav_for_vad, self.vad_model, return_seconds=True
        )

        pre_segments = split_long_vad_segments_for_padded_chunks(
            speech_timestamps,
            self.max_transcribe_chunk_duration,
            self.transcribe_padding_seconds,
        )

        chunks = group_vad_segments_no_overlap(
            pre_segments,
            self.max_transcribe_chunk_duration,
            self.transcribe_padding_seconds,
        )

        file_duration = (wav_for_vad.numel() / VAD_SR)-0.10

        request_arr = []
        for chunk in chunks:
            start = chunk['padded_start']
            end   = min(chunk['padded_end'], file_duration)
            if start >= file_duration:
                continue
            dur = end - start
            wav_bytes = audio_file_to_wav16k_bytes_voxtral(
                path, dur_s=dur, off_s=start
            )
            request_arr.append(
                {
                    "temperature": self.temperature,
                    "model": self.model_name,
                    "language": self.language,
                    "file": wav_bytes,
                    "start": start,
                    "end": end,
                }
            )
        return path, request_arr

    def __iter__(self):
        self._ensure_vad_model()

        worker_id, num_workers = self._get_worker_info()
        total_shards = self.world_size * num_workers
        global_worker_id = self.rank * num_workers + worker_id

        for root in self.roots:
            if not root.is_dir():
                continue
            for path in _iter_audio_files(root):
                path_str = path.as_posix()
                if self.shard_seed is not None:
                    string_to_hash = self.shard_seed + path_str
                else:
                    string_to_hash = path_str
                h = self.hash_fn(string_to_hash.encode("utf-8")) & 0xFFFFFFFF

                dst = get_output_path(path)
                if h % total_shards != global_worker_id or dst.exists():
                    continue
                try:
                    yield self._process_file(path)
                except Exception as e:
                    print(e)
                    continue

dataset = VadChunkDataset(
    roots=FOLDERS_TO_TRANSCRIBE,
    rank = args.rank,
    world_size = args.world_size,
    max_transcribe_chunk_duration=max_transcribe_chunk_duration,
    transcribe_padding_seconds=transcribe_padding_seconds,
    model_name="mistralai/Voxtral-Mini-3B-2507",
    language="en",
    temperature=0.0,
    shard_seed=args.shard_seed,
)

loader = DataLoader(
    dataset,
    batch_size=None,
    num_workers=32,
    pin_memory=False,
    worker_init_fn=lambda _: torch.set_num_threads(1),
)


async def worker(path, requests):
    dst = get_output_path(path)
    if dst.exists():
        print(f"{dst} already exists")
        return

    try:
        segment_arr = []
        for req in requests:
            chunk_start = req.pop("start")
            chunk_end = req.pop("end")
            #response = await client.audio.transcriptions.create(**req, request_timeout=90)
            for attempt in range(OAI_MAX_RETRIES):
                try:
                    response = await client.audio.transcriptions.create(**req)
                except (Timeout, APIConnectionError) as err:
                    if attempt == OAI_MAX_RETRIES - 1:
                        raise
                    delay = OAI_BASE_DELAY * math.pow(2, attempt) + random.uniform(0, 1)
                    logging.warning("Timeout (%s) - retrying in %.1fs", err, delay)
                    await asyncio.sleep(delay)

            segment_arr.append({"text": response.text, "start":chunk_start, "end":chunk_end})

        await write_json(dst, segment_arr)
    except Exception as e:
        print(e)

async def worker(path, requests):
    dst = get_output_path(path)
    if dst.exists():
        print(f"SKIPPING: {dst} already exists")
        return

    try:
        segment_arr = []
        for req in requests:
            chunk_start = req.pop("start")
            chunk_end = req.pop("end")

            response = None

            for attempt in range(OAI_MAX_RETRIES):
                try:
                    response = await client.audio.transcriptions.create(**req)
                    break
                except (APITimeoutError, APIConnectionError, httpx.ReadTimeout) as err:
                    if attempt == OAI_MAX_RETRIES - 1:
                        print(f"ERROR: Chunk for {path} failed after {OAI_MAX_RETRIES} retries. Skipping file. Error: {err}")
                        return

                    delay = OAI_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    print(f"WARNING: API error on {path}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)

            if response:
                segment_arr.append({"text": response.text, "start": chunk_start, "end": chunk_end})
            else:
                print(f"ERROR: Could not process a chunk for {path}. Skipping file.")
                return 

        if segment_arr:
            await write_json(dst, segment_arr)

    except Exception as e:
        print(f"FATAL WORKER ERROR for path {path}: {type(e).__name__} - {e}")

async def main():
    pending = set()
    results = []
    concurrency=32

    async def schedule(path, requests):
        t = asyncio.create_task(worker(path, requests))
        pending.add(t)

    for path, requests in loader:
        if len(pending) >= concurrency:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                results.append(await d)
        await schedule(path, requests)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            results.append(await d)

if __name__ == "__main__":
    asyncio.run(main())
