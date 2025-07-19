from __future__ import annotations

import argparse, asyncio, json, os, sys, random
from pathlib import Path
from typing import AsyncIterator, Dict, Sequence, Set, List

import httpx
from openai import AsyncOpenAI
from openai.types.audio import Transcription, TranscriptionSegment
from tqdm.asyncio import tqdm_asyncio


PORTS: Sequence[int] = range(8000, 8008)
OPENAI_API_KEY = "EMPTY"
AUDIO_EXTS: Set[str] = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".m4b", ".wma"}

FOLDERS_TO_TRANSCRIBE = [
    #"/nfsdata/datasets/tts/raw/Podcasts",
    "/nfsdata/datasets/tts/raw/Podcasts_by_language",
    "/nfsdata/datasets/tts/raw/movies",
    "/nfsdata/datasets/tts/raw/downloaded_audiobooks",
    "/nfsdata/datasets/tts/raw/otheraudiobooks",
    "/nfsdata/datasets/tts/raw/wyndlabs/min_filtered_v2",
    "/nfsdata/datasets/tts/raw/wyndlabs/minimal_channel_contents_2_updated",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--local-rank", type=int, default=0,
                    help="Rank index of this feeder (for multi-proc sharding)")
    ap.add_argument("-w", "--world-size", type=int, default=1,
                    help="Total number of feeder processes")
    ap.add_argument("-q", "--queue-cap", type=int, default=10,
                    help="Pause when a worker queue reaches this size")
    return ap.parse_args()


async def all_audio() -> AsyncIterator[tuple[Path, bytes]]:
    """Yield (path, bytes) for every audio file under every root."""
    MAX_BYTES = 25 * 1024 * 1024  # 25 MB
    for folder in FOLDERS_TO_TRANSCRIBE:
        root = Path(folder)
        if not root.is_dir():
            sys.stderr.write(f"[WARN] Folder missing or not a directory: {root}\n")
            continue

        for path in root.rglob("*"):
            if path.suffix.lower() in AUDIO_EXTS and path.is_file():
                try:
                    if path.stat().st_size > MAX_BYTES:
                        # Use tqdm.write to avoid breaking the progress bar
                        tqdm_asyncio.write(f"[SKIP] {path} is too large: {path.stat().st_size / 1e6:.1f} MB")
                        continue
                    # Reading files can block, so run it in a thread
                    data = await asyncio.to_thread(path.read_bytes)
                    yield path, data
                except OSError as e:
                    tqdm_asyncio.write(f"[WARN] Could not read file {path}: {e}")


async def queue_len(client: httpx.AsyncClient, port: int) -> float:
    """
    Return vllm:num_requests_waiting for *port*.
    Returns float('inf') if the server is unreachable.
    """
    try:
        r = await client.get(f"http://127.0.0.1:{port}/metrics", timeout=0.5)
        r.raise_for_status()  # Raise an exception for non-2xx status codes
        for line in r.text.splitlines():
            if line.startswith("vllm:num_requests_waiting"):
                return float(line.rsplit(" ", 1)[-1])
        # Metric not found, but server is alive. Assume queue is 0.
        return 0.0
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        # If server is down, timed out, or gives an error, it's "infinitely" busy
        # so it won't be selected by the load balancer.
        # Uncomment the next line for debugging connection issues.
        # tqdm_asyncio.write(f"[DEBUG] Could not connect to port {port}: {e}")
        return float('inf')


async def write_json(dst: Path, segments: list[TranscriptionSegment]) -> None:
    """Converts segment objects to dicts and writes a JSON file."""
    out = {
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": None, # word-level timestamps are not in the standard response
                "speaker": "p001",
            }
            for seg in segments
        ],
        "source": "whisper-large-v3-vllm",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(dst.write_text, json.dumps(out, ensure_ascii=False, indent=2))


async def transcribe(api: AsyncOpenAI, port: int, data: bytes, src: Path) -> None:
    """Sends audio data to a vLLM endpoint for transcription."""
    try:
        # The file argument to the API needs to be a tuple of (filename, file_data)
        file_tuple = (src.name, data)
        resp: Transcription = await api.audio.transcriptions.create(
            file=file_tuple,
            model="openai/whisper-large-v3",
            language="en",
            response_format="verbose_json", # Use verbose_json to get segments
            temperature=0.0,
        )
        dst = src.with_suffix(".json")
        if resp.segments:
            await write_json(dst, resp.segments)
        else:
            tqdm_asyncio.write(f"[INFO] No segments found for {src} via port {port}")

    except Exception as exc:
        # Using tqdm.write is safer inside the async loop than stderr.write or print
        tqdm_asyncio.write(f"[WARN] Failed to transcribe {src} via port {port}: {exc}")


async def main() -> None:
    args = parse_args()

    clients: Dict[int, AsyncOpenAI] = {
        p: AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=f"http://127.0.0.1:{p}/v1")
        for p in PORTS
    }

    async with httpx.AsyncClient() as metrics_client:
        pending: Set[asyncio.Task] = set()
        idx = 0

        # Create an async iterator for the audio files
        audio_iterator = all_audio()

        async for path, blob in tqdm_asyncio(audio_iterator, desc=f"Rank {args.local_rank}", unit=" file"):
            if idx % args.world_size != args.local_rank:
                idx += 1
                continue
            idx += 1

            # Loop until a worker is available
            while True:
                qlens = await asyncio.gather(*[queue_len(metrics_client, p) for p in PORTS])

                min_qlen = min(qlens)

                # If all workers are down, qlens will be all 'inf'
                if min_qlen == float('inf'):
                    tqdm_asyncio.write("[ERROR] All VLLM workers are down. Exiting.")
                    await asyncio.sleep(5) # Wait before exiting to see message
                    return

                # If the least busy worker is already at capacity, wait.
                if min_qlen >= args.queue_cap:
                    await asyncio.sleep(0.1)
                    continue

                # Find all workers with the minimum queue length and pick one randomly
                # to distribute the load evenly when multiple workers are idle.
                best_ports = [p for i, p in enumerate(PORTS) if qlens[i] == min_qlen]
                port = random.choice(best_ports)

                # We found a free worker, break the while loop and submit the task
                break

            task = asyncio.create_task(transcribe(clients[port], port, blob, path))
            pending.add(task)
            # This callback automatically removes the task from the set when it's done.
            task.add_done_callback(pending.discard)

        # Wait for any remaining tasks to complete after the loop finishes.
        if pending:
            tqdm_asyncio.write(f"All files queued. Waiting for {len(pending)} remaining tasks...")
            await asyncio.gather(*pending)

if __name__ == "__main__":
    asyncio.run(main())