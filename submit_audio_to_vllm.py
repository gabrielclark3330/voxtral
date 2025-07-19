from __future__ import annotations

import argparse, asyncio, json, os, sys
from pathlib import Path
from typing import AsyncIterator, Dict, Sequence, Set

import httpx
from openai import AsyncOpenAI
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
    for folder in FOLDERS_TO_TRANSCRIBE:
        root = Path(folder)
        if not root.exists():
            sys.stderr.write(f"[WARN] Folder missing: {root}\n")
            continue
        MAX_BYTES = 25 * 1024 * 1024  # 25 MB

        for path in root.rglob("*"):
            if path.suffix.lower() in AUDIO_EXTS and path.is_file():
                if path.stat().st_size > MAX_BYTES:
                    print(f"[SKIP] {path} is {path.stat().st_size / 1e6:.1f} MB")
                    continue
                data = await asyncio.to_thread(path.read_bytes)
                yield path, data

async def queue_len(client: httpx.AsyncClient, port: int) -> int:
    """Return vllm:num_requests_waiting for *port*."""
    try:
        r = await client.get(f"http://127.0.0.1:{port}/metrics", timeout=0.3)
        for line in r.text.splitlines():
            if line.startswith("vllm:num_requests_waiting"):
                return int(float(line.rsplit(" ", 1)[-1]))
    except Exception:
        pass
    return 0

async def write_json(dst: Path, segments_json: list[dict]) -> None:
    out = {
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "words": None,
                "speaker": "p001",
            }
            for seg in segments_json
        ],
        "source": "ground_truth",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(dst.write_text, json.dumps(out, ensure_ascii=False, indent=2))

async def transcribe(api: AsyncOpenAI, port: int, data: bytes, src: Path) -> None:
    try:
        resp = await api.audio.transcriptions.create(
            file=data,
            model="openai/whisper-large-v3",
            language="en",
            response_format="json",   # gives timestamped segments
            temperature=0.0,
        )
        dst = src.with_suffix(".json")
        print(f"{resp=}")
        #await write_json(dst, resp.segments)  # type: ignore[attr-defined]
    except Exception as exc:
        sys.stderr.write(f"[WARN] {src} via port {port}: {exc}\n")


async def main() -> None:
    args = parse_args()

    clients: Dict[int, AsyncOpenAI] = {
        p: AsyncOpenAI(api_key=OPENAI_API_KEY,
                       base_url=f"http://127.0.0.1:{p}/v1")
        for p in PORTS
    }

    async with httpx.AsyncClient(timeout=0.3) as metrics_client:
        pending: set[asyncio.Task] = set()
        idx = 0

        async for path, blob in tqdm_asyncio(all_audio(),
                                             desc=f"rank {args.local_rank}",
                                             unit="file"):
            if idx % args.world_size != args.local_rank:
                idx += 1
                continue
            idx += 1

            qlens = await asyncio.gather(*[queue_len(metrics_client, p) for p in PORTS])
            port = min(PORTS, key=lambda p: qlens[p - PORTS.start])

            if qlens[port - PORTS.start] >= args.queue_cap:
                await asyncio.sleep(0.05)
                continue

            task = asyncio.create_task(transcribe(clients[port], port, blob, path))
            pending.add(task)
            task.add_done_callback(pending.discard)
            if idx>50:
                break

        if pending:
            await asyncio.gather(*pending)

if __name__ == "__main__":
    asyncio.run(main())
