from __future__ import annotations

import argparse, asyncio, json, os, sys, itertools
from pathlib import Path
from typing import AsyncIterator, Dict, Sequence, Set, List

# --- New Imports for the Mistral-style API ---
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio
# --- End New Imports ---

import httpx
from openai import AsyncOpenAI
# Note: The response object is still compatible with the standard openai types
from openai.types.audio import Transcription, TranscriptionSegment
from tqdm.asyncio import tqdm_asyncio


PORTS: Sequence[int] = range(8000, 8008)
OPENAI_API_KEY = "EMPTY"
AUDIO_EXTS: Set[str] = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".m4b", ".wma"}

FOLDERS_TO_TRANSCRIBE = [
    "/nfsdata/datasets/tts/raw/Podcasts",
    "/nfsdata/datasets/tts/raw/Podcasts_by_language",
    "/nfsdata/datasets/tts/raw/movies",
    "/nfsdata/datasets/tts/raw/downloaded_audiobooks",
    "/nfsdata/datasets/tts/raw/otheraudiobooks",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--local-rank", type=int, default=0, help="Rank index of this feeder")
    ap.add_argument("-w", "--world-size", type=int, default=1, help="Total number of feeder processes")
    ap.add_argument("-c", "--concurrency", type=int, default=16,
                    help="Max number of in-flight requests from this client.")
    return ap.parse_args()


async def all_audio() -> AsyncIterator[tuple[Path, bytes]]:
    """Yield (path, bytes) for every audio file under every root."""
    MAX_BYTES = 25 * 1024 * 1024
    for folder in FOLDERS_TO_TRANSCRIBE:
        root = Path(folder)
        if not root.is_dir():
            sys.stderr.write(f"[WARN] Folder missing or not a directory: {root}\n")
            continue

        for path in root.rglob("*"):
            if path.suffix.lower() in AUDIO_EXTS and path.is_file():
                try:
                    if path.stat().st_size > MAX_BYTES:
                        tqdm_asyncio.write(f"[SKIP] {path} is too large: {path.stat().st_size / 1e6:.1f} MB")
                        continue
                    data = await asyncio.to_thread(path.read_bytes)
                    yield path, data
                except OSError as e:
                    tqdm_asyncio.write(f"[WARN] Could not read file {path}: {e}")


async def write_json(dst: Path, segments: list[TranscriptionSegment]) -> None:
    """Converts segment objects to dicts and writes a JSON file."""
    out = {
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": None,
                "speaker": "p001",
            }
            for seg in segments
        ],
        "source": "new-vllm-model",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(dst.write_text, json.dumps(out, ensure_ascii=False, indent=2))


async def transcribe(
    api: AsyncOpenAI, model_name: str, port: int, data: bytes, src: Path
) -> None:
    """
    Sends audio data to a vLLM endpoint using the mistral_common request format.
    """
    try:
        audio_obj = await asyncio.to_thread(Audio.from_bytes, data, strict=False)

        raw_audio_obj = RawAudio.from_audio(audio_obj)

        req = TranscriptionRequest(
            model=model_name,
            audio=raw_audio_obj,
            language="en",
            temperature=0.0
        )

        req_dict = await asyncio.to_thread(req.to_openai, exclude=("top_p", "seed"))

        resp: Transcription = await api.audio.transcriptions.create(**req_dict)
        print(resp)
        dst = src.with_suffix(".json")
        if resp.segments:
            await write_json(dst, resp.segments)
        else:
            tqdm_asyncio.write(f"[INFO] No segments found for {src} via port {port}")

    except Exception as exc:
        tqdm_asyncio.write(f"[WARN] Failed to transcribe {src} via port {port}: {exc}")


async def main() -> None:
    args = parse_args()

    clients: Dict[int, AsyncOpenAI] = {
        p: AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=f"http://127.0.0.1:{p}/v1")
        for p in PORTS
    }

    first_port = PORTS[0]
    first_client = clients[first_port]
    model_name: str
    try:
        models = await first_client.models.list()
        if not models.data:
            raise ValueError("No models found on the server.")
        model_name = models.data[0].id
        tqdm_asyncio.write(f"Discovered model '{model_name}' from server on port {first_port}.")
    except Exception as e:
        sys.stderr.write(f"[FATAL] Could not get model list from server on port {first_port}: {e}\n")
        return

    port_cycler = itertools.cycle(PORTS)
    pending: Set[asyncio.Task] = set()
    file_idx = 0
    audio_iterator = all_audio()

    async for path, blob in tqdm_asyncio(audio_iterator, desc=f"Rank {args.local_rank}", unit=" file"):
        if file_idx % args.world_size != args.local_rank:
            file_idx += 1
            continue
        file_idx += 1

        if len(pending) >= args.concurrency:
            _done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        port = next(port_cycler)
        
        # Pass the discovered model_name to the transcribe function
        task = asyncio.create_task(transcribe(clients[port], model_name, port, blob, path))
        pending.add(task)

    if pending:
        tqdm_asyncio.write(f"All files queued. Waiting for {len(pending)} remaining tasks...")
        await asyncio.gather(*pending)

if __name__ == "__main__":
    asyncio.run(main())