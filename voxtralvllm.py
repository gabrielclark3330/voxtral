from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio
import subprocess as sp
from openai import AsyncOpenAI
import asyncio
import random

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"

client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

#ffmpeg_bin = "/nfsdata/osman/workspace_2/cursed_sox/cursed_ffmpeg/ffmpeg-build-script-1.50/workspace/bin/ffmpeg"

def audio_file_to_wav16k_bytes(
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

path = "mohamedcapellaen.mp3"
path = "iamrussian.mp3"
path = "realquiet.mp3"
path = "backgroundspeaker.mp3"
path = "sluredwords.mp3"
path = "bestinterviewever.mp3"
path = "yellingattrumpet.mp3"
path = "whisper-synthetic-5.mp3"
path = "3min-kitchen-audio.mp3"
path = "lillsush.mp3"
path = "/nfsdata/datasets/tts/raw/movies/Zootopia (2016) [1080p] [YTS.AG]/Zootopia.2016.1080p.BluRay.x264-[YTS.AG].m4a"

async def worker():
    req={'temperature': 0.0, 'model': 'mistralai/Voxtral-Mini-3B-2507', 'language': 'en', 'file': audio_file_to_wav16k_bytes(path, dur_s=29, off_s=60*random.randint(1,100))}
    response = await client.audio.transcriptions.create(**req)
    print(response.text)
    return response.text

async def main():
    results = []
    try:
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(worker()) for i in range(400)]
        for t in tasks:
            results.append(t.result())
        print(results)
    except* RuntimeError as eg:
        for e in eg.exceptions:
            print("A worker failed:", e)

asyncio.run(main())