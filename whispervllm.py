import asyncio
import json
import httpx
from openai import OpenAI
from vllm.assets.audio import AudioAsset

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8001/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

audio_path = "realquiet.mp3"
audio_path = "iamrussian.mp3"
audio_path = "yellingattrumpet.mp3"
audio_path = "whisper-synthetic-5.mp3"
audio_path = "lillsush.mp3"

def sync_openai():
    with open(str(audio_path), "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-large-v3",
            language="en",
            response_format="json",
            temperature=0.0)
        print("transcription result:", transcription.text)


sync_openai()
