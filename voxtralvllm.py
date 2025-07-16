from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio

from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

#audio = Audio.from_file("mohamedcapellaen.mp3", strict=False)
#audio = Audio.from_file("iamrussian.mp3", strict=False)
#audio = Audio.from_file("realquiet.mp3", strict=False)
#audio = Audio.from_file("backgroundspeaker.mp3", strict=False)
#audio = Audio.from_file("sluredwords.mp3", strict=False)
#audio = Audio.from_file("bestinterviewever.mp3", strict=False)
#audio = Audio.from_file("whisper-synthetic-5.mp3", strict=False)
#audio = Audio.from_file("3min-kitchen-audio.mp3", strict=False)
audio = Audio.from_file("lillsush.mp3", strict=False)

audio = RawAudio.from_audio(audio)
req = TranscriptionRequest(model=model, audio=audio, language="en", temperature=0.0).to_openai(exclude=("top_p", "seed"))

response = client.audio.transcriptions.create(**req)
print(response)
