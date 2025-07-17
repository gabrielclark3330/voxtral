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

#async def stream_openai_response():
    #data = {
        #"language": "en",
        #'stream': True,
        #"model": "openai/whisper-large-v3",
    #}
    #url = openai_api_base + "/audio/transcriptions"
    #print("transcription result:", end=' ')
    #async with httpx.AsyncClient() as client:
        #with open(str(winning_call), "rb") as f:
            #async with client.stream('POST', url, files={'file': f},
                                     #data=data) as response:
                #async for line in response.aiter_lines():
                    ## Each line is a JSON object prefixed with 'data: '
                    #if line:
                        #if line.startswith('data: '):
                            #line = line[len('data: '):]
                        ## Last chunk, stream ends
                        #if line.strip() == '[DONE]':
                            #break
                        ## Parse the JSON response
                        #chunk = json.loads(line)
                        ## Extract and print the content
                        #content = chunk['choices'][0].get('delta',
                                                          #{}).get('content')
                        #print(content, end='')

#asyncio.run(stream_openai_response())
