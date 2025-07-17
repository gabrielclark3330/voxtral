#!/bin/bash

CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Voxtral-Small-24B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice &
CUDA_VISIBLE_DEVICES=1 vllm serve openai/whisper-large-v3 --task transcription --port 8001 &
#vllm serve mistralai/Voxtral-Small-24B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tensor-parallel-size 2 --tool-call-parser mistral --enable-auto-tool-choice
#vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral
