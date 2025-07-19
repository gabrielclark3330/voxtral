#!/bin/bash

export HF_HOME=/mnt/beegfs/cache
vllm serve openai/whisper-large-v3 --task transcription --port 8000 &

CUDA_VISIBLE_DEVICES=0 vllm serve openai/whisper-large-v3 --task transcription --port 8000 &
CUDA_VISIBLE_DEVICES=1 vllm serve openai/whisper-large-v3 --task transcription --port 8001 &
CUDA_VISIBLE_DEVICES=2 vllm serve openai/whisper-large-v3 --task transcription --port 8002 &
CUDA_VISIBLE_DEVICES=3 vllm serve openai/whisper-large-v3 --task transcription --port 8003 &

CUDA_VISIBLE_DEVICES=4 vllm serve openai/whisper-large-v3 --task transcription --port 8004 &
CUDA_VISIBLE_DEVICES=5 vllm serve openai/whisper-large-v3 --task transcription --port 8005 &
CUDA_VISIBLE_DEVICES=6 vllm serve openai/whisper-large-v3 --task transcription --port 8006 &
CUDA_VISIBLE_DEVICES=7 vllm serve openai/whisper-large-v3 --task transcription --port 8007 &
