#!/bin/bash

export HF_HOME=/mnt/beegfs/cache
#vllm serve mistralai/Voxtral-Small-24B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice  --port 8000  --tensor-parallel-size 4 &

CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice  --port 8000 &
CUDA_VISIBLE_DEVICES=1 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice  --port 8001 &
CUDA_VISIBLE_DEVICES=2 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice  --port 8002 &
CUDA_VISIBLE_DEVICES=3 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice  --port 8003 &

CUDA_VISIBLE_DEVICES=4 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice  --port 8004 &
CUDA_VISIBLE_DEVICES=5 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice  --port 8005 &
CUDA_VISIBLE_DEVICES=6 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice  --port 8006 &
CUDA_VISIBLE_DEVICES=7 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice  --port 8007 &
