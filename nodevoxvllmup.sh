#!/bin/bash

export HF_HOME=/mnt/beegfs/cache
RUN_ID=$(date +%Y%m%d-%H%M%S)_$RANDOM
mkdir -p vllmlogs

CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --port 9000  > vllmlogs/vllm_9000_${RUN_ID}.log 2>&1 &
sleep 80
CUDA_VISIBLE_DEVICES=1 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --port 9001  > vllmlogs/vllm_9001_${RUN_ID}.log 2>&1 &
sleep 80
CUDA_VISIBLE_DEVICES=2 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --port 9002  > vllmlogs/vllm_9002_${RUN_ID}.log 2>&1 &
sleep 80
CUDA_VISIBLE_DEVICES=3 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --port 9003  > vllmlogs/vllm_9003_${RUN_ID}.log 2>&1 &
sleep 80

CUDA_VISIBLE_DEVICES=4 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --port 9004  > vllmlogs/vllm_9004_${RUN_ID}.log 2>&1 &
sleep 80
CUDA_VISIBLE_DEVICES=5 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --port 9005  > vllmlogs/vllm_9005_${RUN_ID}.log 2>&1 &
sleep 80
CUDA_VISIBLE_DEVICES=6 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --port 9006  > vllmlogs/vllm_9006_${RUN_ID}.log 2>&1 &
sleep 80
CUDA_VISIBLE_DEVICES=7 vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --port 9007  > vllmlogs/vllm_9007_${RUN_ID}.log 2>&1 &