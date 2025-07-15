FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3-pip \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/0.7.21/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_SYSTEM_PYTHON=1

RUN uv pip install -U "vllm[audio]" \
       --torch-backend=auto \
       --extra-index-url https://wheels.vllm.ai/nightly
