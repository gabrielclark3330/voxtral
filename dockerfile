#FROM nvcr.io/nvidia/pytorch:24.08-py3
FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3-pip \
        llvm-18-dev \
        clang-18 \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/0.7.21/install.sh | sh
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}"
RUN uv pip install --upgrade pip

RUN uv pip install -U "vllm[audio]" \
        --torch-backend=auto \
        --break-system-packages \
        --extra-index-url https://wheels.vllm.ai/nightly

RUN uv pip install --upgrade --break-system-packages mistral_common\[audio\]

RUN uv pip install --upgrade --force-reinstall "numpy<2.3"