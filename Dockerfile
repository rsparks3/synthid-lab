# DGX Spark (GB10, aarch64) image. The NGC PyTorch container ships an NVIDIA-built torch
# that already supports Blackwell sm_120/121, so we deliberately do NOT pip install torch here.
FROM nvcr.io/nvidia/pytorch:25.09-py3

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY synthid_lab ./synthid_lab
COPY static ./static
COPY scripts ./scripts
# data/ (keys, stories) is mounted at runtime so keys never bake into the image.

ENV SYNTHID_MODEL=Qwen/Qwen2.5-3B-Instruct \
    SYNTHID_NGRAM_LEN=3 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000
CMD ["uvicorn", "synthid_lab.server:app", "--host", "0.0.0.0", "--port", "8000"]
