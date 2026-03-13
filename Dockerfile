FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libopenblas-dev libhdf5-dev git curl && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Install PyTorch with CUDA first (big layer, cached separately)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Install requirements (without torch since it's already installed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

EXPOSE 10000

CMD ["gunicorn", "app:app", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "600", \
     "--preload", \
     "--bind", "0.0.0.0:10000"]
