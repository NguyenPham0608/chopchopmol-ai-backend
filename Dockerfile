FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps + SSH for RunPod web terminal
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libopenblas-dev libhdf5-dev git curl wget nano htop \
    openssh-server && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    mkdir -p /run/sshd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure SSH for RunPod (public key auth via PUBLIC_KEY env var)
RUN sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Install PyTorch with CUDA first (big cached layer)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Copy start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 10000 22

CMD ["/start.sh"]
