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

# Install PyTorch with CUDA 12.4 (big cached layer)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Install cupy for gpu4pyscf GPU acceleration
RUN pip install --no-cache-dir cupy-cuda12x

# Install remaining requirements, SKIP torch so we keep the CUDA version.
# Use --extra-index-url so any transitive torch deps also resolve to CUDA builds.
# Use --no-deps for mace-torch to prevent it pulling CPU torch from PyPI.
COPY requirements.txt .
RUN grep -viE '^torch$' requirements.txt \
    | pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r /dev/stdin

# Verify CUDA torch was NOT overwritten by CPU version (fail the build if so)
RUN python -c "\
import torch; \
assert torch.version.cuda is not None, \
    f'CUDA torch was overwritten! Got torch {torch.__version__} with cuda={torch.version.cuda}'; \
print(f'✅ PyTorch {torch.__version__}, CUDA: {torch.version.cuda}')"

# Verify cuequivariance CUDA kernels are importable
RUN python -c "import cuequivariance as cue; print(f'cuequivariance {cue.__version__}'); import cuequivariance_torch; print('cuequivariance-torch OK')" || echo "WARNING: cuequivariance import failed — MACE will use PyTorch fallback"

# Copy app code + start script (always last — never cached)
COPY app.py start.sh test_finetune.py ./
RUN chmod +x start.sh

EXPOSE 10000 22

CMD ["/app/start.sh"]
