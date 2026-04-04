#!/bin/bash
set -e

# ── SSH setup (RunPod injects PUBLIC_KEY env var) ──
if [[ -n "${PUBLIC_KEY:-}" ]]; then
    echo "Setting up SSH..."
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys

    # Generate host keys if missing
    [[ -f /etc/ssh/ssh_host_rsa_key ]]     || ssh-keygen -t rsa     -f /etc/ssh/ssh_host_rsa_key     -q -N ''
    [[ -f /etc/ssh/ssh_host_ecdsa_key ]]   || ssh-keygen -t ecdsa   -f /etc/ssh/ssh_host_ecdsa_key   -q -N ''
    [[ -f /etc/ssh/ssh_host_ed25519_key ]] || ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ''

    mkdir -p /run/sshd
    /usr/sbin/sshd
    echo "SSH ready."
else
    echo "PUBLIC_KEY not set — skipping SSH."
fi

# ── Export env vars so SSH sessions can access API keys ──
printenv | grep -E '^[A-Z_][A-Z0-9_]*=' \
    | grep -v '^PUBLIC_KEY=' \
    | awk -F= '{ val=$0; sub(/^[^=]*=/,"",val); print "export " $1 "=\"" val "\"" }' \
    > /etc/rp_environment 2>/dev/null || true
echo 'source /etc/rp_environment' >> ~/.bashrc 2>/dev/null || true

# ── Prevent OpenMP/MKL deadlocks under threaded gunicorn ──
# PySCF (libcint, libxc) and numpy spawn OpenMP threads internally.
# Under gunicorn --threads, multiple OpenMP pools fight Python threads → random freezes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Verify GPU is accessible at runtime ──
echo "Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected! torch.cuda.is_available() = False')
    print(f'PyTorch: {torch.__version__}, CUDA compiled: {torch.version.cuda}')
    print('Check: is NVIDIA runtime enabled? (--gpus all or nvidia container toolkit)')
"

# ── Start gunicorn (no --preload so CUDA inits in worker, not master) ──
echo "Starting ChopChopMol backend on :10000 ..."
exec gunicorn app:app \
    --workers 1 \
    --threads 8 \
    --timeout 600 \
    --bind 0.0.0.0:10000
