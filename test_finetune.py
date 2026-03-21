"""Run on RunPod to diagnose fine-tuning SIGSEGV. Usage: python test_finetune.py"""
import os
import subprocess
import sys

# Step 1: Find MACE training script
import mace
script = os.path.join(os.path.dirname(mace.__file__), "cli", "run_train.py")
if not os.path.exists(script):
    script = os.path.join(os.path.dirname(mace.__file__), "scripts", "run_train.py")
print("Train script:", script, "exists:", os.path.exists(script))

# Step 2: Get foundation model path
from mace.calculators.foundations_models import download_mace_mp_checkpoint
foundation = download_mace_mp_checkpoint("medium")
print("Foundation model:", foundation, "exists:", os.path.exists(foundation))

# Step 3: Create minimal training data
train_path = "/tmp/test_train.extxyz"
with open(train_path, "w") as f:
    for i in range(5):
        f.write(f"""3
Lattice="100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0" Properties=species:S:1:pos:R:3:forces:R:3 energy={-2078.0 + i*0.01} pbc="T T T"
O    0.0  0.0  0.0   0.1  -0.2  0.0
H    0.96 0.0  0.0  -0.05  0.1  0.0
H   -0.24 0.93 0.0  -0.05  0.1  0.0
""")
print("Training data written:", train_path)

# Step 4: Run training
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "N/A")

cmd = [
    sys.executable, script,
    "--name=test-finetune",
    f"--train_file={train_path}",
    "--valid_fraction=0.2",
    "--energy_key=energy",
    "--forces_key=forces",
    "--E0s=average",
    "--model=MACE",
    "--hidden_irreps=128x0e + 128x1o",
    "--r_max=4.0",
    "--num_interactions=2",
    "--batch_size=2",
    "--max_num_epochs=2",
    "--ema", "--ema_decay=0.99",
    "--amsgrad",
    f"--device={device}",
    "--default_dtype=float32",
    f"--foundation_model={foundation}",
    "--correlation=3",
    "--seed=42",
    "--checkpoints_dir=/tmp",
    "--results_dir=/tmp",
    "--work_dir=/tmp",
]

print("\nRunning:", " ".join(cmd[:5]), "...")
print("=" * 60)
result = subprocess.run(cmd, capture_output=False)
print("=" * 60)
print("Exit code:", result.returncode)
