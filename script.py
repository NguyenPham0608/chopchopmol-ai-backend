import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import numpy as np
import torch
import pyscf

parser = argparse.ArgumentParser(
    description="GPU-accelerated DFT energy and forces for extxyz frames"
)
parser.add_argument(
    "--xc", default="wb97m-d3bj", help="DFT functional (default: wb97m-d3bj)"
)
parser.add_argument(
    "--basis", default="def2-tzvp", help="Basis set (default: def2-tzvp)"
)
parser.add_argument("--input", default="cafein.extxyz", help="Input extxyz file")
parser.add_argument("--output", default="cafein_out.extxyz", help="Output extxyz file")
args = parser.parse_args()

# Auto-detect GPU vs CPU
_use_gpu = False
try:
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        from gpu4pyscf.dft import rks

        _use_gpu = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        from pyscf.dft import rks

        print("No GPU detected, using CPU PySCF")
except ImportError:
    from pyscf.dft import rks

    print("gpu4pyscf not installed, using CPU PySCF")


def read_extxyz_frames(filename):
    frames = []
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            natoms = int(line)
            comment = f.readline().strip()
            atoms = []
            for _ in range(natoms):
                parts = f.readline().split()
                atoms.append(
                    (parts[0], (float(parts[1]), float(parts[2]), float(parts[3])))
                )
            frames.append((natoms, comment, atoms))
    return frames


FLUSH_EVERY = 1  # flush to disk every N frames

frames = read_extxyz_frames(args.input)
print(f"Read {len(frames)} frames from {args.input}")
print(f"Settings: xc={args.xc}, basis={args.basis}, gpu={_use_gpu}")

prev_dm = None  # reuse density matrix across frames for faster convergence

with open(args.output, "w") as fout:
    for idx, (natoms, comment, atoms) in enumerate(frames):
        print(f"\n--- Frame {idx+1}/{len(frames)} ({natoms} atoms) ---")

        mol = pyscf.M(
            atom=atoms,
            basis=args.basis,
            charge=0,
            spin=0,
            verbose=1,
        )

        mf = rks.RKS(mol, xc=args.xc).density_fit()
        mf.conv_tol = 1e-8
        mf.max_cycle = 200
        mf.grids.atom_grid = (75, 302)

        # Warm-start from previous frame's density matrix
        if prev_dm is not None:
            energy = mf.kernel(dm0=prev_dm)
        else:
            energy = mf.kernel()

        # Retry with damping if not converged
        if not mf.converged:
            print(f"  SCF not converged, retrying with damping...")
            mf.damp = 0.5
            mf.max_cycle = 300
            energy = mf.kernel(dm0=mf.make_rdm1())

        prev_dm = mf.make_rdm1()

        g = mf.nuc_grad_method()
        gradient = np.asarray(g.kernel())
        forces = -gradient

        fout.write(f"{natoms}\n")
        fout.write(
            f'Properties="species:S:1:pos:R:3:forces:R:3" '
            f'energy={energy:.10f} charge={mol.charge} spin={mol.spin} pbc="T T T"\n'
        )
        for i, (sym, (x, y, z)) in enumerate(atoms):
            fx, fy, fz = forces[i]
            fout.write(
                f"{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f} {fx:16.8f} {fy:16.8f} {fz:16.8f}\n"
            )

        if (idx + 1) % FLUSH_EVERY == 0:
            fout.flush()
        print(f"Frame {idx+1} energy: {energy:.10f} Hartree (converged={mf.converged})")

print(f"\nAll frames done. Output written to {args.output}")
