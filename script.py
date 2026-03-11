import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

USE_GPU = False
try:
    import cupy
    cupy.cuda.runtime.getDeviceCount()
    from gpu4pyscf.dft import rks as gpu_rks
    USE_GPU = True
except Exception:
    pass

import numpy as np
import pyscf
from pyscf.dft import rks as cpu_rks

print(f'Running on: {"GPU" if USE_GPU else "CPU"}')


def read_extxyz(filename):
    frames = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            natoms = int(line.strip())
            comment = f.readline().strip()
            atoms = []
            for _ in range(natoms):
                parts = f.readline().split()
                atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
            frames.append((natoms, comment, atoms))
    return frames


def compute_frame(atoms, charge=0, spin=0):
    mol = pyscf.M(
        atom=atoms,
        basis='def2-tzvppd',
        charge=charge,
        spin=spin,
        verbose=0,
    )

    if USE_GPU:
        mf = gpu_rks.RKS(mol, xc='wb97m-d3bj').density_fit()
    else:
        mf = cpu_rks.RKS(mol, xc='wb97m-d3bj').density_fit()
    mf.conv_tol = 1e-10
    mf.grids.atom_grid = (99, 590)
    energy = mf.kernel()

    g = mf.nuc_grad_method()
    gradient = np.asarray(g.kernel())
    BOHR_TO_ANG = 0.529177210903
    forces = -gradient / BOHR_TO_ANG  # Hartree/Angstrom

    return mol, energy, forces


def write_frame(fout, natoms, atoms, energy, forces, charge, spin):
    fout.write(f'{natoms}\n')
    fout.write(
        f'Properties="species:S:1:pos:R:3:forces:R:3" '
        f'energy={energy:.10f} charge={charge} spin={spin} pbc="F F F"\n'
    )
    for i, (sym, (x, y, z)) in enumerate(atoms):
        fx, fy, fz = forces[i]
        fout.write(f'{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f} {fx:16.8f} {fy:16.8f} {fz:16.8f}\n')


frames = read_extxyz('cafein.extxyz')
print(f'Read {len(frames)} frames from cafein.extxyz')

charge = 0
spin = 0

with open('cafein_out.extxyz', 'w') as fout:
    for i, (natoms, comment, atoms) in enumerate(frames):
        mol, energy, forces = compute_frame(atoms, charge, spin)
        write_frame(fout, natoms, atoms, energy, forces, mol.charge, mol.spin)
        fout.flush()

        print(f'Frame {i+1}/{len(frames)}  E={energy:.10f} Hartree')

print(f'Done. {len(frames)} frames written to cafein_out.extxyz')
