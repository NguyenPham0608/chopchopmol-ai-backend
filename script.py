import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import numpy as np
import pyscf
from gpu4pyscf.dft import rks

parser = argparse.ArgumentParser(description='GPU-accelerated DFT energy and forces for extxyz frames')
parser.add_argument('--xc', default='wb97m-d3bj', help='DFT functional (default: wb97m-d3bj)')
parser.add_argument('--basis', default='def2-tzvppd', help='Basis set (default: def2-tzvppd)')
args = parser.parse_args()


def read_extxyz_frames(filename):
    frames = []
    with open(filename, 'r') as f:
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
                atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
            frames.append((natoms, comment, atoms))
    return frames


FLUSH_EVERY = 1  # flush to disk every N frames

frames = read_extxyz_frames('cafein.extxyz')
print(f'Read {len(frames)} frames from cafein.extxyz')

with open('cafein_out.extxyz', 'w') as fout:
    for idx, (natoms, comment, atoms) in enumerate(frames):
        print(f'\n--- Frame {idx+1}/{len(frames)} ({natoms} atoms) ---')

        mol = pyscf.M(
            atom=atoms,
            basis=args.basis,
            charge=0,
            spin=0,
            verbose=1,
        )

        mf = rks.RKS(mol, xc=args.xc).density_fit()
        mf.conv_tol = 1e-10
        mf.grids.atom_grid = (99, 590)
        energy = mf.kernel()

        g = mf.nuc_grad_method()
        gradient = np.asarray(g.kernel())
        forces = -gradient

        fout.write(f'{natoms}\n')
        fout.write(
            f'Properties="species:S:1:pos:R:3:forces:R:3" '
            f'energy={energy:.10f} charge={mol.charge} spin={mol.spin} pbc="F F F"\n'
        )
        for i, (sym, (x, y, z)) in enumerate(atoms):
            fx, fy, fz = forces[i]
            fout.write(f'{sym:2s} {x:16.8f} {y:16.8f} {z:16.8f} {fx:16.8f} {fy:16.8f} {fz:16.8f}\n')

        if (idx + 1) % FLUSH_EVERY == 0:
            fout.flush()
        print(f'Frame {idx+1} energy: {energy:.10f} Hartree')

print('\nAll frames done. Output written to cafein_out.extxyz')
