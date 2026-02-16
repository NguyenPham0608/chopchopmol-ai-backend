from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI
from anthropic import Anthropic
import os
import orjson
import tempfile
import numpy as np
from ase import Atoms
import base64
import hashlib
import zlib
from io import BytesIO
from time import time
import torch
import httpx

# Lazy-load MACE to avoid slow startup
_mace_calculators = {}

MACE_MODELS = {
    "mace-mp-0a": {
        "url": "medium",
        "name": "MACE-MP-0a",
        "description": "Original foundation model. Good general-purpose accuracy for materials.",
        "speed": "Fast",
        "best_for": "General materials, quick calculations",
    },
    "mace-mp-0b3": {
        "url": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
        "name": "MACE-MP-0b3",
        "description": "Improved high-pressure stability and better reference energies.",
        "speed": "Fast",
        "best_for": "High-pressure systems, better energy references",
    },
    "mace-mpa-0": {
        "url": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model",
        "name": "MACE-MPA-0",
        "description": "Best accuracy for materials. Trained on MPTrj + sAlex datasets.",
        "speed": "Medium",
        "best_for": "Highest accuracy, production calculations",
    },
}


def get_mace_calculator(model_id="mace-mp-0a"):
    global _mace_calculators
    if model_id not in _mace_calculators:
        from mace.calculators import mace_mp

        model_url = MACE_MODELS.get(model_id, MACE_MODELS["mace-mp-0a"])["url"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _mace_calculators[model_id] = mace_mp(
            model=model_url, default_dtype="float32", device=device
        )
    return _mace_calculators[model_id]


def dumps(obj):
    return orjson.dumps(obj).decode()


app = Flask(__name__)
CORS(
    app,
    resources={r"/ai/*": {"origins": "*"}, r"/api/*": {"origins": "*"}},
    supports_credentials=False,
)

# Store sessions in memory
sessions = {}  # {session_id: {"history": [], "last_access": timestamp}}
MAX_SESSIONS = 500
SESSION_TTL = 3600  # 1 hour

# Prompt cache for speed optimization
prompt_cache = {}  # {state_hash: prompt_string}
MAX_PROMPT_CACHE = 1000

# Molden molecule + AO grid cache
# Key: (content_hash, gridSize, padding)
# Value: dict with mol, mo_coeff, mo_energy, mo_occ, ao_values, grid_shape, grid_meta
molden_cache = {}
MAX_MOLDEN_CACHE = 10
MOLDEN_CACHE_TTL = 1800  # 30 minutes
BOHR_TO_ANG = 0.529177249


def get_or_create_molden_cache(molden_content, grid_size, padding):
    """Load and cache PySCF mol object + AO grid. The AO grid is the expensive
    step and is identical for ALL orbitals of the same molecule at the same
    grid resolution. Individual MO values are just ao_values @ mo_coeff[:, i]."""
    content_hash = hashlib.sha256(molden_content.encode()).hexdigest()[:16]
    cache_key = (content_hash, grid_size, padding)

    if cache_key in molden_cache:
        molden_cache[cache_key]["last_access"] = time()
        return molden_cache[cache_key]

    # Evict expired or overflow entries
    now = time()
    expired = [
        k for k, v in molden_cache.items() if now - v["last_access"] > MOLDEN_CACHE_TTL
    ]
    for k in expired:
        del molden_cache[k]
    if len(molden_cache) >= MAX_MOLDEN_CACHE:
        oldest = min(molden_cache, key=lambda k: molden_cache[k]["last_access"])
        del molden_cache[oldest]

    from pyscf.tools import molden as pyscf_molden

    with tempfile.NamedTemporaryFile(mode="w", suffix=".molden", delete=False) as tmp:
        tmp.write(molden_content)
        tmp_path = tmp.name

    try:
        mol, mo_energy, mo_coeff, mo_occ, irrep_labels, spins = pyscf_molden.load(
            tmp_path
        )
    finally:
        os.unlink(tmp_path)

    # Build 3D grid around the molecule
    coords = mol.atom_coords()  # Bohr
    x_min, y_min, z_min = coords.min(axis=0) - padding
    x_max, y_max, z_max = coords.max(axis=0) + padding

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    z = np.linspace(z_min, z_max, grid_size)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Evaluate ALL atomic orbitals on grid ONCE (the expensive step)
    ao_values = mol.eval_gto("GTOval_sph", grid_points)

    spacing = [
        (x[1] - x[0]) * BOHR_TO_ANG if len(x) > 1 else 1.0,
        (y[1] - y[0]) * BOHR_TO_ANG if len(y) > 1 else 1.0,
        (z[1] - z[0]) * BOHR_TO_ANG if len(z) > 1 else 1.0,
    ]

    n_mo = mo_coeff.shape[1]
    n_occ = int(sum(mo_occ) // 2)
    homo_index = n_occ - 1 if n_occ > 0 else -1
    lumo_index = n_occ if n_occ < n_mo else -1

    # Pre-compute torch tensors for GPU-accelerated batch MO computation
    try:
        _torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        ao_tensor = torch.from_numpy(ao_values.astype(np.float32)).to(_torch_device)
        coeff_tensor = torch.from_numpy(mo_coeff.astype(np.float32)).to(_torch_device)
    except Exception as e:
        print(f"⚠️ Torch tensor creation failed ({e}), falling back to numpy")
        _torch_device = "numpy"
        ao_tensor = None
        coeff_tensor = None

    entry = {
        "mol": mol,
        "mo_coeff": mo_coeff,
        "mo_energy": mo_energy,
        "mo_occ": mo_occ,
        "ao_values": ao_values,
        "ao_tensor": ao_tensor,
        "coeff_tensor": coeff_tensor,
        "torch_device": _torch_device,
        "grid_shape": X.shape,
        "grid_meta": {
            "origin": [x[0] * BOHR_TO_ANG, y[0] * BOHR_TO_ANG, z[0] * BOHR_TO_ANG],
            "spacing": spacing,
            "dimensions": [grid_size, grid_size, grid_size],
        },
        "n_mo": n_mo,
        "homo_index": homo_index,
        "lumo_index": lumo_index,
        "last_access": time(),
    }
    molden_cache[cache_key] = entry

    # Pre-compute HOMO/LUMO MO values in background so they're ready for batch requests
    import threading

    def _precompute_key_orbitals():
        try:
            key_indices = []
            if 0 <= homo_index < n_mo:
                key_indices.append(homo_index)
            if 0 <= lumo_index < n_mo:
                key_indices.append(lumo_index)
            if not key_indices:
                return

            if ao_tensor is not None:
                idx_t = torch.tensor(key_indices, dtype=torch.long)
                coeff_sub = coeff_tensor[:, idx_t]
                mo_vals = torch.matmul(ao_tensor, coeff_sub).cpu().numpy()
            else:
                coeff_sub = mo_coeff[:, key_indices]
                mo_vals = ao_values @ coeff_sub

            # Store pre-computed MO values in the cache entry
            entry["precomputed_mo"] = {
                key_indices[i]: mo_vals[:, i].copy() for i in range(len(key_indices))
            }
        except Exception as e:
            print(f"⚠️ Background HOMO/LUMO precompute failed: {e}")

    threading.Thread(target=_precompute_key_orbitals, daemon=True).start()

    return entry


def get_molden_cache_by_key(cache_key_str):
    """Resolve a cacheKey string (from /prepare) back to cached data.
    cacheKey format: '{content_hash}_{gridSize}_{padding}'
    """
    parts = cache_key_str.rsplit("_", 2)
    if len(parts) != 3:
        return None
    content_hash, grid_size_str, padding_str = parts
    try:
        key = (content_hash, int(grid_size_str), float(padding_str))
    except ValueError:
        return None
    entry = molden_cache.get(key)
    if entry:
        entry["last_access"] = time()
    return entry


def compute_orbital_type(orbital_index, homo_index, lumo_index):
    """Determine orbital type label (HOMO, LUMO, H-n, L+n)."""
    if orbital_index == homo_index:
        return "HOMO"
    elif orbital_index == lumo_index:
        return "LUMO"
    elif homo_index >= 0 and orbital_index < homo_index:
        return f"H-{homo_index - orbital_index}"
    elif lumo_index >= 0:
        return f"L+{orbital_index - lumo_index}"
    return f"MO{orbital_index + 1}"


def encode_orbital_binary(mo_values_flat, compress=True):
    """Encode orbital volumetric data as base64, optionally with zlib compression."""
    raw_bytes = mo_values_flat.astype(np.float32).tobytes()
    if compress:
        compressed = zlib.compress(raw_bytes, level=1)
        return base64.b64encode(compressed).decode(), "base64_float32_zlib"
    else:
        return base64.b64encode(raw_bytes).decode(), "base64_float32"


def encode_mesh_binary(vertices, normals, compress=True):
    """Encode mesh vertices and normals as base64, optionally with zlib compression."""
    verts_bytes = vertices.astype(np.float32).tobytes()
    norms_bytes = normals.astype(np.float32).tobytes()
    if compress:
        return (
            base64.b64encode(zlib.compress(verts_bytes, level=1)).decode(),
            base64.b64encode(zlib.compress(norms_bytes, level=1)).decode(),
        )
    else:
        return (
            base64.b64encode(verts_bytes).decode(),
            base64.b64encode(norms_bytes).decode(),
        )


def compute_orbital_mesh(mo_values_flat, grid_shape, grid_meta, isovalue=0.02):
    """Run marching cubes on orbital volume data, return mesh for positive and negative phases.

    Returns dict with 'positive' and 'negative' keys, each containing
    compressed base64 vertices/normals, or None if no surface found.
    """
    from skimage.measure import marching_cubes

    mo_3d = mo_values_flat.reshape(grid_shape)
    origin = np.array(grid_meta["origin"])
    spacing = np.array(grid_meta["spacing"])

    result = {"positive": None, "negative": None}

    for phase, iso in [("positive", isovalue), ("negative", -isovalue)]:
        try:
            verts, faces, norms, _ = marching_cubes(
                mo_3d, level=iso, spacing=tuple(spacing)
            )
        except (ValueError, RuntimeError):
            # No surface at this isovalue
            continue

        if len(verts) == 0:
            continue

        # Shift vertices to world coordinates
        verts = verts + origin

        # Flip normals for negative phase
        if phase == "negative":
            norms = -norms

        # Expand indexed faces to flat triangle list (3 verts per triangle)
        tri_verts = verts[faces.ravel()].astype(np.float32)
        tri_norms = norms[faces.ravel()].astype(np.float32)

        verts_b64, norms_b64 = encode_mesh_binary(tri_verts.ravel(), tri_norms.ravel())
        result[phase] = {
            "vertices": verts_b64,
            "normals": norms_b64,
            "numVertices": len(tri_verts),
            "numTriangles": len(faces),
        }

    return result


# Global OpenAI client - reuses TCP connection
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
claude_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Tools schema - exact copy from original buildToolsSchema()

TOOLS_JSON = """[
  {"type":"function","function":{"name":"get_molecule_info","description":"Get molecule overview: total atoms, element counts, bond count, selection. Use as first step to understand the loaded structure.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"get_atom_info","description":"Get position (x,y,z) and element for specific atoms. Returns coordinates in Angstroms.","parameters":{"type":"object","properties":{"indices":{"type":"array","items":{"type":"integer"},"description":"Atom indices (0-based)"}},"required":["indices"]}}},
  {"type":"function","function":{"name":"get_bonded_atoms","description":"Get bond connectivity for atoms. Returns bonded atom indices and elements. Essential for understanding topology before scans or edits.","parameters":{"type":"object","properties":{"indices":{"type":"array","items":{"type":"integer"},"description":"Atom indices to query. If omitted, uses current selection."}}}}},
  {"type":"function","function":{"name":"measure_distance","description":"Measure distance between 2 atoms in Angstroms. Returns distance_angstrom. Can specify indices directly (preferred) or use selection.","parameters":{"type":"object","properties":{"atom1":{"type":"integer","description":"First atom index (0-based)"},"atom2":{"type":"integer","description":"Second atom index (0-based)"}},"required":[]}}},
  {"type":"function","function":{"name":"measure_angle","description":"Measure angle formed by 3 atoms in degrees (atom2 is vertex). Returns angle_degrees. Can specify indices directly (preferred) or use selection.","parameters":{"type":"object","properties":{"atom1":{"type":"integer","description":"First atom (0-based)"},"atom2":{"type":"integer","description":"Vertex atom (0-based)"},"atom3":{"type":"integer","description":"Third atom (0-based)"}},"required":[]}}},
  {"type":"function","function":{"name":"measure_dihedral","description":"Measure dihedral/torsion angle between 4 atoms in degrees. Returns dihedral_degrees. Can specify indices directly (preferred) or use selection.","parameters":{"type":"object","properties":{"atom1":{"type":"integer","description":"First atom (0-based)"},"atom2":{"type":"integer","description":"Second atom (0-based)"},"atom3":{"type":"integer","description":"Third atom (0-based)"},"atom4":{"type":"integer","description":"Fourth atom (0-based)"}},"required":[]}}},
  {"type":"function","function":{"name":"get_cached_energies","description":"Retrieve cached energy results from last calculate_all_energies/optimize_geometry/run_md. Avoids recalculation. Follow with create_chart.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"read_file","description":"Read text content of a file in the open folder.","parameters":{"type":"object","properties":{"filename":{"type":"string","description":"Filename to read"}},"required":["filename"]}}},
  {"type":"function","function":{"name":"list_folder_files","description":"List all files in the currently open folder.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"web_search","description":"Search the web for chemistry info, properties, SMILES, safety data, reactions. Returns answer and source snippets. Use for anything you don't know.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"search_depth":{"type":"string","enum":["basic","advanced"],"description":"basic (fast) or advanced (thorough). Default: basic"},"max_results":{"type":"integer","description":"Results count (1-10, default: 5)"},"topic":{"type":"string","enum":["general","news"],"description":"Default: general"}},"required":["query"]}}},
  {"type":"function","function":{"name":"select_atoms","description":"Select atoms by 0-based indices. Sets context for edit tools (remove_atoms, change_atom_element, set_bond_distance, set_angle, set_dihedral_angle). Use add:true to extend.","parameters":{"type":"object","properties":{"indices":{"type":"array","items":{"type":"integer"},"description":"Atom indices to select"},"add":{"type":"boolean","description":"If true, add to current selection"}},"required":["indices"]}}},
  {"type":"function","function":{"name":"select_atoms_by_element","description":"Select all atoms of an element type (e.g. C, O, N). Faster than listing indices. Use add:true to combine with existing selection.","parameters":{"type":"object","properties":{"element":{"type":"string","description":"Element symbol"},"add":{"type":"boolean","description":"If true, add to current selection"}},"required":["element"]}}},
  {"type":"function","function":{"name":"select_all_atoms","description":"Select every atom in the molecule.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"select_connected","description":"Expand selection to include atoms directly bonded to currently selected atoms.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"clear_selection","description":"Clear all atom selections.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"add_atom","description":"Add atom at coordinates or bonded to selected atom. Use bondToSelected:true with 1 atom selected for automatic positioning.","parameters":{"type":"object","properties":{"element":{"type":"string","description":"Element symbol (e.g. C, H, O)"},"x":{"type":"number","description":"X coordinate (optional if bondToSelected)"},"y":{"type":"number","description":"Y coordinate"},"z":{"type":"number","description":"Z coordinate"},"bondToSelected":{"type":"boolean","description":"Bond to selected atom at typical bond length"}},"required":["element"]}}},
  {"type":"function","function":{"name":"remove_atoms","description":"Delete all currently selected atoms. Requires selection first.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"change_atom_element","description":"Change element of selected atoms. Requires selection. Use select_atoms first.","parameters":{"type":"object","properties":{"element":{"type":"string","description":"New element symbol"}},"required":["element"]}}},
  {"type":"function","function":{"name":"set_bond_distance","description":"Set exact distance between 2 selected atoms in Angstroms. Moves the smaller fragment. Requires exactly 2 atoms selected.","parameters":{"type":"object","properties":{"distance":{"type":"number","description":"Target distance in Angstroms"}},"required":["distance"]}}},
  {"type":"function","function":{"name":"set_angle","description":"Set bond angle for 3 selected atoms to exact value. B is vertex (A-B-C). Rotates fragment on A side. Requires exactly 3 atoms selected.","parameters":{"type":"object","properties":{"angle":{"type":"number","description":"Target angle in degrees (0-180)"}},"required":["angle"]}}},
  {"type":"function","function":{"name":"set_dihedral_angle","description":"Set dihedral for 4 selected atoms. Rotates fragment on D side around B-C axis (A-B-C-D). Requires exactly 4 atoms selected.","parameters":{"type":"object","properties":{"angle":{"type":"number","description":"Target dihedral in degrees (0-360)"}},"required":["angle"]}}},
  {"type":"function","function":{"name":"transform_atoms","description":"Rotate or translate atoms around an axis. Specify axisAtom1, axisAtom2, atomsToMove, and either angle (degrees) or distance (Angstroms). Use get_bonded_atoms or split_molecule to identify fragment indices.","parameters":{"type":"object","properties":{"axisAtom1":{"type":"integer","description":"First axis atom (0-based)"},"axisAtom2":{"type":"integer","description":"Second axis atom (0-based)"},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Atom indices to move (0-based)"},"angle":{"type":"number","description":"Rotation degrees (use this OR distance)"},"distance":{"type":"number","description":"Translation Angstroms (use this OR angle)"}},"required":["axisAtom1","axisAtom2","atomsToMove"]}}},
  {"type":"function","function":{"name":"split_molecule","description":"Split molecule by breaking bond. Returns fragment1[] and fragment2[] with atom indices. Use smaller fragment as atomsToMove for rotational_scan or translation_scan.","parameters":{"type":"object","properties":{"atom1":{"type":"integer","description":"First atom (0-based)"},"atom2":{"type":"integer","description":"Second atom (0-based, bonded to atom1)"}},"required":["atom1","atom2"]}}},
  {"type":"function","function":{"name":"add_hydrogens","description":"Add missing hydrogen atoms to entire molecule using standard valence rules.","parameters":{"type":"object","properties":{"pH":{"type":"number","description":"pH for protonation state (default 7.4)"}},"required":[]}}},
  {"type":"function","function":{"name":"rotational_scan","description":"Torsion scan: generate frames by rotating fragment around axis. Returns frameCount. Auto-picks smaller fragment. Not for rings. Follow with calculate_all_energies then create_chart.","parameters":{"type":"object","properties":{"axisAtom1":{"type":"integer","description":"First axis atom (0-based)"},"axisAtom2":{"type":"integer","description":"Second axis atom (0-based)"},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Override: atoms to rotate. If omitted, auto-picks smaller fragment."},"increment":{"type":"number","description":"Step size in degrees (default: 10)"},"startAngle":{"type":"number","description":"Start angle (default: 0)"},"endAngle":{"type":"number","description":"End angle (default: 360)"}},"required":["axisAtom1","axisAtom2"]}}},
  {"type":"function","function":{"name":"translation_scan","description":"Dissociation scan: translate fragment along axis in distance increments. Returns frameCount. Follow with calculate_all_energies then create_chart.","parameters":{"type":"object","properties":{"axisAtom1":{"type":"integer","description":"First axis atom (0-based)"},"axisAtom2":{"type":"integer","description":"Second axis atom (0-based)"},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Atoms to translate (0-based)"},"startDistance":{"type":"number","description":"Start distance in Angstroms (default: 0)"},"endDistance":{"type":"number","description":"End distance (default: 3)"},"increment":{"type":"number","description":"Step size in Angstroms (default: 0.2)"}},"required":["axisAtom1","axisAtom2","atomsToMove"]}}},
  {"type":"function","function":{"name":"angle_scan","description":"Angle scan: rotate fragment through angle range around pivot atom2 (A-B-C). Returns frameCount. Follow with calculate_all_energies then create_chart.","parameters":{"type":"object","properties":{"atom1":{"type":"integer","description":"First atom (0-based)"},"atom2":{"type":"integer","description":"Pivot atom (0-based)"},"atom3":{"type":"integer","description":"Third atom (0-based)"},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Atoms to rotate (0-based)"},"increment":{"type":"number","description":"Step degrees (default: 10)"},"startAngle":{"type":"number","description":"Start (default: 0)"},"endAngle":{"type":"number","description":"End (default: 360)"}},"required":["atom1","atom2","atom3"]}}},
  {"type":"function","function":{"name":"calculate_energy","description":"Single-point MACE energy for current geometry. Returns energy_eV. Set includeForces:true for per-atom forces (needed for toggle_force_arrows). Always specify model.","parameters":{"type":"object","properties":{"model":{"type":"string","enum":["mace-mp-0a","mace-mp-0b3","mace-mpa-0"],"description":"MACE model"},"includeForces":{"type":"boolean","description":"Include forces (default: false)"}},"required":["model"]}}},
  {"type":"function","function":{"name":"calculate_all_energies","description":"Batch MACE energy for all frames. Returns energies array with scanXValues for charting. Required after any scan. Follow with create_chart. Always specify model.","parameters":{"type":"object","properties":{"model":{"type":"string","enum":["mace-mp-0a","mace-mp-0b3","mace-mpa-0"],"description":"mace-mp-0a (fast), mace-mp-0b3 (high-P), mace-mpa-0 (accurate)"},"includeForces":{"type":"boolean","description":"Include forces (default: false)"}},"required":["model"]}}},
  {"type":"function","function":{"name":"optimize_geometry","description":"MACE geometry optimization. Returns converged, steps, energy_eV, trajectory. Follow with get_cached_energies and create_chart. Always specify model.","parameters":{"type":"object","properties":{"model":{"type":"string","enum":["small","medium","large","mace-mpa-0"],"description":"small (fast), medium (balanced), large (accurate), mace-mpa-0 (best)"},"fmax":{"type":"number","description":"Force threshold eV/A (default: 0.05)"},"maxSteps":{"type":"integer","description":"Max steps (default: 100)"},"includeForces":{"type":"boolean","description":"Include forces (default: false)"}},"required":["model"]}}},
  {"type":"function","function":{"name":"run_md","description":"MACE molecular dynamics (Langevin NVT). Returns trajectory frameCount. Follow with get_cached_energies and create_chart. Always specify model.","parameters":{"type":"object","properties":{"model":{"type":"string","enum":["small","medium","large","mace-mpa-0"],"description":"MACE model"},"temperature":{"type":"number","description":"Temp in K (default: 300)"},"steps":{"type":"integer","description":"MD steps (default: 500)"},"timestep":{"type":"number","description":"fs (default: 1.0)"},"friction":{"type":"number","description":"1/fs (default: 0.01)"},"saveInterval":{"type":"integer","description":"Save every N steps (default: 10)"},"includeForces":{"type":"boolean","description":"Include forces (default: false)"}},"required":["model"]}}},
  {"type":"function","function":{"name":"load_molecule","description":"Load molecule by name from PubChem (e.g. caffeine, aspirin). Follow with get_molecule_info to inspect.","parameters":{"type":"object","properties":{"name":{"type":"string","description":"Molecule name"}},"required":["name"]}}},
  {"type":"function","function":{"name":"create_chart","description":"Display line/bar/scatter chart from x and y arrays. Use scanXValues from calculate_all_energies for x-axis. Style params let you customize colors, line width, point size, fill, grid, legend, etc.","parameters":{"type":"object","properties":{"type":{"type":"string","enum":["line","bar","scatter"],"description":"Chart type (default: line)"},"title":{"type":"string","description":"Chart title"},"xLabel":{"type":"string","description":"X-axis label"},"yLabel":{"type":"string","description":"Y-axis label"},"x":{"type":"array","items":{"type":"number"},"description":"X values"},"y":{"type":"array","items":{"type":"number"},"description":"Y values"},"labels":{"type":"array","items":{"type":"string"},"description":"Series labels"},"lineColor":{"type":"string","description":"Line/border color as hex or CSS color (default: #667eea)"},"pointColor":{"type":"string","description":"Point fill color (default: same as lineColor)"},"highlightColor":{"type":"string","description":"Color for highlighted/current-frame point (default: #f093fb)"},"backgroundColor":{"type":"string","description":"Chart area background color (default: rgba(0,0,0,0.3))"},"lineWidth":{"type":"number","description":"Line thickness 1-6 (default: 2)"},"pointSize":{"type":"number","description":"Point radius 0-10 (default: 3)"},"tension":{"type":"number","description":"Curve smoothness 0-1 where 0=straight, 1=very smooth (default: 0.3)"},"fill":{"type":"boolean","description":"Fill area under line (default: false)"},"fillColor":{"type":"string","description":"Fill color with opacity e.g. rgba(102,126,234,0.2) (default: auto from lineColor)"},"showGrid":{"type":"boolean","description":"Show grid lines (default: true)"},"showLegend":{"type":"boolean","description":"Show chart legend (default: false)"},"showPoints":{"type":"boolean","description":"Show data points (default: true)"},"fontSize":{"type":"number","description":"Base font size for labels and ticks (default: 12)"},"datasets":{"type":"array","items":{"type":"object","properties":{"y":{"type":"array","items":{"type":"number"},"description":"Y values for this series"},"label":{"type":"string","description":"Series label"},"color":{"type":"string","description":"Line color for this series"}},"required":["y"]},"description":"Multiple data series (overrides y param). Each has y, label, color."}},"required":["x","y"]}}},
  {"type":"function","function":{"name":"save_file","description":"Export molecule to file (xyz, extxyz, mol, pdb, pqr, gro, mol2). Auto-includes forces/energies.","parameters":{"type":"object","properties":{"filename":{"type":"string","description":"Filename (default: auto)"},"format":{"type":"string","enum":["xyz","extxyz","mol","pdb","pqr","gro","mol2"],"description":"Format (default: xyz)"},"allFrames":{"type":"boolean","description":"All frames (default: true)"},"saveToLocal":{"type":"boolean","description":"Save to local folder vs download"}}}}},
  {"type":"function","function":{"name":"save_image","description":"Save screenshot of current 3D view as PNG.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"create_file","description":"Create a new file in the open folder.","parameters":{"type":"object","properties":{"filename":{"type":"string","description":"Filename with extension"},"content":{"type":"string","description":"File content"}},"required":["filename"]}}},
  {"type":"function","function":{"name":"edit_file","description":"Overwrite content of an AI-created file.","parameters":{"type":"object","properties":{"filename":{"type":"string","description":"Filename to edit"},"content":{"type":"string","description":"New content"}},"required":["filename","content"]}}},
  {"type":"function","function":{"name":"toggle_labels","description":"Show/hide atom labels. showElements for symbols (C, O), showIndices for numbers (0, 1), or both for combined (C0, O1).","parameters":{"type":"object","properties":{"showElements":{"type":"boolean","description":"Show element symbols"},"showIndices":{"type":"boolean","description":"Show atom indices"}},"required":[]}}},
  {"type":"function","function":{"name":"toggle_force_arrows","description":"Show/hide force vectors on atoms (green=low, red=high). Requires prior calculate_energy with includeForces:true.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"toggle_charge_visualization","description":"Show/hide charge coloring and labels (red=negative, blue=positive). Requires ORCA charge data.","parameters":{"type":"object","properties":{"showColors":{"type":"boolean","description":"Charge-based coloring"},"showLabels":{"type":"boolean","description":"Charge value labels"},"chargeType":{"type":"string","enum":["mulliken","loewdin"],"description":"Charge type (default: mulliken)"}},"required":[]}}},
  {"type":"function","function":{"name":"toggle_ribbon","description":"Toggle protein backbone ribbon visualization.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"set_style","description":"Change visual style: roughness, metalness, opacity, atomSize, backgroundColor.","parameters":{"type":"object","properties":{"roughness":{"type":"number","description":"0-1"},"metalness":{"type":"number","description":"0-1"},"opacity":{"type":"number","description":"0-1 (1=solid)"},"atomSize":{"type":"number","description":"0.1-3"},"backgroundColor":{"type":"string","description":"Hex color"}}}}},
  {"type":"function","function":{"name":"show_all_bond_lengths","description":"Display bond length labels on every bond.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"remove_bond_label","description":"Remove specific bond label (atom1+atom2) or all labels (all:true).","parameters":{"type":"object","properties":{"atom1":{"type":"integer","description":"First atom"},"atom2":{"type":"integer","description":"Second atom"},"all":{"type":"boolean","description":"Remove all labels"}},"required":[]}}},
  {"type":"function","function":{"name":"clear_measurements","description":"Remove all distance/angle/dihedral measurement labels.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"reset_camera","description":"Reset camera to default view.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"zoom_to_fit","description":"Zoom camera to fit entire molecule.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"rotate_camera","description":"Rotate camera view by angle degrees.","parameters":{"type":"object","properties":{"angle":{"type":"number","description":"Degrees"}},"required":["angle"]}}},
  {"type":"function","function":{"name":"define_axis","description":"Define rotation/translation axis from 2 selected atoms. Required before manual transform_atoms.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"remove_axis","description":"Remove the defined axis.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"create_fragment","description":"Group selected atoms into a named fragment.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"isolate_selection","description":"Isolate selected atoms to view/edit separately.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"undo","description":"Undo last action.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"redo","description":"Redo last undone action.","parameters":{"type":"object","properties":{}}}}
]"""

TOOLS = orjson.loads(TOOLS_JSON)

# Pre-compute Claude tools schema (avoid recomputing on every request)
CLAUDE_TOOLS = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS
]


def hash_state(state):
    """Create a hash of the state for caching"""
    import hashlib

    # Only hash the parts that affect the prompt
    key_parts = [
        state.get("hasAtoms", False),
        state.get("atomCount", 0),
        state.get("selectedCount", 0),
        tuple(state.get("selectedIndices", [])),
        len(state.get("fragments", [])),
        state.get("hasAxis", False),
        tuple(state.get("axisAtoms", [])) if state.get("hasAxis") else (),
        state.get("frameCount", 0),
        state.get("currentFrame", 0),
        state.get("hasEnergies", False),
        state.get("hasForces", False),
        state.get("hasMaceCache", False),
        state.get("maceFrameCount", 0),
        state.get("currentFileName", ""),
    ]
    return hashlib.md5(str(key_parts).encode()).hexdigest()


USE_LAYER_PROMPT = True


def build_system_prompt_legacy(state):
    """Original prompt — kept as fallback. Set USE_LAYER_PROMPT=False to use."""
    return f"""ChopChopMol AI. Execute immediately.

STATE: Atoms={state.get('atomCount', 0) if state.get('hasAtoms') else 0}, Selected={state.get('selectedCount', 0)}{' '+str(state.get('selectedIndices', [])) if state.get('selectedCount', 0) > 0 else ''}, Axis={'atoms '+str(state.get('axisAtoms', [])[0])+'-'+str(state.get('axisAtoms', [])[1]) if state.get('hasAxis') and len(state.get('axisAtoms', [])) == 2 else 'None'}, Frames={state.get('frameCount', 0)}, Energies={'Y' if state.get('hasEnergies') else 'N'}, Forces={'Y' if state.get('hasForces') else 'N'}, MACE={'Y('+str(state.get('maceFrameCount', 0))+')' if state.get('hasMaceCache') else 'N'}

CRITICAL: User uses 1-based indices, you use 0-based. "atom 5" → index 4.

TORSION SCAN (bond X-Y):
1. rotational_scan(axisAtom1=X-1, axisAtom2=Y-1, increment=10) - auto-picks smaller fragment
2. calculate_all_energies(model) - ASK user for model first
3. create_chart(x=[0,10,...,350], y=energies, xLabel="Angle (deg)", yLabel="Energy (kcal/mol)")

GEO OPTIMIZATION:
1. optimize_geometry(model) - ASK user
2. get_cached_energies()
3. create_chart

RULES:
1. Min tool calls. 2. ALWAYS ask user for MACE model unless specified.
3. Brief responses. 4. Use cached energies if available.
5. Use web_search for unknown facts. Answer known facts directly.
"""


def build_system_prompt(state):
    if not USE_LAYER_PROMPT:
        return build_system_prompt_legacy(state)

    return f"""ChopChopMol AI — molecular visualization and computation assistant.

STATE: Atoms={state.get('atomCount', 0) if state.get('hasAtoms') else 0}, Selected={state.get('selectedCount', 0)}{' '+str(state.get('selectedIndices', [])) if state.get('selectedCount', 0) > 0 else ''}, Axis={'atoms '+str(state.get('axisAtoms', [])[0])+'-'+str(state.get('axisAtoms', [])[1]) if state.get('hasAxis') and len(state.get('axisAtoms', [])) == 2 else 'None'}, Frames={state.get('frameCount', 0)}, CachedEnergies={'Y('+str(state.get('maceFrameCount', 0))+')' if state.get('hasMaceCache') else 'N'}

TOOL LAYERS (compose bottom-up):
L1 QUERY: get_molecule_info, get_atom_info, get_bonded_atoms, measure_distance, measure_angle, measure_dihedral, get_cached_energies, web_search (read-only, no side effects)
L2 SELECT: select_atoms, select_atoms_by_element, select_all_atoms, select_connected, clear_selection (set context for L3)
L3 EDIT: add_atom, remove_atoms, change_atom_element, set_bond_distance, set_angle, set_dihedral_angle, transform_atoms, split_molecule (modify molecule, most require selection)
L4 GENERATE: rotational_scan, translation_scan, angle_scan, calculate_energy, calculate_all_energies, optimize_geometry, run_md, load_molecule (create frames/data)
L5 OUTPUT: create_chart, save_file, save_image, create_file, edit_file (present results)
L6 VIEW: toggle_labels, toggle_force_arrows, toggle_charge_visualization, set_style, camera, undo, redo (non-destructive)

RULES:
1. Atom indices: 0-based.
2. ALWAYS ask user for MACE model (mace-mp-0a, mace-mp-0b3, mace-mpa-0) before energy/optimization/MD unless already specified.
3. Tool results include nextSteps hints — follow them for multi-step workflows.
4. If CachedEnergies=Y, use get_cached_energies instead of recalculating.
5. Brief responses (1-2 sentences). Execute tools immediately.
6. Measurement tools accept atom indices directly — no need to select first.
7. For unknown chemistry facts, use web_search. For known facts, answer directly.
"""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


def convert_to_claude_messages(history):
    claude_msgs = []
    current_user_content = []

    for msg in history:
        role = msg["role"]

        # 1. Before adding a User or Assistant message, FLUSH pending tool results
        if role in ["user", "assistant"] and current_user_content:
            claude_msgs.append({"role": "user", "content": current_user_content})
            current_user_content = []

        if role == "user":
            claude_msgs.append({"role": "user", "content": msg["content"]})

        elif role == "assistant":
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    # FIX: Handle empty argument strings safely
                    args_str = tc["function"]["arguments"]
                    if not args_str:
                        tool_input = {}
                    else:
                        try:
                            tool_input = orjson.loads(args_str)
                        except:
                            tool_input = {}  # Fallback if JSON is malformed

                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": tool_input,
                        }
                    )
            claude_msgs.append({"role": "assistant", "content": content})
        elif role == "tool":
            current_user_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
            )

    # 2. Flush any remaining tool results at the end
    if current_user_content:
        claude_msgs.append({"role": "user", "content": current_user_content})

    return claude_msgs


def repair_claude_history_for_tool_pairing(history):
    """
    Ensures that every assistant message with tool_use is immediately followed
    by corresponding tool_result blocks in the Claude message format.
    Inserts missing tool_result blocks if necessary.
    """
    repaired = []
    i = 0
    while i < len(history):
        msg = history[i]
        repaired.append(msg)

        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_ids = {tc["id"] for tc in msg["tool_calls"]}

            # Look ahead for tool messages
            provided_ids = set()
            j = i + 1
            while j < len(history) and history[j].get("role") == "tool":
                provided_ids.add(history[j].get("tool_call_id"))
                j += 1

            # Calculate missing IDs
            missing = tool_ids - provided_ids

            # CRITICAL FIX: If we are at the end of history and have tools,
            # we MUST provide results (dummies) or Claude throws 400.
            # This happens if the user prompt caused a tool call but the frontend
            # hasn't executed it yet, or if history slicing cut off the results.
            if missing:
                for mid in missing:
                    repaired.append(
                        {
                            "role": "tool",
                            "tool_call_id": mid,
                            "content": "Tool result missing (placeholder to satisfy API pairing requirement).",
                        }
                    )

        i += 1
    return repaired


@app.route("/ai/chat/stream", methods=["POST"])
def chat_stream():
    import time

    t_request = time.time()

    data = request.json
    session_id = data.get("sessionId", "default")
    user_message = data.get("message", "")
    state = data.get("state", {})
    tool_results = data.get("toolResults")
    model = data.get("model", "gpt-5-mini")

    print(
        f"📥 Request received: {len(user_message)} chars, state: {len(str(state))} chars",
        flush=True,
    )

    if not client.api_key:
        return jsonify({"error": "API key not set"}), 500

    # Cleanup old sessions periodically
    now = time.time()
    if len(sessions) > MAX_SESSIONS or len(sessions) % 100 == 0:
        expired = [
            sid for sid, s in sessions.items() if now - s["last_access"] > SESSION_TTL
        ]
        for sid in expired:
            del sessions[sid]

    if session_id not in sessions:
        sessions[session_id] = {"history": [], "last_access": now}
    else:
        sessions[session_id]["last_access"] = now

    conversationHistory = sessions[session_id]["history"]

    if tool_results is None:
        conversationHistory.append({"role": "user", "content": user_message})
    else:
        # Append actual tool results
        for result in tool_results["results"]:
            conversationHistory.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                }
            )

            # === UPDATED SAFETY RECONSTRUCTION ===
        # Ensure the immediate previous message to the tool results is an assistant with matching tool_calls.
        # If not (e.g., due to history corruption, truncation, or ordering issues), reconstruct it.
        if conversationHistory and conversationHistory[-1]["role"] == "tool":
            # Count consecutive tool messages at the end (these are the ones just appended)
            num_tools = 0
            for i in range(len(conversationHistory) - 1, -1, -1):
                if conversationHistory[i]["role"] == "tool":
                    num_tools += 1
                else:
                    break
            first_tool_idx = len(conversationHistory) - num_tools

            reconstruct = True
            if first_tool_idx > 0:
                prev_msg = conversationHistory[first_tool_idx - 1]
                if prev_msg["role"] == "assistant" and prev_msg.get("tool_calls"):
                    assistant_tc_ids = {tc["id"] for tc in prev_msg["tool_calls"]}
                    tool_tc_ids = {
                        conversationHistory[j]["tool_call_id"]
                        for j in range(first_tool_idx, len(conversationHistory))
                    }
                    if tool_tc_ids.issubset(assistant_tc_ids):
                        reconstruct = False  # Already correctly paired

            if reconstruct:
                # Collect tool_call_ids in the exact order of the tool messages
                tool_ids = [
                    conversationHistory[j]["tool_call_id"]
                    for j in range(first_tool_idx, len(conversationHistory))
                ]
                reconstructed_tool_calls = [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": "reconstructed_tool", "arguments": "{}"},
                    }
                    for tc_id in tool_ids
                ]
                reconstructed_assistant = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": reconstructed_tool_calls,
                }
                # Insert immediately before the tools
                conversationHistory.insert(first_tool_idx, reconstructed_assistant)
    t_prompt = time.time()

    # Use cached prompt if available
    state_hash = hash_state(state)
    if state_hash in prompt_cache:
        systemPrompt = prompt_cache[state_hash]
        print(f"⚡ Using cached prompt (hash: {state_hash[:8]})", flush=True)
    else:
        systemPrompt = build_system_prompt(state)
        prompt_cache[state_hash] = systemPrompt
        # Limit cache size
        if len(prompt_cache) > MAX_PROMPT_CACHE:
            prompt_cache.pop(next(iter(prompt_cache)))
        print(
            f"⏱️ Prompt built: {(time.time() - t_prompt) * 1000:.0f}ms, length: {len(systemPrompt)} chars",
            flush=True,
        )

    # === ROBUST HISTORY PREPARATION WITH PAIRING GUARANTEE ===
    # Take recent history
    max_history_messages = 50  # Increased for safety in multi-turn tool flows
    history_slice = conversationHistory[-max_history_messages:]

    # Remove leading orphaned tool messages (shouldn't happen, but safe)
    while history_slice and history_slice[0].get("role") == "tool":
        history_slice = history_slice[1:]

    # === OPTIMIZED: Only repair if there are actually tool_calls in history ===
    # Check if repair is needed (saves ~10-20ms per request when no tools)
    has_tool_calls = any(
        msg.get("role") == "assistant" and msg.get("tool_calls")
        for msg in history_slice
    )

    if has_tool_calls:
        i = 0
        while i < len(history_slice) - 1:
            msg = history_slice[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                assistant_tc_ids = {tc["id"] for tc in msg["tool_calls"]}
                # Look at all following messages until next assistant/user
                following_tool_ids = set()
                j = i + 1
                while j < len(history_slice) and history_slice[j].get("role") == "tool":
                    following_tool_ids.add(history_slice[j].get("tool_call_id"))
                    j += 1
                missing_ids = assistant_tc_ids - following_tool_ids
                if missing_ids:
                    # Reconstruct missing tool responses as errors (or empty) to satisfy validation
                    for mid in missing_ids:
                        history_slice.insert(
                            j,
                            {
                                "role": "tool",
                                "tool_call_id": mid,
                                "content": "Error: Tool execution result missing (possibly from previous session). Assuming success for continuation.",
                            },
                        )
            i += 1
        print(f"📜 History repaired: {len(history_slice)} messages", flush=True)
    else:
        print(f"📜 History clean (no tools): {len(history_slice)} messages", flush=True)

    messages = [{"role": "system", "content": systemPrompt}] + history_slice

    # Safe token estimation (fix for None content)
    total_tokens_est = sum(len(str(m.get("content") or "")) // 4 for m in messages)
    print(
        f"📊 Messages: {len(messages)}, estimated tokens: {total_tokens_est}",
        flush=True,
    )

    def generate():
        t0 = time.time()
        print(f"🚀 Starting OpenAI call...", flush=True)
        try:
            is_claude = "claude" in model.lower()

            if is_claude:
                if not claude_client:
                    raise ValueError("ANTHROPIC_API_KEY not set")
                # Use pre-computed Claude tools schema (cached at startup)
                # Repair history specifically for Claude's strict pairing requirement
                repaired_history = repair_claude_history_for_tool_pairing(history_slice)
                claude_messages = convert_to_claude_messages(repaired_history)
                call_params = {
                    "model": model,
                    "max_tokens": 4096,  # Reduced from 8192 for faster responses
                    "messages": claude_messages,
                    "system": [
                        {
                            "type": "text",
                            "text": systemPrompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "tools": CLAUDE_TOOLS if TOOLS else None,
                    "stream": True,
                }
            else:
                # Original OpenAI params
                call_params = {
                    "model": model,
                    "messages": messages,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "stream": True,
                }
                if model.startswith("gpt-5"):
                    call_params["max_completion_tokens"] = 16384
                    call_params["reasoning_effort"] = "low"
                    call_params["verbosity"] = "low"
                else:
                    call_params["max_tokens"] = 16384

            # Create stream
            if is_claude:
                stream = claude_client.messages.create(**call_params)
            else:
                stream = client.chat.completions.create(**call_params)

            print(f"📡 Stream created: {(time.time() - t0) * 1000:.0f}ms", flush=True)

            collected_content = ""
            tool_calls_data = {}
            first_chunk = True
            current_block_index = None

            for chunk in stream:
                if first_chunk:
                    print(
                        f"🔥 OpenAI/Claude TTFT: {(time.time() - t0) * 1000:.0f}ms",
                        flush=True,
                    )
                    first_chunk = False

                if is_claude:
                    if chunk.type == "content_block_start":
                        current_block_index = chunk.index
                        if chunk.content_block.type == "text":
                            tool_calls_data[current_block_index] = {
                                "type": "text",
                                "text": "",
                            }
                        elif chunk.content_block.type == "tool_use":
                            tool_calls_data[current_block_index] = {
                                "id": chunk.content_block.id,
                                "name": chunk.content_block.name,
                                "arguments": "",
                            }
                            # Send immediate tool status for Claude
                            yield f"data: {dumps({'type': 'tool_status', 'toolName': chunk.content_block.name})}\n\n"
                    elif chunk.type == "content_block_delta":
                        if chunk.delta.type == "text_delta":
                            text = chunk.delta.text
                            collected_content += text
                            yield f"data: {dumps({'type': 'text', 'content': text})}\n\n"
                        elif chunk.delta.type == "input_json_delta":
                            partial_json = chunk.delta.partial_json
                            if current_block_index in tool_calls_data:
                                tool_calls_data[current_block_index][
                                    "arguments"
                                ] += partial_json
                    elif chunk.type == "ping":
                        continue
                else:
                    # Original OpenAI chunk processing
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                    if delta.content:
                        collected_content += delta.content
                        yield f"data: {dumps({'type': 'text', 'content': delta.content})}\n\n"
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls_data:
                                tool_calls_data[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc.id:
                                tool_calls_data[idx]["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_data[idx]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls_data[idx][
                                        "arguments"
                                    ] += tc.function.arguments

            print(
                f"✅ Stream complete: {len(collected_content)} chars, total: {(time.time() - t0) * 1000:.0f}ms",
                flush=True,
            )

            # Post-stream processing (shared)
            if tool_calls_data:
                tool_calls = []
                for tc in tool_calls_data.values():
                    if "id" in tc:
                        tool_calls.append(
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                        )
                assistant_msg = {
                    "role": "assistant",
                    "content": collected_content,
                    "tool_calls": tool_calls,  # Adapted to OpenAI-style for history
                }
                conversationHistory.append(assistant_msg)
                yield f"data: {dumps({'type': 'tool_calls', 'toolCalls': tool_calls, 'assistantMessage': assistant_msg, 'sessionId': session_id})}\n\n"
            else:
                if collected_content:
                    conversationHistory.append(
                        {"role": "assistant", "content": collected_content}
                    )
                yield f"data: {dumps({'type': 'done', 'sessionId': session_id})}\n\n"

        except Exception as e:
            yield f"data: {dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


@app.route("/ai/mace/energy", methods=["POST"])
def calculate_energy():
    """Calculate single-point energy using MACE-MP"""
    data = request.json
    atoms_data = data.get("atoms", [])
    model_id = data.get("model", "mace-mp-0a")
    include_forces = data.get("includeForces", False)

    if not atoms_data:
        return jsonify({"error": "No atoms provided"}), 400

    try:
        symbols = [a["element"] for a in atoms_data]
        positions = [[a["x"], a["y"], a["z"]] for a in atoms_data]

        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = get_mace_calculator(model_id)

        energy = float(atoms.get_potential_energy())

        result = {
            "success": True,
            "energy_eV": energy,
            "energy_kcal": energy * 23.0609,  # eV to kcal/mol
        }

        if include_forces:
            forces = atoms.get_forces().tolist()
            result["forces"] = forces
            result["max_force"] = float(np.max(np.linalg.norm(forces, axis=1)))

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ai/mace/test", methods=["GET"])
def test_mace():
    """Test if MACE loads"""
    import traceback

    try:
        calc = get_mace_calculator()
        return jsonify({"success": True, "message": "MACE loaded"})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/ai/mace/optimize", methods=["POST"])
def optimize_geometry():
    """Geometry optimization using MACE-MP + ASE BFGS"""
    from ase import Atoms
    from ase.optimize import BFGS
    from mace.calculators import mace_mp
    import traceback
    import numpy as np

    data = request.json
    atoms_data = data.get("atoms", [])
    fmax = data.get("fmax", 0.05)
    max_steps = data.get("maxSteps", 100)
    include_forces = data.get("includeForces", False)

    if not atoms_data:
        return jsonify({"error": "No atoms provided"}), 400

    try:
        symbols = [a["element"] for a in atoms_data]
        positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms_data])

        # Create atoms object with a large enough cell for molecules
        pos_min = positions.min(axis=0) if len(positions) > 0 else np.array([0, 0, 0])
        pos_max = (
            positions.max(axis=0) if len(positions) > 0 else np.array([10, 10, 10])
        )
        cell_size = np.maximum(pos_max - pos_min + 20.0, 30.0)

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell_size, pbc=False)

        # Get model name and initialize MACE calculator
        model_name = data.get("model", "medium")

        model_map = {
            "small": "small",
            "medium": "medium",
            "large": "large",
            "mace-mpa-0": "medium",
        }

        mace_model = model_map.get(model_name, "medium")
        _opt_device = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_mp(model=mace_model, device=_opt_device, default_dtype="float64")
        atoms.calc = calc

        # Store trajectory frames
        trajectory_frames = []

        def observer():
            """Callback to capture each optimization step"""
            pos = atoms.get_positions().copy()
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            max_force = float(np.sqrt((forces**2).sum(axis=1).max()))

            frame_data = {
                "positions": pos.tolist(),
                "energy_eV": energy,
                "max_force": max_force,
            }
            if include_forces:
                frame_data["forces"] = forces.tolist()

            trajectory_frames.append(frame_data)

        # Run optimization with observer to capture frames
        opt = BFGS(atoms, logfile=None, trajectory=None, restart=None)
        opt.attach(observer, interval=1)  # Call observer after each step
        opt.run(fmax=fmax, steps=max_steps)

        # Final convergence check
        forces = atoms.get_forces()
        max_force = np.sqrt((forces**2).sum(axis=1).max())
        converged = bool(max_force < fmax)

        # Get final positions and energy
        final_positions = atoms.get_positions().tolist()
        final_energy = float(atoms.get_potential_energy())

        result = {
            "success": True,
            "converged": converged,
            "steps": int(opt.nsteps),
            "energy_eV": final_energy,
            "energy_kcal": final_energy * 23.0609,
            "max_force": float(max_force),
            "positions": [
                {"index": i, "x": p[0], "y": p[1], "z": p[2]}
                for i, p in enumerate(final_positions)
            ],
            "trajectory": trajectory_frames,  # All intermediate frames
        }

        if include_forces:
            result["forces"] = forces.tolist()

        return jsonify(result)
    except Exception as e:
        print(f"❌ MACE Optimization Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/ai/mace/energy-batch", methods=["POST"])
def calculate_energy_batch():
    data = request.json
    frames_data = data.get("frames", [])

    if not frames_data:
        return jsonify({"error": "No frames provided"}), 400

    try:
        model_id = data.get("model", "mace-mp-0a")
        calc = get_mace_calculator(model_id)
        results = []

        # Create atoms object once from first frame
        first_frame = frames_data[0]
        symbols = [a["element"] for a in first_frame]
        positions = np.array(
            [[a["x"], a["y"], a["z"]] for a in first_frame], dtype=np.float32
        )
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc

        for i, atoms_data in enumerate(frames_data):
            positions = np.array(
                [[a["x"], a["y"], a["z"]] for a in atoms_data], dtype=np.float32
            )
            atoms.set_positions(positions)

            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))

            frame_result = {
                "frame": i,
                "energy_eV": round(energy, 6),
                "energy_kcal": round(energy * 23.0609, 4),
                "max_force_eV_A": round(max_force, 6),
            }
            if data.get("includeForces", False):
                frame_result["forces"] = forces.tolist()
            results.append(frame_result)

        energies = [r["energy_eV"] for r in results]
        min_idx = int(np.argmin(energies))
        max_idx = int(np.argmax(energies))

        return jsonify(
            {
                "success": True,
                "frameCount": len(results),
                "energies": results,
                "lowestEnergyFrame": min_idx,
                "highestEnergyFrame": max_idx,
                "energyRange_eV": round(max(energies) - min(energies), 6),
            }
        )
    except Exception as e:
        print(f"MACE batch error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/ai/mace/md", methods=["POST"])
def run_molecular_dynamics():
    """Run NVT molecular dynamics using MACE-MP + ASE Langevin"""
    from ase import Atoms, units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from mace.calculators import mace_mp
    import traceback
    import numpy as np

    data = request.json
    atoms_data = data.get("atoms", [])
    temperature_K = data.get("temperature", 300)  # Kelvin
    timestep_fs = data.get("timestep", 1.0)  # femtoseconds
    n_steps = data.get("steps", 500)
    friction = data.get("friction", 0.01)  # 1/fs
    save_interval = data.get("saveInterval", 10)  # Save every N steps
    include_forces = data.get("includeForces", False)

    if not atoms_data:
        return jsonify({"error": "No atoms provided"}), 400

    try:
        symbols = [a["element"] for a in atoms_data]
        positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms_data])

        # Create atoms object
        pos_min = positions.min(axis=0) if len(positions) > 0 else np.array([0, 0, 0])
        pos_max = (
            positions.max(axis=0) if len(positions) > 0 else np.array([10, 10, 10])
        )
        cell_size = np.maximum(pos_max - pos_min + 20.0, 30.0)

        atoms = Atoms(symbols=symbols, positions=positions, cell=cell_size, pbc=False)

        # Initialize MACE calculator
        model_name = data.get("model", "medium")
        model_map = {
            "small": "small",
            "medium": "medium",
            "large": "large",
            "mace-mpa-0": "medium",
        }
        mace_model = model_map.get(model_name, "medium")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        calc = mace_mp(model=mace_model, device=device, default_dtype="float64")
        atoms.calc = calc

        # Initialize velocities from Maxwell-Boltzmann distribution
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

        # Store trajectory frames
        trajectory_frames = []

        def observer():
            """Capture frame at each save interval"""
            pos = atoms.get_positions().copy()
            vel = atoms.get_velocities()
            energy = float(atoms.get_potential_energy())
            kinetic = float(atoms.get_kinetic_energy())
            temp = float(kinetic / (1.5 * len(atoms) * units.kB))

            frame_data = {
                "positions": pos.tolist(),
                "energy_eV": energy,
                "kinetic_eV": kinetic,
                "total_eV": energy + kinetic,
                "temperature_K": temp,
                "step": len(trajectory_frames) * save_interval,
            }

            if include_forces:
                forces = atoms.get_forces()
                frame_data["forces"] = forces.tolist()
                frame_data["max_force"] = float(np.max(np.linalg.norm(forces, axis=1)))

            trajectory_frames.append(frame_data)

        # Set up Langevin dynamics (NVT)
        dyn = Langevin(
            atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=temperature_K,
            friction=friction / units.fs,
        )

        # Attach observer
        dyn.attach(observer, interval=save_interval)

        # Capture initial frame
        observer()

        # Run MD
        dyn.run(n_steps)

        # Final state
        final_positions = atoms.get_positions().tolist()
        final_energy = float(atoms.get_potential_energy())

        result = {
            "success": True,
            "steps": n_steps,
            "temperature_K": temperature_K,
            "timestep_fs": timestep_fs,
            "energy_eV": final_energy,
            "energy_kcal": final_energy * 23.0609,
            "frameCount": len(trajectory_frames),
            "positions": [
                {"index": i, "x": p[0], "y": p[1], "z": p[2]}
                for i, p in enumerate(final_positions)
            ],
            "trajectory": trajectory_frames,
        }

        if include_forces:
            forces = atoms.get_forces()
            result["forces"] = forces.tolist()
            result["max_force"] = float(np.max(np.linalg.norm(forces, axis=1)))

        return jsonify(result)

    except Exception as e:
        print(f"❌ MACE MD Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/ai/chart", methods=["POST"])
def generate_chart():
    """Generate a chart image from data"""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    data = request.json
    chart_type = data.get("type", "line")
    title = data.get("title", "")
    x_label = data.get("xLabel", "")
    y_label = data.get("yLabel", "")
    x_values = data.get("x", [])
    y_values = data.get("y", [])
    labels = data.get("labels", None)  # For multiple series

    try:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        # Style for dark theme
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

        colors = ["#667eea", "#f093fb", "#4fd1c5", "#f6ad55", "#fc8181"]

        if chart_type == "line":
            if isinstance(y_values[0], list):  # Multiple series
                for i, series in enumerate(y_values):
                    label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
                    ax.plot(
                        x_values,
                        series,
                        marker="o",
                        color=colors[i % len(colors)],
                        label=label,
                        linewidth=2,
                        markersize=4,
                    )
                ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
            else:
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    color=colors[0],
                    linewidth=2,
                    markersize=4,
                )

        elif chart_type == "bar":
            ax.bar(
                x_values, y_values, color=colors[0], edgecolor="white", linewidth=0.5
            )

        elif chart_type == "scatter":
            ax.scatter(
                x_values, y_values, c=colors[0], s=50, edgecolor="white", linewidth=0.5
            )

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        if x_label:
            ax.set_xlabel(x_label, fontsize=11)
        if y_label:
            ax.set_ylabel(y_label, fontsize=11)

        ax.grid(True, alpha=0.2, color="white")
        plt.tight_layout()

        # Save to base64
        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return jsonify({"success": True, "image": img_base64})

    except Exception as e:
        print(f"Chart error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/ai/clear", methods=["POST"])
def clear_history():
    data = request.json or {}
    session_id = data.get("sessionId", "default")
    if session_id in sessions:
        del sessions[session_id]  # Fully remove, don't just empty
    return jsonify({"success": True})


# --- Web Search (Tavily API) ---
_search_cache = {}  # {query_hash: {"result": ..., "time": timestamp}}
SEARCH_CACHE_TTL = 300  # 5 minutes
SEARCH_CACHE_MAX = 200


@app.route("/ai/knowledge/search", methods=["POST"])
def knowledge_search():
    """Search the web using Tavily API for chemistry/science queries."""
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        return jsonify({"error": "TAVILY_API_KEY not configured"}), 500

    data = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    search_depth = data.get("search_depth", "basic")
    max_results = min(data.get("max_results", 5), 10)
    topic = data.get("topic", "general")

    # Check cache
    cache_key = hashlib.md5(
        f"{query}:{search_depth}:{max_results}".encode()
    ).hexdigest()
    now = time()
    if cache_key in _search_cache:
        cached = _search_cache[cache_key]
        if now - cached["time"] < SEARCH_CACHE_TTL:
            print(f"🔍 Search cache hit: {query[:50]}", flush=True)
            return jsonify(cached["result"])

    # Evict expired entries
    expired = [
        k for k, v in _search_cache.items() if now - v["time"] > SEARCH_CACHE_TTL
    ]
    for k in expired:
        del _search_cache[k]
    if len(_search_cache) >= SEARCH_CACHE_MAX:
        oldest = min(_search_cache, key=lambda k: _search_cache[k]["time"])
        del _search_cache[oldest]

    try:
        print(f"🔍 Web search: {query[:80]} (depth={search_depth})", flush=True)
        resp = httpx.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {tavily_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "topic": topic,
                "include_answer": True,
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        tavily_data = resp.json()

        result = {
            "success": True,
            "answer": tavily_data.get("answer", ""),
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500],
                }
                for r in tavily_data.get("results", [])[:max_results]
            ],
            "query": query,
        }

        # Cache result
        _search_cache[cache_key] = {"result": result, "time": now}
        print(f"✅ Search returned {len(result['results'])} results", flush=True)
        return jsonify(result)

    except httpx.HTTPStatusError as e:
        print(f"❌ Tavily API error: {e.response.status_code}", flush=True)
        return jsonify({"error": f"Search API error: {e.response.status_code}"}), 502
    except httpx.TimeoutException:
        print("❌ Tavily API timeout", flush=True)
        return jsonify({"error": "Search timed out"}), 504
    except Exception as e:
        print(f"❌ Search error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/ai/transcribe", methods=["POST"])
def transcribe_audio():
    """Transcribe audio using OpenAI Whisper API"""
    if not client.api_key:
        return jsonify({"error": "API key not set"}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    try:

        # Save to temp file (OpenAI needs a file-like object with a name)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=f, response_format="text"
            )

        os.unlink(tmp_path)  # Clean up temp file

        return jsonify({"text": transcript})

    except Exception as e:
        print(f"Transcription error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MOLECULAR ORBITAL VISUALIZATION (PySCF)
# ============================================================================


@app.route("/ai/molden/orbital", methods=["POST"])
def evaluate_orbital():
    """
    Evaluate a molecular orbital from a molden file on a 3D grid using PySCF.
    Uses cached mol object and AO grid for fast repeated evaluations.
    """
    import traceback

    data = request.json
    cache_key_str = data.get("cacheKey")
    molden_content = data.get("moldenContent")
    orbital_index = data.get("orbitalIndex", 0)
    grid_size = data.get("gridSize", 50)
    padding = data.get("padding", 4.0)
    use_binary = data.get("binary", False)

    try:
        # Resolve cache: prefer cacheKey (no re-upload), fall back to moldenContent
        cache = None
        if cache_key_str:
            cache = get_molden_cache_by_key(cache_key_str)
        if cache is None:
            if not molden_content:
                return (
                    jsonify({"error": "No moldenContent or valid cacheKey provided"}),
                    400,
                )
            cache = get_or_create_molden_cache(molden_content, grid_size, padding)

        n_mo = cache["n_mo"]
        if orbital_index < 0 or orbital_index >= n_mo:
            return (
                jsonify(
                    {
                        "error": f"Orbital index {orbital_index} out of range (0-{n_mo-1})"
                    }
                ),
                400,
            )

        # MO computation via torch (GPU-accelerated) or numpy fallback
        if cache["ao_tensor"] is not None:
            mo_values = (
                torch.matmul(
                    cache["ao_tensor"], cache["coeff_tensor"][:, orbital_index]
                )
                .cpu()
                .numpy()
            )
        else:
            mo_values = cache["ao_values"] @ cache["mo_coeff"][:, orbital_index]
        mo_values = mo_values.reshape(cache["grid_shape"])

        homo_index = cache["homo_index"]
        lumo_index = cache["lumo_index"]
        orbital_type = compute_orbital_type(orbital_index, homo_index, lumo_index)

        gm = cache["grid_meta"]
        sp = gm["spacing"]

        # Encode volume data
        if use_binary:
            volume_data, data_format = encode_orbital_binary(
                mo_values.flatten(), compress=True
            )
        else:
            volume_data = mo_values.flatten().tolist()
            data_format = "json"

        # Build orbital list for frontend
        orbitals_info = [
            {
                "index": i,
                "energy": float(cache["mo_energy"][i]),
                "occupation": float(cache["mo_occ"][i]),
                "spin": "Alpha",
            }
            for i in range(n_mo)
        ]

        return jsonify(
            {
                "success": True,
                "volumeData": volume_data,
                "dataFormat": data_format,
                "gridInfo": {
                    "origin": {
                        "x": gm["origin"][0],
                        "y": gm["origin"][1],
                        "z": gm["origin"][2],
                    },
                    "dimensions": gm["dimensions"],
                    "spacing": sp,
                    "vectors": {
                        "x": [sp[0], 0, 0],
                        "y": [0, sp[1], 0],
                        "z": [0, 0, sp[2]],
                    },
                },
                "minValue": float(mo_values.min()),
                "maxValue": float(mo_values.max()),
                "orbitalIndex": orbital_index,
                "orbitalType": orbital_type,
                "energy": float(cache["mo_energy"][orbital_index]),
                "occupation": float(cache["mo_occ"][orbital_index]),
                "homoIndex": homo_index,
                "lumoIndex": lumo_index,
                "numOrbitals": n_mo,
                "orbitals": orbitals_info,
                "comment": f"MO {orbital_index + 1} ({orbital_type})",
            }
        )

    except ImportError as e:
        return (
            jsonify(
                {
                    "error": "PySCF not installed. Install with: pip install pyscf",
                    "details": str(e),
                }
            ),
            500,
        )
    except Exception as e:
        print(f"❌ Molden orbital evaluation error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/ai/molden/info", methods=["POST"])
def get_molden_info():
    """
    Get information about orbitals in a molden file without computing the grid.
    Also primes the cache so subsequent orbital requests are fast.
    """
    import traceback

    data = request.json
    molden_content = data.get("moldenContent")
    grid_size = data.get("gridSize", 50)
    padding = data.get("padding", 4.0)

    if not molden_content:
        return jsonify({"error": "No molden content provided"}), 400

    try:
        # Use cache helper — this also primes the AO grid for later orbital requests
        cache = get_or_create_molden_cache(molden_content, grid_size, padding)

        mol = cache["mol"]
        mo_energy = cache["mo_energy"]
        mo_occ = cache["mo_occ"]
        n_mo = cache["n_mo"]
        homo_index = cache["homo_index"]
        lumo_index = cache["lumo_index"]
        n_occ = int(sum(mo_occ) // 2)

        # Get atom info
        atoms = []
        for i in range(mol.natm):
            symbol = mol.atom_symbol(i)
            coord = mol.atom_coord(i)
            atoms.append(
                {
                    "element": symbol,
                    "x": coord[0] * BOHR_TO_ANG,
                    "y": coord[1] * BOHR_TO_ANG,
                    "z": coord[2] * BOHR_TO_ANG,
                }
            )

        orbitals = [
            {
                "index": i,
                "energy": float(mo_energy[i]),
                "occupation": float(mo_occ[i]),
                "spin": "Alpha",
            }
            for i in range(n_mo)
        ]

        gap = None
        if homo_index >= 0 and lumo_index >= 0 and lumo_index < n_mo:
            gap = float(mo_energy[lumo_index] - mo_energy[homo_index])

        return jsonify(
            {
                "success": True,
                "numAtoms": mol.natm,
                "numOrbitals": n_mo,
                "numOccupied": n_occ,
                "homoIndex": homo_index,
                "lumoIndex": lumo_index,
                "homoLumoGap": gap,
                "orbitals": orbitals,
                "atoms": atoms,
            }
        )

    except ImportError as e:
        return (
            jsonify(
                {
                    "error": "PySCF not installed. Install with: pip install pyscf",
                    "details": str(e),
                }
            ),
            500,
        )
    except Exception as e:
        print(f"❌ Molden info error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/ai/molden/prepare", methods=["POST"])
def prepare_molden():
    """
    Upload molden content once, compute and cache AO grid, return a cacheKey.
    Subsequent /orbital and /orbital-batch requests can use cacheKey instead of
    re-uploading the full molden content, saving significant bandwidth on remote backends.
    """
    import traceback

    data = request.json
    molden_content = data.get("moldenContent")
    grid_size = data.get("gridSize", 50)
    padding = data.get("padding", 4.0)

    if not molden_content:
        return jsonify({"error": "No molden content provided"}), 400

    try:
        cache = get_or_create_molden_cache(molden_content, grid_size, padding)

        content_hash = hashlib.sha256(molden_content.encode()).hexdigest()[:16]
        cache_key = f"{content_hash}_{grid_size}_{padding}"

        n_mo = cache["n_mo"]
        homo_index = cache["homo_index"]
        lumo_index = cache["lumo_index"]
        mo_energy = cache["mo_energy"]
        mo_occ = cache["mo_occ"]

        orbitals = [
            {
                "index": i,
                "energy": float(mo_energy[i]),
                "occupation": float(mo_occ[i]),
                "spin": "Alpha",
            }
            for i in range(n_mo)
        ]

        gap = None
        if homo_index >= 0 and lumo_index >= 0 and lumo_index < n_mo:
            gap = float(mo_energy[lumo_index] - mo_energy[homo_index])

        print(
            f"✅ Molden prepared: {n_mo} MOs, cache_key={cache_key}, device={cache['torch_device']}"
        )

        return jsonify(
            {
                "success": True,
                "cacheKey": cache_key,
                "numOrbitals": n_mo,
                "homoIndex": homo_index,
                "lumoIndex": lumo_index,
                "homoLumoGap": gap,
                "orbitals": orbitals,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/ai/molden/orbital-batch", methods=["POST"])
def evaluate_orbital_batch():
    """
    Compute multiple (or all) orbitals in a single request.
    Streams results as NDJSON (one JSON object per line) so the frontend can
    update a progress bar and cache each orbital as it arrives.

    Optimizations over naive approach:
    - Accepts cacheKey to avoid re-uploading molden content
    - Single batch torch.matmul on GPU computes ALL orbitals at once
    - Parallel zlib compression via ThreadPoolExecutor
    """
    import traceback
    from concurrent.futures import ThreadPoolExecutor

    data = request.json
    cache_key_str = data.get("cacheKey")
    molden_content = data.get("moldenContent")
    grid_size = data.get("gridSize", 50)
    padding = data.get("padding", 4.0)
    orbital_indices = data.get("orbitalIndices")  # None = all
    mesh_mode = data.get("meshMode", False)  # If true, return mesh instead of volume
    iso_value = data.get("isoValue", 0.02)  # Isovalue for mesh mode

    try:
        # Resolve cache: prefer cacheKey (no re-upload), fall back to moldenContent
        cache = None
        if cache_key_str:
            cache = get_molden_cache_by_key(cache_key_str)
        if cache is None:
            if not molden_content:
                return Response(
                    orjson.dumps(
                        {
                            "type": "error",
                            "error": "No moldenContent or valid cacheKey provided",
                        }
                    ).decode()
                    + "\n",
                    status=400,
                    mimetype="application/x-ndjson",
                    headers={"Access-Control-Allow-Origin": "*"},
                )
            cache = get_or_create_molden_cache(molden_content, grid_size, padding)
    except Exception as e:
        traceback.print_exc()
        return Response(
            orjson.dumps({"type": "error", "error": str(e)}).decode() + "\n",
            status=500,
            mimetype="application/x-ndjson",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    n_mo = cache["n_mo"]
    homo_index = cache["homo_index"]
    lumo_index = cache["lumo_index"]

    # Determine orbital order: HOMO/LUMO first, then expand outward
    if orbital_indices is None:
        ordered = []
        added = set()
        for idx in [homo_index, lumo_index]:
            if 0 <= idx < n_mo and idx not in added:
                ordered.append(idx)
                added.add(idx)
        center = homo_index if homo_index >= 0 else n_mo // 2
        for r in range(1, n_mo):
            for idx in [center - r, center + r]:
                if 0 <= idx < n_mo and idx not in added:
                    ordered.append(idx)
                    added.add(idx)
        orbital_indices = ordered

    gm = cache["grid_meta"]
    sp = gm["spacing"]

    def generate():
        # First line: metadata
        meta = {
            "type": "meta",
            "meshMode": mesh_mode,
            "numOrbitals": n_mo,
            "totalRequested": len(orbital_indices),
            "homoIndex": homo_index,
            "lumoIndex": lumo_index,
            "gridInfo": {
                "origin": {
                    "x": gm["origin"][0],
                    "y": gm["origin"][1],
                    "z": gm["origin"][2],
                },
                "dimensions": gm["dimensions"],
                "spacing": sp,
                "vectors": {
                    "x": [sp[0], 0, 0],
                    "y": [0, sp[1], 0],
                    "z": [0, 0, sp[2]],
                },
            },
        }
        yield orjson.dumps(meta).decode() + "\n"

        # === Batch compute: GPU torch.matmul or numpy fallback ===
        try:
            if cache["ao_tensor"] is not None:
                indices_tensor = torch.tensor(orbital_indices, dtype=torch.long)
                coeff_subset = cache["coeff_tensor"][
                    :, indices_tensor
                ]  # (n_ao, n_requested)
                all_mo_values = (
                    torch.matmul(cache["ao_tensor"], coeff_subset).cpu().numpy()
                )
            else:
                # Numpy fallback: batch matrix multiply
                coeff_subset = cache["mo_coeff"][:, orbital_indices]
                all_mo_values = cache["ao_values"] @ coeff_subset
            # all_mo_values shape: (n_grid_points, n_requested)
        except Exception as e:
            yield orjson.dumps(
                {"type": "error", "error": f"Batch MO compute failed: {e}"}
            ).decode() + "\n"
            return

        grid_shape = cache["grid_shape"]
        mo_energy = cache["mo_energy"]
        mo_occ = cache["mo_occ"]

        # === Parallel compression + streaming ===
        def process_one(i):
            """Process a single orbital: compress volume or compute mesh."""
            mo_flat = all_mo_values[:, i].copy()
            mo_3d = mo_flat.reshape(grid_shape)
            idx = orbital_indices[i]

            base = {
                "type": "orbital",
                "orbitalIndex": idx,
                "minValue": float(mo_3d.min()),
                "maxValue": float(mo_3d.max()),
                "orbitalType": compute_orbital_type(idx, homo_index, lumo_index),
                "energy": float(mo_energy[idx]),
                "occupation": float(mo_occ[idx]),
            }

            if mesh_mode:
                mesh = compute_orbital_mesh(mo_flat, grid_shape, gm, isovalue=iso_value)
                base["dataFormat"] = "mesh"
                base["mesh"] = mesh
            else:
                encoded, data_format = encode_orbital_binary(mo_flat, compress=True)
                base["volumeData"] = encoded
                base["dataFormat"] = data_format

            return base

        # Process in batches of 4 for parallel compression while maintaining streaming
        batch_size = 4
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for batch_start in range(0, len(orbital_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(orbital_indices))
                futures = [
                    pool.submit(process_one, i) for i in range(batch_start, batch_end)
                ]
                for future in futures:
                    try:
                        result = future.result()
                        yield orjson.dumps(result).decode() + "\n"
                    except Exception as e:
                        yield orjson.dumps(
                            {"type": "error", "error": str(e)}
                        ).decode() + "\n"

    return Response(
        generate(),
        mimetype="application/x-ndjson",
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ============================================================================
# REMOTE FILE SYSTEM ACCESS (SSH/SFTP)
# ============================================================================

import paramiko
import stat as stat_module
from threading import Lock

# Store active SFTP connections per session
sftp_connections = (
    {}
)  # {session_id: {"client": SSHClient, "sftp": SFTPClient, "host": str}}
sftp_lock = Lock()


def cleanup_sftp_connection(session_id):
    """Close and remove SFTP connection"""
    with sftp_lock:
        if session_id in sftp_connections:
            try:
                conn = sftp_connections[session_id]
                if conn.get("sftp"):
                    conn["sftp"].close()
                if conn.get("client"):
                    conn["client"].close()
            except:
                pass
            del sftp_connections[session_id]


@app.route("/api/remote/connect", methods=["POST"])
def remote_connect():
    """Connect to remote host via SSH/SFTP"""
    data = request.json
    session_id = data.get("sessionId")
    host = data.get("host")
    port = data.get("port", 22)
    username = data.get("username")
    password = data.get("password")
    key_file = data.get("keyFile")  # Base64 encoded private key (optional)

    if not all([session_id, host, username]):
        return jsonify({"error": "Missing required fields"}), 400

    # Close existing connection if any
    cleanup_sftp_connection(session_id)

    try:
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect with password or key
        if key_file:
            # Decode base64 key
            import io

            key_bytes = base64.b64decode(key_file)
            key = paramiko.RSAKey.from_private_key(io.StringIO(key_bytes.decode()))
            client.connect(host, port=port, username=username, pkey=key, timeout=10)
        elif password:
            client.connect(
                host, port=port, username=username, password=password, timeout=10
            )
        else:
            return jsonify({"error": "Must provide password or keyFile"}), 400

        # Open SFTP session
        sftp = client.open_sftp()

        # Get home directory
        home_dir = sftp.normalize(".")

        # Store connection
        with sftp_lock:
            sftp_connections[session_id] = {
                "client": client,
                "sftp": sftp,
                "host": host,
                "username": username,
                "connected_at": time(),
            }

        return jsonify(
            {
                "success": True,
                "host": host,
                "username": username,
                "homeDir": home_dir,
                "message": f"Connected to {username}@{host}",
            }
        )

    except Exception as e:
        cleanup_sftp_connection(session_id)
        return jsonify({"error": str(e)}), 500


@app.route("/api/remote/disconnect", methods=["POST"])
def remote_disconnect():
    """Disconnect from remote host"""
    data = request.json
    session_id = data.get("sessionId")

    cleanup_sftp_connection(session_id)
    return jsonify({"success": True})


@app.route("/api/remote/list", methods=["POST"])
def remote_list():
    """List files in remote directory"""
    data = request.json
    session_id = data.get("sessionId")
    path = data.get("path", ".")

    with sftp_lock:
        if session_id not in sftp_connections:
            return jsonify({"error": "Not connected"}), 401

        sftp = sftp_connections[session_id]["sftp"]

    try:
        items = []
        for entry in sftp.listdir_attr(path):
            is_dir = stat_module.S_ISDIR(entry.st_mode)
            items.append(
                {
                    "name": entry.filename,
                    "path": (
                        f"{path}/{entry.filename}" if path != "." else entry.filename
                    ),
                    "isDir": is_dir,
                    "size": entry.st_size if not is_dir else 0,
                    "modified": entry.st_mtime,
                    "permissions": oct(entry.st_mode)[-3:],
                }
            )

        # Sort: folders first, then files
        items.sort(key=lambda x: (not x["isDir"], x["name"].lower()))

        return jsonify({"success": True, "items": items, "path": path})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/remote/read", methods=["POST"])
def remote_read():
    """Read remote file content"""
    data = request.json
    session_id = data.get("sessionId")
    path = data.get("path")

    if not path:
        return jsonify({"error": "Path required"}), 400

    with sftp_lock:
        if session_id not in sftp_connections:
            return jsonify({"error": "Not connected"}), 401

        sftp = sftp_connections[session_id]["sftp"]

    try:
        # Read file
        with sftp.file(path, "r") as f:
            content = f.read().decode("utf-8")

        # Get file stats
        stat = sftp.stat(path)

        return jsonify(
            {
                "success": True,
                "content": content,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "filename": path.split("/")[-1],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/remote/write", methods=["POST"])
def remote_write():
    """Write content to remote file"""
    data = request.json
    session_id = data.get("sessionId")
    path = data.get("path")
    content = data.get("content", "")

    if not path:
        return jsonify({"error": "Path required"}), 400

    with sftp_lock:
        if session_id not in sftp_connections:
            return jsonify({"error": "Not connected"}), 401

        sftp = sftp_connections[session_id]["sftp"]

    try:
        # Write file
        with sftp.file(path, "w") as f:
            f.write(content.encode("utf-8"))

        return jsonify({"success": True, "message": "File saved"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/remote/delete", methods=["POST"])
def remote_delete():
    """Delete remote file or directory"""
    data = request.json
    session_id = data.get("sessionId")
    path = data.get("path")
    is_dir = data.get("isDir", False)

    if not path:
        return jsonify({"error": "Path required"}), 400

    with sftp_lock:
        if session_id not in sftp_connections:
            return jsonify({"error": "Not connected"}), 401

        sftp = sftp_connections[session_id]["sftp"]

    try:
        if is_dir:
            sftp.rmdir(path)
        else:
            sftp.remove(path)

        return jsonify({"success": True, "message": "Deleted successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/remote/mkdir", methods=["POST"])
def remote_mkdir():
    """Create remote directory"""
    data = request.json
    session_id = data.get("sessionId")
    path = data.get("path")

    if not path:
        return jsonify({"error": "Path required"}), 400

    with sftp_lock:
        if session_id not in sftp_connections:
            return jsonify({"error": "Not connected"}), 401

        sftp = sftp_connections[session_id]["sftp"]

    try:
        sftp.mkdir(path)
        return jsonify({"success": True, "message": "Directory created"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/remote/status", methods=["POST"])
def remote_status():
    """Get connection status"""
    data = request.json
    session_id = data.get("sessionId")

    with sftp_lock:
        if session_id in sftp_connections:
            conn = sftp_connections[session_id]
            return jsonify(
                {
                    "connected": True,
                    "host": conn["host"],
                    "username": conn["username"],
                    "connectedAt": conn["connected_at"],
                }
            )
        else:
            return jsonify({"connected": False})


print(f"Torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
