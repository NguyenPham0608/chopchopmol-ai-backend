from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI
import os
import orjson
import tempfile
import numpy as np
from ase import Atoms
import base64
from io import BytesIO

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
        _mace_calculators[model_id] = mace_mp(
            model=model_url, default_dtype="float32", device="cpu"
        )
    return _mace_calculators[model_id]


def dumps(obj):
    return orjson.dumps(obj).decode()


app = Flask(__name__)
CORS(app, resources={r"/ai/*": {"origins": "*"}}, supports_credentials=False)

# Store sessions in memory
sessions = {}

# Global OpenAI client - reuses TCP connection
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Tools schema - exact copy from original buildToolsSchema()

TOOLS_JSON = """[
  {"type":"function","function":{"name":"create_file","description":"Create a new file in the open folder. AI can later edit files it creates.","parameters":{"type":"object","properties":{"filename":{"type":"string","description":"Filename with extension (e.g., 'molecule.xyz', 'notes.txt')"},"content":{"type":"string","description":"File content"}},"required":["filename"]}}},
  {"type":"function","function":{"name":"edit_file","description":"Edit a file that the AI previously created. Cannot edit user's original files.","parameters":{"type":"object","properties":{"filename":{"type":"string","description":"Filename to edit"},"content":{"type":"string","description":"New file content"}},"required":["filename","content"]}}},
  {"type":"function","function":{"name":"split_molecule","description":"Split a molecule into two fragments by breaking the bond between two atoms. The two atoms must be bonded. If the atoms are part of a ring (cycle), splitting is not possible.","parameters":{"type":"object","properties":{"atom1":{"type":"integer","description":"Index of the first atom (0-based)"},"atom2":{"type":"integer","description":"Index of the second atom (0-based). Must be bonded to atom1."}},"required":["atom1","atom2"]}}},
  {"type":"function","function":{"name":"read_file","description":"Read contents of a file in the open folder","parameters":{"type":"object","properties":{"filename":{"type":"string","description":"Filename to read"}},"required":["filename"]}}},
  {"type":"function","function":{"name":"list_folder_files","description":"List all files in the currently open folder","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"select_atoms","description":"Select atoms by their indices. Use this before any transformation.","parameters":{"type":"object","properties":{"indices":{"type":"array","items":{"type":"integer"},"description":"Array of atom indices to select"},"add":{"type":"boolean","description":"If true, add to current selection; if false, replace selection"}},"required":["indices"]}}},
  {"type":"function","function":{"name":"add_atom","description":"Add a new atom at a specific position or bonded to a selected atom","parameters":{"type":"object","properties":{"element":{"type":"string","description":"Element symbol (e.g., 'C', 'H', 'O')"},"x":{"type":"number","description":"X coordinate (optional if bonding to selected atom)"},"y":{"type":"number","description":"Y coordinate"},"z":{"type":"number","description":"Z coordinate"},"bondToSelected":{"type":"boolean","description":"If true, add atom bonded to the selected atom at typical bond length"}},"required":["element"]}}},
  {"type":"function","function":{"name":"set_bond_distance","description":"Set exact distance between two selected atoms (in Angstroms)","parameters":{"type":"object","properties":{"distance":{"type":"number","description":"Target distance in Angstroms"}},"required":["distance"]}}},
  {"type":"function","function":{"name":"add_hydrogens","description":"Automatically add missing hydrogen atoms to the entire molecule (standard valence rules, neutral pH assumption).","parameters":{"type":"object","properties":{"pH":{"type":"number","description":"Optional pH value for protonation state (default 7.4). Use for pH-aware addition.","default":7.4}},"required":[]}}},
  {"type":"function","function":{"name":"clear_selection","description":"Clear all atom selections","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"select_all_atoms","description":"Select all atoms in the molecule","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"select_atoms_by_element","description":"Select all atoms of a specific element type (e.g., C, N, O, H). Much faster than selecting by indices.","parameters":{"type":"object","properties":{"element":{"type":"string","description":"Element symbol (e.g., 'C' for carbon, 'N' for nitrogen, 'O' for oxygen, 'H' for hydrogen)"},"add":{"type":"boolean","description":"If true, add to current selection; if false, replace selection"}},"required":["element"]}}},
  {"type":"function","function":{"name":"define_axis","description":"Define rotation/translation axis from 2 selected atoms. MUST have exactly 2 atoms selected first.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"remove_axis","description":"Remove the currently defined axis","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"transform_atoms","description":"Rotate or translate atoms around an axis defined by two atoms. Use this for ALL rotation and translation requests.","parameters":{"type":"object","properties":{"axisAtom1":{"type":"integer","description":"First atom index defining the axis (0-indexed)"},"axisAtom2":{"type":"integer","description":"Second atom index defining the axis (0-indexed)"},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Array of atom indices to transform (0-indexed)"},"angle":{"type":"number","description":"Rotation angle in degrees (use this OR distance, not both)"},"distance":{"type":"number","description":"Translation distance in angstroms (use this OR angle, not both)"}},"required":["axisAtom1","axisAtom2","atomsToMove"]}}},
  {"type":"function","function":{"name":"change_atom_element","description":"Change the element type of selected atoms (e.g., change Carbon to Nitrogen)","parameters":{"type":"object","properties":{"element":{"type":"string","description":"New element symbol (e.g., 'C', 'N', 'O', 'H', 'S', 'P')"}},"required":["element"]}}},
  {"type":"function","function":{"name":"rotational_scan","description":"Perform a rotational scan: rotate a fragment around an axis in increments through a full 360° rotation, generating frames that can be played with the frame slider. Useful for conformational analysis.","parameters":{"type":"object","properties":{"axisAtom1":{"type":"integer","description":"First atom index defining the rotation axis (0-indexed)"},"axisAtom2":{"type":"integer","description":"Second atom index defining the rotation axis (0-indexed)"},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Array of atom indices to rotate (0-indexed). Typically a fragment."},"increment":{"type":"number","description":"Rotation increment in degrees (e.g., 10 for 36 frames, 15 for 24 frames, 30 for 12 frames)"}},"required":["axisAtom1","axisAtom2","atomsToMove","increment"]}}},
  {"type":"function","function":{"name":"remove_atoms","description":"Delete the currently selected atoms from the molecule","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"measure_distance","description":"Measure and display the distance between 2 atoms. Select exactly 2 atoms first.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"measure_angle","description":"Measure and display the angle between 3 atoms. Select exactly 3 atoms (middle atom is vertex).","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"measure_dihedral","description":"Measure dihedral/torsion angle between 4 atoms. Select exactly 4 atoms in order.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"clear_measurements","description":"Remove all distance/angle measurement labels","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"create_fragment","description":"Create a fragment (group) from currently selected atoms for easier manipulation","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"isolate_selection","description":"Isolate selected atoms/fragment to view and edit them separately","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"reset_camera","description":"Reset camera to default view position","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"zoom_to_fit","description":"Zoom camera to fit the entire molecule in view","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"rotate_camera","description":"Rotate camera view around the molecule","parameters":{"type":"object","properties":{"angle":{"type":"number","description":"Rotation angle in degrees"}},"required":["angle"]}}},
  {"type":"function","function":{"name":"toggle_labels","description":"Show or hide atom labels (element symbols or indices)","parameters":{"type":"object","properties":{"show":{"type":"boolean","description":"True to show, false to hide"},"showIndices":{"type":"boolean","description":"If true, show atom numbers instead of elements"}},"required":["show"]}}},
  {"type":"function","function":{"name":"select_connected","description":"Select atoms directly bonded to currently selected atoms (one bond away)","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"get_bonded_atoms","description":"Get bond information for specified atoms. Returns which atoms are bonded to the queried atoms and their element types. Use this to understand molecular connectivity.","parameters":{"type":"object","properties":{"indices":{"type":"array","items":{"type":"integer"},"description":"Array of atom indices to check bonds for. If not provided, uses currently selected atoms."}}}}},
  {"type":"function","function":{"name":"toggle_ribbon","description":"Toggle ribbon view for protein structures","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"set_style","description":"Change visual appearance of the molecule","parameters":{"type":"object","properties":{"roughness":{"type":"number","description":"Surface roughness 0-1"},"metalness":{"type":"number","description":"Metallic look 0-1"},"opacity":{"type":"number","description":"Transparency 0-1 (1=solid)"},"atomSize":{"type":"number","description":"Atom size multiplier 0.1-3"},"backgroundColor":{"type":"string","description":"Background color as hex e.g. #000000"}}}}},
  {"type":"function","function":{"name":"save_image","description":"Save a screenshot of the current view as PNG","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"translation_scan","description":"Perform a translation scan: move a fragment along an axis in increments, generating frames that can be played with the frame slider. Useful for dissociation curves or pulling fragments apart.","parameters":{"type":"object","properties":{"axisAtom1":{"type":"integer","description":"First atom index defining the axis direction (0-indexed)"},"axisAtom2":{"type":"integer","description":"Second atom index defining the axis direction (0-indexed). Fragment moves in direction from atom1 to atom2."},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Array of atom indices to translate (0-indexed)"},"totalDistance":{"type":"number","description":"Total translation distance in angstroms (default: 3)"},"increment":{"type":"number","description":"Step size in angstroms (default: 0.2)"}},"required":["axisAtom1","axisAtom2","atomsToMove"]}}},
  {"type":"function","function":{"name":"save_xyz","description":"Save molecule as XYZ file. If a rotational scan was performed (multiple frames), automatically exports ALL frames as a trajectory. Can save to downloads or to the local file explorer folder.","parameters":{"type":"object","properties":{"filename":{"type":"string","description":"Output filename (default: auto-generated with timestamp)"},"allFrames":{"type":"boolean","description":"If false, only export current frame even if multiple frames exist. Default is true."},"saveToLocal":{"type":"boolean","description":"If true, save to the open local folder in file explorer instead of downloading. Requires a folder to be open."}}}}},
  {"type":"function","function":{"name":"load_molecule","description":"Search and load a molecule by name from PubChem database","parameters":{"type":"object","properties":{"name":{"type":"string","description":"Molecule name e.g. caffeine, aspirin"}},"required":["name"]}}},
  {"type":"function","function":{"name":"get_molecule_info","description":"Get information about the loaded molecule","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"get_atom_info","description":"Get detailed info about specific atoms","parameters":{"type":"object","properties":{"indices":{"type":"array","items":{"type":"integer"},"description":"Atom indices to get info for"}},"required":["indices"]}}},
  {"type":"function","function":{"name":"undo","description":"Undo the last action","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"redo","description":"Redo the last undone action","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"show_all_bond_lengths","description":"Show bond length labels for ALL bonds in the molecule at once","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"remove_bond_label","description":"Remove bond length label(s). Specify atom1 and atom2 to remove a specific label, or set all:true to remove all labels","parameters":{"type":"object","properties":{"atom1":{"type":"integer","description":"First atom index"},"atom2":{"type":"integer","description":"Second atom index"},"all":{"type":"boolean","description":"Set true to remove all bond labels"}},"required":[]}}},
  {"type":"function","function":{"name":"calculate_energy","description":"Calculate the potential energy using MACE ML potential. IMPORTANT: Ask user which model to use first if not specified.","parameters":{"type":"object","properties":{"model":{"type":"string","enum":["mace-mp-0a","mace-mp-0b3","mace-mpa-0"],"description":"MACE model to use"}},"required":["model"]}}},
  {"type":"function","function":{"name":"calculate_all_energies","description":"Calculate energy for ALL frames using MACE. IMPORTANT: Ask user which model to use first if not specified.","parameters":{"type":"object","properties":{"model":{"type":"string","enum":["mace-mp-0a","mace-mp-0b3","mace-mpa-0"],"description":"MACE model to use"}},"required":["model"]}}},
  {"type":"function","function":{"name":"optimize_geometry","description":"Optimize molecular geometry using MACE ML potential. IMPORTANT: Ask user which model to use first if not specified.","parameters":{"type":"object","properties":{"model":{"type":"string","enum":["mace-mp-0a","mace-mp-0b3","mace-mpa-0"],"description":"MACE model to use"},"fmax":{"type":"number","description":"Force convergence threshold in eV/Å (default: 0.05)"},"maxSteps":{"type":"integer","description":"Maximum optimization steps (default: 100)"}},"required":["model"]}}},
  {"type":"function","function":{"name":"create_chart","description":"Create a chart/graph to visualize data. The chart will be displayed in the chat. Use for energy profiles, scan results, or any numerical data.","parameters":{"type":"object","properties":{"type":{"type":"string","enum":["line","bar","scatter"],"description":"Chart type (default: line)"},"title":{"type":"string","description":"Chart title"},"xLabel":{"type":"string","description":"X-axis label"},"yLabel":{"type":"string","description":"Y-axis label"},"x":{"type":"array","items":{"type":"number"},"description":"X-axis values"},"y":{"type":"array","items":{"type":"number"},"description":"Y-axis values"},"labels":{"type":"array","items":{"type":"string"},"description":"Labels for multiple series (optional)"}},"required":["x","y"]}}},
  {"type":"function","function":{"name":"get_cached_energies","description":"Get the cached MACE energy results from the last calculate_all_energies call. Use this to plot or analyze energy data WITHOUT recalculating. Returns the same data as calculate_all_energies if cache exists.","parameters":{"type":"object","properties":{}}}}
]"""

TOOLS = orjson.loads(TOOLS_JSON)


def build_system_prompt(state):
    # Exact copy of systemPrompt from original aiAgent.js
    return f"""You are an AI assistant for ChopChopMol, a 3D molecular editor.

STATE:
- Molecule: {str(state.get('atomCount', 0)) + ' atoms' if state.get('hasAtoms') else 'None loaded'}
- Selected: {state.get('selectedCount', 0)} atoms {('[' + ','.join(map(str, state.get('selectedIndices', []))) + ']') if state.get('selectedCount', 0) > 0 else ''}
- Fragments: {len(state.get('fragments', []))} fragments and list of fragments and atoms: {dumps(state.get('fragments', []))} also it is zero based
- Axis: {'DEFINED from atom ' + str(state.get('axisAtoms', [])[0]+1) + ' to atom ' + str(state.get('axisAtoms', [])[1]+1) + ' (0-based indices: ' + str(state.get('axisAtoms', [])[0]) + ' and ' + str(state.get('axisAtoms', [])[1]) + ')' if state.get('hasAxis') and len(state.get('axisAtoms', [])) == 2 else 'NOT defined'}
- Protein: {'Yes' if state.get('hasRibbon') else 'No'}
- Bond Labels: {len(state.get('bondLabels', []))} labels on bonds {dumps(state.get('bondLabels', []))}
- Frames: {state.get('frameCount', 0)} frames loaded{' (current: ' + str(state.get('currentFrame', 0) + 1) + ')' if state.get('frameCount', 0) > 1 else ''}
Make sure that when talking about atoms, what you see is 0-based but what the user sees is 1-based, so refer to atom 0 as atom 1 and so on.

You are an AI assistant for ChopChopMol, a 3D molecular editor.

CRITICAL: Use the MINIMUM number of tool calls needed. Many functions are self-contained.
- transform_atoms: Handles axis definition + rotation/translation in ONE call. Do NOT separately call define_axis or select_atoms first.
- rotational_scan/translation_scan: Define their own axis. Do NOT call define_axis first.
- select_atoms_by_element: Selects by element directly. Don't loop through indices.

Only chain multiple tools when truly necessary (e.g., select atoms THEN delete them).

=== TRANSFORMATIONS ===
Use transform_atoms for ALL rotation/translation requests. It handles everything in one call.

Example: "translate fragment 2 by 3 angstroms along axis from atoms 3 to 6"
→ transform_atoms({{axisAtom1: 2, axisAtom2: 5, atomsToMove: [fragment 2's atoms], distance: 3}})

Example: "rotate atoms 5,6,7 by 45 degrees around the bond between atoms 1 and 2"
→ transform_atoms({{axisAtom1: 0, axisAtom2: 1, atomsToMove: [4, 5, 6], angle: 45}})

ALL AVAILABLE FUNCTIONS:
=== SELECTION ===
- select_atoms: Select atoms by indices array. Use add:true to add to selection, add:false to replace.
- clear_selection: Deselect all atoms
- select_all_atoms: Select every atom in the molecule
- select_atoms_by_element: Select all atoms of a specific element (e.g., "C", "N", "O", "H")
- select_connected: Expand selection to all atoms bonded/connected to currently selected atoms

=== BUILDING & EDITING ===
- add_atom: Add a new atom. Specify element and x,y,z coordinates, OR set bondToSelected:true to place near selected atom
- change_atom_element: Change selected atoms to a different element type
- remove_atoms: Delete all currently selected atoms
- set_bond_distance: Set exact distance (in Å) between 2 selected atoms

=== TRANSFORMATIONS (require axis) ===
- define_axis: Create rotation/translation axis from 2 selected atoms (MUST select 2 atoms first)
- remove_axis: Remove the current axis
- rotate_molecule: Rotate selected atoms around axis by angle in degrees (-180 to 180)
- translate_molecule: Move selected atoms along axis by distance in angstroms
- to "translate/rotate atoms in a fragment", you need to create the axis, then un-select all the axis atoms and select the atoms you want to move. Then you can perform the translation/rotation. Remember, a fragment is the same as a group of atoms.
- examples: "translate fragment 4 by 3 angstroms on the axis formed when connecting atoms 2 and 6.", "translate atoms [2,4,5,7,3] by 3 angstroms on the axis formed when connecting atoms 4 and 3."

=== MEASUREMENTS ===
- measure_distance: Create distance label between 2 selected atoms
- measure_angle: Create angle label for 3 selected atoms (middle atom is vertex)
- measure_dihedral: Create dihedral/torsion angle label for 4 selected atoms
- clear_measurements: Remove all measurement labels
- show_all_bond_lengths: Show bond length labels for ALL bonds in the molecule
- remove_bond_label: Remove bond label(s). Use atom1/atom2 for specific bond, or all:true to remove all

=== FRAGMENTS ===
- create_fragment: Group selected atoms into a fragment for easier manipulation
- isolate_selection: Isolate selected atoms/fragment to view separately
- split_molecule: Split molecule into 2 fragments by breaking a bond between 2 atoms. Returns error if atoms form a ring.

=== VIEW & CAMERA ===
- reset_camera: Reset camera to default position
- zoom_to_fit: Zoom camera to fit entire molecule in view
- rotate_camera: Rotate camera view by angle in degrees
- toggle_labels: Show/hide atom labels. Use show:true/false, showIndices:true for atom numbers
- toggle_ribbon: Toggle ribbon view for proteins (only works if protein loaded)

=== STYLE ===
- set_style: Change appearance. Options: roughness (0-1), metalness (0-1), opacity (0-1), atomSize (0.1-3), backgroundColor (hex like "#000000")

=== FILE OPERATIONS ===
- save_image: Save screenshot as PNG
- save_xyz: Export molecule as XYZ file. Automatically exports ALL frames if a rotational scan was performed. Options:
  - filename: custom filename
  - saveToLocal: true to save to open folder in file explorer (instead of downloading)
  - allFrames: false to export only current frame
- load_molecule: Search PubChem database by molecule name (e.g., "caffeine", "aspirin")

=== ANALYSIS ===
- rotational_scan: Generate a 360° rotational scan of atoms around an axis. Specify axis atoms, atoms to rotate, and increment (e.g., 10° = 36 frames). Results play in the frame slider.
  Example: "do a rotational scan of fragment 2 around the bond between atoms 3-4 with 15 degree increments"
  → rotational_scan(axisAtom1: 2, axisAtom2: 3, atomsToMove: [fragment 2's atoms], increment: 15)

=== FILE EXPLORER ===
- create_file: Create a new file in the open folder. You can edit files you create.
- edit_file: Edit a file you previously created (cannot edit user's original files)
- read_file: Read contents of any file in the folder
- list_folder_files: List all files in the open folder

Note: You can only edit files that YOU created, not the user's original files. This protects user data.

=== COMPOUND OPERATIONS ===
Some requests require chaining multiple functions together. Execute them in sequence:

**Rotational scan on a bond** (e.g., "rotational scan bond 6,7" or "scan around bond 3-4"):
1. First call split_molecule(atom1, atom2) to split at that bond → returns fragment1 and fragment2
2. Then call rotational_scan(axisAtom1: atom1, axisAtom2: atom2, atomsToMove: fragment2, increment: 10)
   - Use the fragment containing atom2 as atomsToMove
   - Default increment is 10° unless specified

**Examples:**
- "rotational scan bond 6,7" → split_molecule(atom1:5, atom2:6) then rotational_scan(axisAtom1:5, axisAtom2:6, atomsToMove:[fragment2 atoms], increment:10)
- translation_scan: Generate a translation scan moving atoms along an axis. Specify axis atoms, atoms to move, total distance (default 3Å), and increment (default 0.2Å). Results play in the frame slider.
  Example: "translation scan of fragment 2 along bond 3-4 for 5 angstroms with 0.1 step"
  → translation_scan(axisAtom1: 3, axisAtom2: 4, atomsToMove: [fragment 2's atoms], totalDistance: 5, increment: 0.1)
- "scan around atoms 3 and 4 with 15 degree steps" → split_molecule(atom1:2, atom2:3) then rotational_scan(axisAtom1:2, axisAtom2:3, atomsToMove:[fragment2], increment:15)
- "rotate fragment around bond 1-2" → split_molecule first, then rotational_scan

When user mentions "bond X,Y" or "bond X-Y", always split first, then use the resulting fragments for the scan.

=== ENERGY & OPTIMIZATION ===
- calculate_energy: Calculate potential energy using MACE ML potential (returns eV, kcal/mol, forces)
- optimize_geometry: Geometry optimization to minimize energy. Optional: fmax (convergence, default 0.05), maxSteps (default 100)

**MACE WORKFLOW BEST PRACTICES:**
- ALWAYS check state.hasMaceCache before recalculating energies
- If hasMaceCache is true, use get_cached_energies instead of calculate_all_energies
- When user asks to "plot energy" or "show energy profile" after a calculation, use get_cached_energies
- MACE results are auto-saved to file explorer as mace_batch_TIMESTAMP.extxyz or mace_energy_TIMESTAMP.extxyz
- You can read saved MACE files with read_file if needed

=== ENERGY & OPTIMIZATION (MACE ML) ===
**IMPORTANT: When user requests energy calculation or optimization, ALWAYS ask which MACE model they want to use BEFORE running the calculation, unless they already specified one.**

Available MACE Models:
1. **mace-mp-0a** - Original foundation model. Fast, good general accuracy. Best for: quick calculations, general materials.
2. **mace-mp-0b3** - Improved high-pressure stability and reference energies. Best for: high-pressure systems, better energy comparisons.
3. **mace-mpa-0** - Highest accuracy, trained on MPTrj + sAlex. Best for: production calculations, when accuracy matters most.

When user says "calculate energy" or "optimize", respond with something like:
"Which MACE model would you like to use?
- **mace-mp-0a** - Fast, good general accuracy
- **mace-mp-0b3** - Better for high-pressure, improved references  
- **mace-mpa-0** - Highest accuracy (recommended for final results)"

Then use their choice in the tool call.

- calculate_energy: Get potential energy of CURRENT frame (requires model parameter)
- calculate_all_energies: Calculate energy for ALL frames (requires model parameter)
- optimize_geometry: Geometry optimization (requires model parameter)
- get_cached_energies: Retrieve cached results WITHOUT recalculating

=== INFO ===
- get_molecule_info: Get atom count, element breakdown, selection status
- get_atom_info: Get coordinates and element type for specific atom indices

=== VISUALIZATION ===
- create_chart: Generate a chart (line/bar/scatter) and display it in chat. Use for energy profiles, scan results, any data visualization.
  Example: After calculate_all_energies, plot energy vs frame with create_chart(x: [0,1,2...], y: [energies], title: "Energy Profile", xLabel: "Frame", yLabel: "Energy (eV)")

=== UNDO/REDO ===
- undo: Undo last action
- redo: Redo last undone action

Atom indices are 0-based internally. Use the FEWEST tool calls possible. Most operations need just ONE function.
After tool calls, respond with 1-2 sentences max. No explanations needed.
IMPORTANT: Do NOT call define_axis or select_atoms before transform_atoms - it handles everything internally. """


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


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

    if session_id not in sessions:
        sessions[session_id] = []
    conversationHistory = sessions[session_id]

    if tool_results is None:
        conversationHistory.append({"role": "user", "content": user_message})

    t_prompt = time.time()
    systemPrompt = build_system_prompt(state)
    print(
        f"⏱️ Prompt built: {(time.time() - t_prompt) * 1000:.0f}ms, length: {len(systemPrompt)} chars",
        flush=True,
    )

    messages = [{"role": "system", "content": systemPrompt}] + conversationHistory[-10:]
    total_tokens_est = sum(len(m.get("content", "")) // 4 for m in messages)
    print(
        f"📊 Messages: {len(messages)}, estimated tokens: {total_tokens_est}",
        flush=True,
    )

    if tool_results:
        messages.append(tool_results["assistantMessage"])
        for result in tool_results["results"]:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                }
            )

    def generate():
        t0 = time.time()
        print(f"🚀 Starting OpenAI call...", flush=True)
        try:
            stream = client.chat.completions.create(
                model="gpt-5.1",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=1024,
                reasoning_effort="low",
                verbosity="low",
                stream=True,
            )

            print(f"📡 Stream created: {(time.time() - t0) * 1000:.0f}ms", flush=True)

            collected_content = ""
            tool_calls_data = {}
            first_chunk = True

            for chunk in stream:
                if first_chunk:
                    print(
                        f"🔥 OpenAI TTFT: {(time.time() - t0) * 1000:.0f}ms", flush=True
                    )
                    first_chunk = False
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

            if tool_calls_data:
                tool_calls = [
                    {
                        "id": tc["id"],
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls_data.values()
                ]
                assistant_msg = {
                    "role": "assistant",
                    "content": collected_content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls_data.values()
                    ],
                }
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

    if not atoms_data:
        return jsonify({"error": "No atoms provided"}), 400

    try:
        symbols = [a["element"] for a in atoms_data]
        positions = [[a["x"], a["y"], a["z"]] for a in atoms_data]

        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = get_mace_calculator(model_id)

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces().tolist()

        return jsonify(
            {
                "success": True,
                "energy_eV": energy,
                "energy_kcal": energy * 23.0609,  # eV to kcal/mol
                "forces": forces,
                "max_force": float(np.max(np.abs(forces))),
            }
        )
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
    from ase.optimize import BFGS
    import io

    data = request.json
    atoms_data = data.get("atoms", [])
    fmax = data.get("fmax", 0.05)  # convergence threshold
    max_steps = data.get("maxSteps", 100)

    if not atoms_data:
        return jsonify({"error": "No atoms provided"}), 400

    try:
        symbols = [a["element"] for a in atoms_data]
        positions = [[a["x"], a["y"], a["z"]] for a in atoms_data]

        atoms = Atoms(symbols=symbols, positions=positions)
        model_id = data.get("model", "mace-mp-0a")
        atoms.calc = get_mace_calculator(model_id)

        # Run optimization
        opt = BFGS(atoms, logfile=None)
        converged = opt.run(fmax=fmax, steps=max_steps)

        # Get final positions and energy
        final_positions = atoms.get_positions().tolist()
        final_energy = float(atoms.get_potential_energy())

        return jsonify(
            {
                "success": True,
                "converged": converged,
                "steps": opt.nsteps,
                "energy_eV": final_energy,
                "energy_kcal": final_energy * 23.0609,
                "positions": [
                    {"index": i, "x": p[0], "y": p[1], "z": p[2]}
                    for i, p in enumerate(final_positions)
                ],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
        positions = [[a["x"], a["y"], a["z"]] for a in first_frame]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = calc

        for i, atoms_data in enumerate(frames_data):
            positions = [[a["x"], a["y"], a["z"]] for a in atoms_data]
            atoms.set_positions(positions)  # Update positions, triggers recalc

            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            max_force = float(np.max(np.linalg.norm(forces, axis=1)))

            results.append(
                {
                    "frame": i,
                    "energy_eV": round(energy, 6),
                    "energy_kcal": round(energy * 23.0609, 4),
                    "max_force_eV_A": round(max_force, 6),
                }
            )

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
    data = request.json
    session_id = data.get("sessionId", "default")
    if session_id in sessions:
        sessions[session_id] = []
    return jsonify({"success": True})


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
