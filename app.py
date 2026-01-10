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
from io import BytesIO
from time import time
import torch

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
CORS(app, resources={r"/ai/*": {"origins": "*"}}, supports_credentials=False)

# Store sessions in memory
sessions = {}  # {session_id: {"history": [], "last_access": timestamp}}
MAX_SESSIONS = 500
SESSION_TTL = 3600  # 1 hour

# Global OpenAI client - reuses TCP connection
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
claude_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

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
  {"type":"function","function":{"name":"rotational_scan","description":"Perform a rotational scan: rotate a fragment around an axis in increments, generating frames that can be played with the frame slider. Useful for conformational analysis.","parameters":{"type":"object","properties":{"axisAtom1":{"type":"integer","description":"First atom index defining the rotation axis (0-indexed)"},"axisAtom2":{"type":"integer","description":"Second atom index defining the rotation axis (0-indexed)"},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Array of atom indices to rotate (0-indexed). Typically a fragment."},"increment":{"type":"number","description":"Rotation increment in degrees (default: 10)"},"startAngle":{"type":"number","description":"Starting angle in degrees (default: 0)"},"endAngle":{"type":"number","description":"Ending angle in degrees (default: 360)"}},"required":["axisAtom1","axisAtom2","atomsToMove"]}}},
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
  {"type":"function","function":{"name":"select_connected","description":"Select atoms directly bonded to currently selected atoms (one bond away)","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"toggle_labels","description":"Show or hide atom labels. Can show element symbols, atom indices, or both combined (e.g., 'C1', 'O2')","parameters":{"type":"object","properties":{"showElements":{"type":"boolean","description":"True to show element symbols (e.g., 'C', 'O', 'N')"},"showIndices":{"type":"boolean","description":"True to show atom numbers (e.g., '1', '2', '3')"}},"required":[]}}},
  {"type":"function","function":{"name":"get_bonded_atoms","description":"Get bond information for specified atoms. Returns which atoms are bonded to the queried atoms and their element types. Use this to understand molecular connectivity.","parameters":{"type":"object","properties":{"indices":{"type":"array","items":{"type":"integer"},"description":"Array of atom indices to check bonds for. If not provided, uses currently selected atoms."}}}}},
  {"type":"function","function":{"name":"toggle_ribbon","description":"Toggle ribbon view for protein structures","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"set_style","description":"Change visual appearance of the molecule","parameters":{"type":"object","properties":{"roughness":{"type":"number","description":"Surface roughness 0-1"},"metalness":{"type":"number","description":"Metallic look 0-1"},"opacity":{"type":"number","description":"Transparency 0-1 (1=solid)"},"atomSize":{"type":"number","description":"Atom size multiplier 0.1-3"},"backgroundColor":{"type":"string","description":"Background color as hex e.g. #000000"}}}}},
  {"type":"function","function":{"name":"save_image","description":"Save a screenshot of the current view as PNG","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"translation_scan","description":"Perform a translation scan: move a fragment along an axis in increments, generating frames. Useful for dissociation curves.","parameters":{"type":"object","properties":{"axisAtom1":{"type":"integer","description":"First atom index defining the axis direction (0-indexed)"},"axisAtom2":{"type":"integer","description":"Second atom index defining the axis direction (0-indexed)"},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Array of atom indices to translate (0-indexed)"},"startDistance":{"type":"number","description":"Starting distance in angstroms (default: 0)"},"endDistance":{"type":"number","description":"Ending distance in angstroms (default: 3)"},"increment":{"type":"number","description":"Step size in angstroms (default: 0.2)"}},"required":["axisAtom1","axisAtom2","atomsToMove"]}}},
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
  {"type":"function","function":{"name":"optimize_geometry","description":"Optimize molecular geometry using MACE ML potential. IMPORTANT: Ask user which model to use first if not specified.","parameters":{"type":"object","properties":{"model":{"type":"string","enum":["small","medium","large","mace-mpa-0"],"description":"MACE model: 'small' (fast), 'medium' (balanced), 'large' (accurate), or 'mace-mpa-0' (best for materials)"},"fmax":{"type":"number","description":"Force convergence threshold in eV/Å (default: 0.05)"},"maxSteps":{"type":"integer","description":"Maximum optimization steps (default: 100)"}},"required":["model"]}}},
  {"type":"function","function":{"name":"run_md","description":"Run molecular dynamics (MD) simulation using MACE ML potential with Langevin thermostat (NVT ensemble). Generates trajectory frames that can be played with frame slider. IMPORTANT: Ask user for temperature and model if not specified.","parameters":{"type":"object","properties":{"model":{"type":"string","enum":["small","medium","large","mace-mpa-0"],"description":"MACE model: 'small' (fast), 'medium' (balanced), 'large' (accurate)"},"temperature":{"type":"number","description":"Temperature in Kelvin (default: 300)"},"steps":{"type":"integer","description":"Number of MD steps (default: 500)"},"timestep":{"type":"number","description":"Timestep in femtoseconds (default: 1.0)"},"friction":{"type":"number","description":"Langevin friction coefficient in 1/fs (default: 0.01)"},"saveInterval":{"type":"integer","description":"Save frame every N steps (default: 10)"}},"required":["model"]}}},
  {"type":"function","function":{"name":"create_chart","description":"Create a chart/graph to visualize data. The chart will be displayed in the chat. Use for energy profiles, scan results, or any numerical data.","parameters":{"type":"object","properties":{"type":{"type":"string","enum":["line","bar","scatter"],"description":"Chart type (default: line)"},"title":{"type":"string","description":"Chart title"},"xLabel":{"type":"string","description":"X-axis label"},"yLabel":{"type":"string","description":"Y-axis label"},"x":{"type":"array","items":{"type":"number"},"description":"X-axis values"},"y":{"type":"array","items":{"type":"number"},"description":"Y-axis values"},"labels":{"type":"array","items":{"type":"string"},"description":"Labels for multiple series (optional)"}},"required":["x","y"]}}},
  {"type":"function","function":{"name":"get_cached_energies","description":"Get the cached MACE energy results from the last calculate_all_energies call. Use this to plot or analyze energy data WITHOUT recalculating. Returns the same data as calculate_all_energies if cache exists.","parameters":{"type":"object","properties":{}}}},
  {"type":"function","function":{"name":"set_dihedral_angle","description":"Set the dihedral/torsion angle between 4 selected atoms to a specific value. Rotates the fragment on the 4th atom side around the central bond (atoms 2-3). Select exactly 4 atoms in order: A-B-C-D where B-C is the rotation axis.","parameters":{"type":"object","properties":{"angle":{"type":"number","description":"Target dihedral angle in degrees (0-360)"}},"required":["angle"]}}},
  {"type":"function","function":{"name":"set_angle","description":"Set the bond angle between 3 selected atoms to a specific value. Select exactly 3 atoms in order: A-B-C where B is the vertex atom. Rotates the fragment on atom A's side.","parameters":{"type":"object","properties":{"angle":{"type":"number","description":"Target angle in degrees (0-180)"}},"required":["angle"]}}},
  {"type":"function","function":{"name":"angle_scan","description":"Perform an angle scan: rotate a fragment around an axis perpendicular to the plane formed by 3 atoms through a range of angles. B is the pivot point.","parameters":{"type":"object","properties":{"atom1":{"type":"integer","description":"First atom index (0-indexed)"},"atom2":{"type":"integer","description":"Vertex/pivot atom index (0-indexed)"},"atom3":{"type":"integer","description":"Third atom index (0-indexed)"},"atomsToMove":{"type":"array","items":{"type":"integer"},"description":"Array of atom indices to rotate (0-indexed)"},"increment":{"type":"number","description":"Step size in degrees (default: 10)"},"startAngle":{"type":"number","description":"Starting angle in degrees (default: 0)"},"endAngle":{"type":"number","description":"Ending angle in degrees (default: 360)"}},"required":["atom1","atom2","atom3"]}}}
]"""

TOOLS = orjson.loads(TOOLS_JSON)


def build_system_prompt(state):
    return f"""You are ChopChopMol's AI assistant. Execute commands immediately - don't ask clarifying questions if the user provided enough info.

    STATE:
    - Molecule: {str(state.get('atomCount', 0)) + ' atoms' if state.get('hasAtoms') else 'None'}
    - Selected: {state.get('selectedCount', 0)} atoms {('[' + ','.join(map(str, state.get('selectedIndices', []))) + ']') if state.get('selectedCount', 0) > 0 else ''}
    - Fragments: {len(state.get('fragments', []))} {dumps(state.get('fragments', []))}
    - Axis: {'atoms ' + str(state.get('axisAtoms', [])[0]) + '-' + str(state.get('axisAtoms', [])[1]) if state.get('hasAxis') and len(state.get('axisAtoms', [])) == 2 else 'None'}
    - Frames: {state.get('frameCount', 0)}{' (current: ' + str(state.get('currentFrame', 0)) + ')' if state.get('frameCount', 0) > 1 else ''}
    - Frame Energy: {energies if (energies := state.get('frameEnergies', [])) else 'no energies calculated'}
    - MACE cache: {'Yes (' + str(state.get('maceFrameCount', 0)) + ' frames)' if state.get('hasMaceCache') else 'No'}
    - Powered by {str(state.get('aiModel', 'ChopChopMol'))} AI
    - Prioritize current Frame energy array over MACE cache or use one if another is not available. If both are available, use Energy and if none are available, prompt the user to calculate energy. However we don't want to prompt the user to calculate energy if one of the energy sources exists.

    INDEXING: User sees 1-based, you use 0-based. If the user says"atom 5" internally, you use/see index 4.

    TOOLS:
    Selection: select_atoms, clear_selection, select_all_atoms, select_atoms_by_element, select_connected
    Editing: add_atom, change_atom_element, remove_atoms, set_bond_distance, set_dihedral_angle, add_hydrogens
    Transform: transform_atoms (handles axis+move in ONE call), define_axis, remove_axis
    Measure: measure_distance, measure_angle, measure_dihedral, clear_measurements, show_all_bond_lengths, remove_bond_label, set_angle
    Scans: rotational_scan, translation_scan, angle_scan - all generate frames
    Fragments: create_fragment, isolate_selection, split_molecule (breaks bond into 2 fragments)
    Energy: calculate_energy, calculate_all_energies, optimize_geometry, get_cached_energies (use if hasMaceCache=true)
    View: reset_camera, zoom_to_fit, rotate_camera, toggle_labels, toggle_ribbon, set_style
    Files: save_image, save_xyz, load_molecule, create_file, edit_file, read_file, list_folder_files
    Info: get_molecule_info, get_atom_info, get_bonded_atoms
    MD: run_md (NVT Langevin dynamics)
    Other: undo, redo, create_chart

    KEY WORKFLOWS:
    1. Torsion scan on bond X,Y: split_molecule(X-1, Y-1) → rotational_scan(axisAtom1=X-1, axisAtom2=Y-1, atomsToMove=fragment2, increment=10) → calculate_all_energies → create_chart
    2. transform_atoms: pass axisAtom1, axisAtom2, atomsToMove, and angle OR distance. Don't call define_axis first.
    3. Geometry optimization with plot: optimize_geometry → get_cached_energies → create_chart(x=step indices, y=energies)
    4. Torsion scan with plot (any order): rotational_scan → calculate_all_energies (or get_cached_energies if cache exists) → create_chart(x=angles, y=energies)
    5. Torsion scan with save (any order): rotational_scan → calculate_all_energies (or get_cached_energies) → create_file(filename="glucose_calc.json", content=JSON.dumps(energies))
    6. Torsion scan with both plot and save (any order): rotational_scan → calculate_all_energies → create_chart → get_cached_energies → create_file (reuse cache to avoid redundant calculations)
    7. MD simulation with plot: run_md(temperature=300, steps=500) → get_cached_energies → create_chart(x=time steps, y=energies or temperature)
    MACE MODELS (ask if not specified): mace-mp-0a (fast), mace-mp-0b3 (high-pressure), mace-mpa-0 (best accuracy)

    RULES:
    - Use MINIMUM tool calls
    - After scans, ALWAYS calculate_all_energies then create_chart
    - Respond briefly (1-2 sentences) after actions
    - Always use markdown formatting. Don't overuse it, but use lists, and bolding.
    - For plotting or saving energy results after a scan: Check STATE for MACE cache. If 'No', call calculate_all_energies first (ask for model if unspecified). If 'Yes', call get_cached_energies to retrieve the data without recalculating.
    - To plot results (via create_chart): Use energies from calculate_all_energies or get_cached_energies as y-values; generate x-values based on the scan parameters (e.g., for torsion scan, angles from 0 to 360 in 'increment' steps). If there is nothing in the MACE cache, check Frame Energies in STATE, and if that is empty, prompt the user to calculate energy. Make the labels energy and frame index.
    - To save energy outputs: Get energies via calculate_all_energies or get_cached_energies, then call create_file with the filename and content as JSON.dumps of the energy data (include frame indices, energies in eV and kcal, etc.).
    - Follow user instructions in the requested order, but automatically insert prerequisite steps (e.g., energy calculation before plot/save) to handle dependencies—do not fail or ask for clarification if order implies this.
    - When doing mace calculations, ALWAYS confirm with the user with what model they want to use. No exceptions.
    - After optimize_geometry, ALWAYS get_cached_energies then create_chart (energies already calculated during optimization)
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
    systemPrompt = build_system_prompt(state)
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

    # === FINAL SAFETY: Ensure every assistant with tool_calls has matching tool responses ===
    # This runs on EVERY request, not just when tool_results is present
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
                # Optional: also reconstruct assistant if completely missing — but usually not needed
        i += 1

    print(
        f"📜 Final history slice after pairing fix: {len(history_slice)} messages",
        flush=True,
    )

    messages = [{"role": "system", "content": systemPrompt}] + history_slice

    # Safe token estimation (fix for None content)
    total_tokens_est = sum(len(str(m.get("content") or "")) // 4 for m in messages)
    print(
        f"📊 Messages: {len(messages)}, estimated tokens: {total_tokens_est}",
        flush=True,
    )
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
                claude_tools = [
                    {
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "input_schema": t["function"]["parameters"],
                    }
                    for t in TOOLS
                ]
                # Repair history specifically for Claude's strict pairing requirement
                repaired_history = repair_claude_history_for_tool_pairing(history_slice)
                claude_messages = convert_to_claude_messages(repaired_history)
                call_params = {
                    "model": model,
                    "max_tokens": 8192,
                    "messages": claude_messages,
                    "system": systemPrompt,
                    "tools": claude_tools if TOOLS else None,
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
    from ase import Atoms
    from ase.optimize import BFGS
    from mace.calculators import mace_mp
    import traceback
    import numpy as np

    data = request.json
    atoms_data = data.get("atoms", [])
    fmax = data.get("fmax", 0.05)
    max_steps = data.get("maxSteps", 100)

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
        calc = mace_mp(model=mace_model, device="cpu", default_dtype="float64")
        atoms.calc = calc

        # Store trajectory frames
        trajectory_frames = []

        def observer():
            """Callback to capture each optimization step"""
            pos = atoms.get_positions().copy()
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            max_force = float(np.sqrt((forces**2).sum(axis=1).max()))

            trajectory_frames.append(
                {"positions": pos.tolist(), "energy_eV": energy, "max_force": max_force}
            )

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

        return jsonify(
            {
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
        )
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

            trajectory_frames.append(
                {
                    "positions": pos.tolist(),
                    "energy_eV": energy,
                    "kinetic_eV": kinetic,
                    "total_eV": energy + kinetic,
                    "temperature_K": temp,
                    "step": len(trajectory_frames) * save_interval,
                }
            )

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

        return jsonify(
            {
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
        )

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
