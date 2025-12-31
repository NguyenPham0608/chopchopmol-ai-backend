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
    return f"""You are ChopChopMol's AI assistant. Execute commands immediately - don't ask clarifying questions if the user provided enough info.

    STATE:
    - Molecule: {str(state.get('atomCount', 0)) + ' atoms' if state.get('hasAtoms') else 'None'}
    - Selected: {state.get('selectedCount', 0)} atoms {('[' + ','.join(map(str, state.get('selectedIndices', []))) + ']') if state.get('selectedCount', 0) > 0 else ''}
    - Fragments: {len(state.get('fragments', []))} {dumps(state.get('fragments', []))}
    - Axis: {'atoms ' + str(state.get('axisAtoms', [])[0]) + '-' + str(state.get('axisAtoms', [])[1]) if state.get('hasAxis') and len(state.get('axisAtoms', [])) == 2 else 'None'}
    - Frames: {state.get('frameCount', 0)}{' (current: ' + str(state.get('currentFrame', 0)) + ')' if state.get('frameCount', 0) > 1 else ''}
    - MACE cache: {'Yes (' + str(state.get('maceFrameCount', 0)) + ' frames)' if state.get('hasMaceCache') else 'No'}

    INDEXING: User sees 1-based, you use 0-based. "atom 5" = index 4.

    TOOLS:
    Selection: select_atoms, clear_selection, select_all_atoms, select_atoms_by_element, select_connected
    Editing: add_atom, change_atom_element, remove_atoms, set_bond_distance, add_hydrogens
    Transform: transform_atoms (handles axis+move in ONE call), define_axis, remove_axis
    Measure: measure_distance, measure_angle, measure_dihedral, clear_measurements, show_all_bond_lengths, remove_bond_label
    Fragments: create_fragment, isolate_selection, split_molecule (breaks bond into 2 fragments)
    Scans: rotational_scan (360° rotation), translation_scan (move along axis) - both generate frames
    Energy: calculate_energy, calculate_all_energies, optimize_geometry, get_cached_energies (use if hasMaceCache=true)
    View: reset_camera, zoom_to_fit, rotate_camera, toggle_labels, toggle_ribbon, set_style
    Files: save_image, save_xyz, load_molecule, create_file, edit_file, read_file, list_folder_files
    Info: get_molecule_info, get_atom_info, get_bonded_atoms
    Other: undo, redo, create_chart

    KEY WORKFLOWS:
    1. Torsion scan on bond X,Y: split_molecule(X-1, Y-1) → rotational_scan(axisAtom1=X-1, axisAtom2=Y-1, atomsToMove=fragment2, increment=10) → calculate_all_energies → create_chart
    2. transform_atoms: pass axisAtom1, axisAtom2, atomsToMove, and angle OR distance. Don't call define_axis first.

    MACE MODELS (ask if not specified): mace-mp-0a (fast), mace-mp-0b3 (high-pressure), mace-mpa-0 (best accuracy)

    RULES:
    - Use MINIMUM tool calls
    - After scans, ALWAYS calculate_all_energies then create_chart
    - Respond briefly (1-2 sentences) after actions
    """


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
        # Append tool results to history BEFORE building messages
        for result in tool_results["results"]:
            conversationHistory.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result["content"],
                }
            )

    t_prompt = time.time()
    systemPrompt = build_system_prompt(state)
    print(
        f"⏱️ Prompt built: {(time.time() - t_prompt) * 1000:.0f}ms, length: {len(systemPrompt)} chars",
        flush=True,
    )

    # Get last 10 messages but ensure valid tool_call pairing
    history_slice = conversationHistory[-10:]

    # Don't start with a tool message
    while history_slice and history_slice[0].get("role") == "tool":
        history_slice = history_slice[1:]

    # If first message is assistant with tool_calls, remove it (orphaned)
    while (
        history_slice
        and history_slice[0].get("role") == "assistant"
        and history_slice[0].get("tool_calls")
    ):
        history_slice = history_slice[1:]

    # Collect all tool_call_ids from assistant messages
    valid_tool_call_ids = set()
    for msg in history_slice:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                valid_tool_call_ids.add(tc.get("id"))

    # Collect all tool_call_ids that have responses
    responded_tool_call_ids = set()
    for msg in history_slice:
        if msg.get("role") == "tool":
            responded_tool_call_ids.add(msg.get("tool_call_id"))

    # Remove orphaned tool messages
    history_slice = [
        msg
        for msg in history_slice
        if msg.get("role") != "tool" or msg.get("tool_call_id") in valid_tool_call_ids
    ]

    # Remove assistant messages with tool_calls missing responses
    history_slice = [
        msg
        for msg in history_slice
        if not (msg.get("role") == "assistant" and msg.get("tool_calls"))
        or all(
            tc.get("id") in responded_tool_call_ids for tc in msg.get("tool_calls", [])
        )
    ]

    messages = [{"role": "system", "content": systemPrompt}] + history_slice
    total_tokens_est = sum(len(m.get("content", "")) // 4 for m in messages)
    print(
        f"📊 Messages: {len(messages)}, estimated tokens: {total_tokens_est}",
        flush=True,
    )

    def generate():
        t0 = time.time()
        print(f"🚀 Starting OpenAI call...", flush=True)
        try:
            # Get model from request (you'll need to pass it from frontend)
            model = data.get("model", "gpt-5-mini")

            # Model-specific parameters
            call_params = {
                "model": model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "stream": True,
            }

            # GPT-5.x models support reasoning_effort and verbosity
            if model.startswith("gpt-5"):
                call_params["max_completion_tokens"] = 1024
                call_params["reasoning_effort"] = "low"
                call_params["verbosity"] = "low"
            else:
                # GPT-4.1 models don't support reasoning params
                call_params["max_tokens"] = 1024

            stream = client.chat.completions.create(**call_params)

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
