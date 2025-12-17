from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI
import os
import json

app = Flask(__name__)
CORS(app)

# Store sessions in memory
sessions = {}

# Tools schema - exact copy from original buildToolsSchema()
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file in the open folder. AI can later edit files it creates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename with extension (e.g., 'molecule.xyz', 'notes.txt')",
                    },
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file that the AI previously created. Cannot edit user's original files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename to edit"},
                    "content": {"type": "string", "description": "New file content"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "split_molecule",
            "description": "Split a molecule into two fragments by breaking the bond between two atoms. The two atoms must be bonded. If the atoms are part of a ring (cycle), splitting is not possible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "atom1": {
                        "type": "integer",
                        "description": "Index of the first atom (0-based)",
                    },
                    "atom2": {
                        "type": "integer",
                        "description": "Index of the second atom (0-based). Must be bonded to atom1.",
                    },
                },
                "required": ["atom1", "atom2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file in the open folder",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename to read"}
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_folder_files",
            "description": "List all files in the currently open folder",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_atoms",
            "description": "Select atoms by their indices. Use this before any transformation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Array of atom indices to select",
                    },
                    "add": {
                        "type": "boolean",
                        "description": "If true, add to current selection; if false, replace selection",
                    },
                },
                "required": ["indices"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_atom",
            "description": "Add a new atom at a specific position or bonded to a selected atom",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Element symbol (e.g., 'C', 'H', 'O')",
                    },
                    "x": {
                        "type": "number",
                        "description": "X coordinate (optional if bonding to selected atom)",
                    },
                    "y": {"type": "number", "description": "Y coordinate"},
                    "z": {"type": "number", "description": "Z coordinate"},
                    "bondToSelected": {
                        "type": "boolean",
                        "description": "If true, add atom bonded to the selected atom at typical bond length",
                    },
                },
                "required": ["element"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_bond_distance",
            "description": "Set exact distance between two selected atoms (in Angstroms)",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance": {
                        "type": "number",
                        "description": "Target distance in Angstroms",
                    }
                },
                "required": ["distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_hydrogens",
            "description": "Automatically add missing hydrogen atoms to the entire molecule (standard valence rules, neutral pH assumption).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pH": {
                        "type": "number",
                        "description": "Optional pH value for protonation state (default 7.4). Use for pH-aware addition.",
                        "default": 7.4,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_selection",
            "description": "Clear all atom selections",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_all_atoms",
            "description": "Select all atoms in the molecule",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_atoms_by_element",
            "description": "Select all atoms of a specific element type (e.g., C, N, O, H). Much faster than selecting by indices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Element symbol (e.g., 'C' for carbon, 'N' for nitrogen, 'O' for oxygen, 'H' for hydrogen)",
                    },
                    "add": {
                        "type": "boolean",
                        "description": "If true, add to current selection; if false, replace selection",
                    },
                },
                "required": ["element"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "define_axis",
            "description": "Define rotation/translation axis from 2 selected atoms. MUST have exactly 2 atoms selected first.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_axis",
            "description": "Remove the currently defined axis",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transform_atoms",
            "description": "Rotate or translate atoms around an axis defined by two atoms. Use this for ALL rotation and translation requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "axisAtom1": {
                        "type": "integer",
                        "description": "First atom index defining the axis (0-indexed)",
                    },
                    "axisAtom2": {
                        "type": "integer",
                        "description": "Second atom index defining the axis (0-indexed)",
                    },
                    "atomsToMove": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Array of atom indices to transform (0-indexed)",
                    },
                    "angle": {
                        "type": "number",
                        "description": "Rotation angle in degrees (use this OR distance, not both)",
                    },
                    "distance": {
                        "type": "number",
                        "description": "Translation distance in angstroms (use this OR angle, not both)",
                    },
                },
                "required": ["axisAtom1", "axisAtom2", "atomsToMove"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_atom_element",
            "description": "Change the element type of selected atoms (e.g., change Carbon to Nitrogen)",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "New element symbol (e.g., 'C', 'N', 'O', 'H', 'S', 'P')",
                    }
                },
                "required": ["element"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotational_scan",
            "description": "Perform a rotational scan: rotate a fragment around an axis in increments through a full 360° rotation, generating frames that can be played with the frame slider. Useful for conformational analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "axisAtom1": {
                        "type": "integer",
                        "description": "First atom index defining the rotation axis (0-indexed)",
                    },
                    "axisAtom2": {
                        "type": "integer",
                        "description": "Second atom index defining the rotation axis (0-indexed)",
                    },
                    "atomsToMove": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Array of atom indices to rotate (0-indexed). Typically a fragment.",
                    },
                    "increment": {
                        "type": "number",
                        "description": "Rotation increment in degrees (e.g., 10 for 36 frames, 15 for 24 frames, 30 for 12 frames)",
                    },
                },
                "required": ["axisAtom1", "axisAtom2", "atomsToMove", "increment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_atoms",
            "description": "Delete the currently selected atoms from the molecule",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_distance",
            "description": "Measure and display the distance between 2 atoms. Select exactly 2 atoms first.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_angle",
            "description": "Measure and display the angle between 3 atoms. Select exactly 3 atoms (middle atom is vertex).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_dihedral",
            "description": "Measure dihedral/torsion angle between 4 atoms. Select exactly 4 atoms in order.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_measurements",
            "description": "Remove all distance/angle measurement labels",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_fragment",
            "description": "Create a fragment (group) from currently selected atoms for easier manipulation",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "isolate_selection",
            "description": "Isolate selected atoms/fragment to view and edit them separately",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_camera",
            "description": "Reset camera to default view position",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_to_fit",
            "description": "Zoom camera to fit the entire molecule in view",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rotate_camera",
            "description": "Rotate camera view around the molecule",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "Rotation angle in degrees",
                    }
                },
                "required": ["angle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_labels",
            "description": "Show or hide atom labels (element symbols or indices)",
            "parameters": {
                "type": "object",
                "properties": {
                    "show": {
                        "type": "boolean",
                        "description": "True to show, false to hide",
                    },
                    "showIndices": {
                        "type": "boolean",
                        "description": "If true, show atom numbers instead of elements",
                    },
                },
                "required": ["show"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_connected",
            "description": "Select atoms directly bonded to currently selected atoms (one bond away)",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bonded_atoms",
            "description": "Get bond information for specified atoms. Returns which atoms are bonded to the queried atoms and their element types. Use this to understand molecular connectivity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Array of atom indices to check bonds for. If not provided, uses currently selected atoms.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_ribbon",
            "description": "Toggle ribbon view for protein structures",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_style",
            "description": "Change visual appearance of the molecule",
            "parameters": {
                "type": "object",
                "properties": {
                    "roughness": {
                        "type": "number",
                        "description": "Surface roughness 0-1",
                    },
                    "metalness": {"type": "number", "description": "Metallic look 0-1"},
                    "opacity": {
                        "type": "number",
                        "description": "Transparency 0-1 (1=solid)",
                    },
                    "atomSize": {
                        "type": "number",
                        "description": "Atom size multiplier 0.1-3",
                    },
                    "backgroundColor": {
                        "type": "string",
                        "description": "Background color as hex e.g. #000000",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_image",
            "description": "Save a screenshot of the current view as PNG",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_xyz",
            "description": "Save molecule as XYZ file",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_molecule",
            "description": "Search and load a molecule by name from PubChem database",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Molecule name e.g. caffeine, aspirin",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_molecule_info",
            "description": "Get information about the loaded molecule",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_atom_info",
            "description": "Get detailed info about specific atoms",
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Atom indices to get info for",
                    }
                },
                "required": ["indices"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo",
            "description": "Undo the last action",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "redo",
            "description": "Redo the last undone action",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def build_system_prompt(state):
    # Exact copy of systemPrompt from original aiAgent.js
    return f"""You are an AI assistant for ChopChopMol, a 3D molecular editor.

STATE:
- Molecule: {str(state.get('atomCount', 0)) + ' atoms' if state.get('hasAtoms') else 'None loaded'}
- Selected: {state.get('selectedCount', 0)} atoms {('[' + ','.join(map(str, state.get('selectedIndices', []))) + ']') if state.get('selectedCount', 0) > 0 else ''}
- Fragments: {len(state.get('fragments', []))} fragments and list of fragments and atoms: {json.dumps(state.get('fragments', []))} also it is zero based, so make sure you refer to the real fragment 0 as fragment 1 and so on. Same with atom indexes.
- Axis: {'DEFINED from atom ' + str(state.get('axisAtoms', [])[0]+1) + ' to atom ' + str(state.get('axisAtoms', [])[1]+1) + ' (0-based indices: ' + str(state.get('axisAtoms', [])[0]) + ' and ' + str(state.get('axisAtoms', [])[1]) + ')' if state.get('hasAxis') and len(state.get('axisAtoms', [])) == 2 else 'NOT defined'}
- Protein: {'Yes' if state.get('hasRibbon') else 'No'}
Make sure that when talking about atoms, what you see is 0-based but what the user sees is 1-based, so refer to atom 0 as atom 1 and so on.

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
- save_xyz: Export molecule as XYZ file
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

=== INFO ===
- get_molecule_info: Get atom count, element breakdown, selection status
- get_atom_info: Get coordinates and element type for specific atom indices

=== UNDO/REDO ===
- undo: Undo last action
- redo: Redo last undone action

Atom indices are 0-based internally. Always call ALL required functions in sequence. Be concise. Use multiple functions when needed!
After all tool calls, respond with 1-2 sentences max summarizing what was done. No need to explain steps or repeat results."""


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ai/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("sessionId", "default")
    user_message = data.get("message", "")
    state = data.get("state", {})
    tool_results = data.get("toolResults")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return (
            jsonify({"error": "API key not set. Click ⚙️ to add your OpenAI API key."}),
            500,
        )

    client = OpenAI(api_key=api_key)

    # Get or create conversationHistory for this session
    if session_id not in sessions:
        sessions[session_id] = []
    conversationHistory = sessions[session_id]

    # If new message, add to history
    if tool_results is None:
        conversationHistory.append({"role": "user", "content": user_message})

    # Build messages array - exact same as original
    systemPrompt = build_system_prompt(state)
    messages = [{"role": "system", "content": systemPrompt}] + conversationHistory[-10:]

    # If we have tool results from frontend, add them
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

    try:
        response = client.chat.completions.create(
            model="gpt-5.1-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_completion_tokens=1024,
        )

        msg = response.choices[0].message

        # If there are tool_calls, return them for frontend to execute
        if msg.tool_calls and len(msg.tool_calls) > 0:
            tool_calls = []
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )

            return jsonify(
                {
                    "sessionId": session_id,
                    "toolCalls": tool_calls,
                    "assistantMessage": {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    },
                }
            )

        # No tool calls - final content
        finalContent = msg.content
        if finalContent:
            conversationHistory.append({"role": "assistant", "content": finalContent})

        return jsonify({"sessionId": session_id, "content": finalContent, "done": True})

    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai/chat/stream", methods=["POST"])
def chat_stream():
    data = request.json
    session_id = data.get("sessionId", "default")
    user_message = data.get("message", "")
    state = data.get("state", {})
    tool_results = data.get("toolResults")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "API key not set"}), 500

    client = OpenAI(api_key=api_key)

    if session_id not in sessions:
        sessions[session_id] = []
    conversationHistory = sessions[session_id]

    if tool_results is None:
        conversationHistory.append({"role": "user", "content": user_message})

    systemPrompt = build_system_prompt(state)
    messages = [{"role": "system", "content": systemPrompt}] + conversationHistory[-10:]

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
        try:
            stream = client.chat.completions.create(
                model="gpt-5.1",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=1024,
                stream=True,
            )

            collected_content = ""
            tool_calls_data = {}

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue

                if delta.content:
                    collected_content += delta.content
                    yield f"data: {json.dumps({'type': 'text', 'content': delta.content})}\n\n"

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
                yield f"data: {json.dumps({'type': 'tool_calls', 'toolCalls': tool_calls, 'assistantMessage': assistant_msg, 'sessionId': session_id})}\n\n"
            else:
                if collected_content:
                    conversationHistory.append(
                        {"role": "assistant", "content": collected_content}
                    )
                yield f"data: {json.dumps({'type': 'done', 'sessionId': session_id})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/ai/clear", methods=["POST"])
def clear_history():
    data = request.json
    session_id = data.get("sessionId", "default")
    if session_id in sessions:
        sessions[session_id] = []
    return jsonify({"success": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
