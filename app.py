from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
import uuid

app = Flask(__name__)
CORS(
    app,
    origins=[
        "https://chopchopmol.com",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
)

# Store sessions in memory (use Redis for production scaling)
sessions = {}

# Tool definitions - must match frontend FUNCTIONS
TOOLS = [
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
                    "x": {"type": "number", "description": "X coordinate"},
                    "y": {"type": "number", "description": "Y coordinate"},
                    "z": {"type": "number", "description": "Z coordinate"},
                    "bondToSelected": {
                        "type": "boolean",
                        "description": "If true, add atom bonded to the selected atom",
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
            "description": "Select all atoms of a specific element type",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {
                        "type": "string",
                        "description": "Element symbol (e.g., 'C', 'N', 'O', 'H')",
                    },
                    "add": {
                        "type": "boolean",
                        "description": "Add to current selection",
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
            "name": "rotate_molecule",
            "description": "Rotate selected atoms around the defined axis by given angle",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {
                        "type": "number",
                        "description": "Rotation angle in degrees (-180 to 180)",
                    }
                },
                "required": ["angle"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_molecule",
            "description": "Move selected atoms along the defined axis by given distance in angstroms",
            "parameters": {
                "type": "object",
                "properties": {
                    "distance": {
                        "type": "number",
                        "description": "Distance to translate in angstroms",
                    }
                },
                "required": ["distance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_atom_element",
            "description": "Change the element type of selected atoms",
            "parameters": {
                "type": "object",
                "properties": {
                    "element": {"type": "string", "description": "New element symbol"}
                },
                "required": ["element"],
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
            "description": "Measure and display the distance between 2 selected atoms",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_angle",
            "description": "Measure angle between 3 atoms (middle atom is vertex)",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "measure_dihedral",
            "description": "Measure dihedral/torsion angle between 4 atoms",
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
            "description": "Create a fragment from currently selected atoms",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "isolate_selection",
            "description": "Isolate selected atoms/fragment to view separately",
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
            "description": "Show or hide atom labels",
            "parameters": {
                "type": "object",
                "properties": {
                    "show": {
                        "type": "boolean",
                        "description": "True to show, false to hide",
                    },
                    "showIndices": {
                        "type": "boolean",
                        "description": "Show atom numbers instead of elements",
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
            "description": "Select atoms directly bonded to currently selected atoms",
            "parameters": {"type": "object", "properties": {}},
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
                    "opacity": {"type": "number", "description": "Transparency 0-1"},
                    "atomSize": {
                        "type": "number",
                        "description": "Atom size multiplier 0.1-3",
                    },
                    "backgroundColor": {
                        "type": "string",
                        "description": "Background color hex",
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
            "description": "Export the molecule as XYZ file format",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_molecule",
            "description": "Search PubChem database for a molecule by name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Molecule name to search"}
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

SYSTEM_PROMPT = """You are an AI assistant for ChopChopMol, a 3D molecular editor.

STATE:
- Molecule: {atom_info}
- Selected: {selected_info}
- Fragments: {fragments_info}
- Axis: {axis_info}
- Protein: {protein_info}

Atom indices are 0-based internally but 1-based for users. Refer to atom 0 as atom 1.

CRITICAL RULES FOR TRANSFORMATIONS:
To rotate or translate atoms, you MUST call functions in this EXACT order:
1. select_atoms([atom1, atom2]) - select 2 atoms that define the axis direction
2. define_axis() - creates the rotation/translation axis
3. select_atoms([atoms to move]) - select the atoms you want to transform
4. rotate_molecule(angle) OR translate_molecule(distance)

EXAMPLE: "rotate atoms 5,6,7 by 30 degrees around axis from atoms 0 to 1"
1. select_atoms({{indices: [0, 1]}})
2. define_axis()
3. select_atoms({{indices: [5, 6, 7]}})
4. rotate_molecule({{angle: 30}})

Always call ALL required functions in sequence. Be concise. Use multiple functions when needed!
After all tool calls, respond with 1-2 sentences max summarizing what was done. No need to explain steps."""


def build_system_prompt(state):
    return SYSTEM_PROMPT.format(
        atom_info=(
            f"{state.get('atomCount', 0)} atoms"
            if state.get("hasAtoms")
            else "None loaded"
        ),
        selected_info=f"{state.get('selectedCount', 0)} atoms {state.get('selectedIndices', [])}",
        fragments_info=f"{len(state.get('fragments', []))} fragments",
        axis_info="DEFINED" if state.get("hasAxis") else "NOT defined",
        protein_info="Yes" if state.get("hasRibbon") else "No",
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ai/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("sessionId") or str(uuid.uuid4())
    message = data.get("message", "")
    state = data.get("state", {})
    tool_results = data.get("toolResults")  # Results from frontend function executions

    # Get API key from env (set in Render dashboard)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "API key not configured on server"}), 500

    client = OpenAI(api_key=api_key)

    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = {"history": []}

    session = sessions[session_id]

    # If this is a new message (not tool results), add to history
    if tool_results is None:
        session["history"].append({"role": "user", "content": message})

    # Build messages
    messages = [{"role": "system", "content": build_system_prompt(state)}]
    messages.extend(session["history"][-10:])  # Last 10 messages

    # If we have tool results, add them
    if tool_results:
        # The last assistant message should have had tool_calls
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
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )

        msg = response.choices[0].message

        # Check if there are tool calls
        if msg.tool_calls:
            # Return tool calls for frontend to execute
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in msg.tool_calls
            ]

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

        # No tool calls - final response
        if msg.content:
            session["history"].append({"role": "assistant", "content": msg.content})

        return jsonify({"sessionId": session_id, "content": msg.content, "done": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ai/clear", methods=["POST"])
def clear_history():
    data = request.json
    session_id = data.get("sessionId")
    if session_id and session_id in sessions:
        sessions[session_id] = {"history": []}
    return jsonify({"success": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
